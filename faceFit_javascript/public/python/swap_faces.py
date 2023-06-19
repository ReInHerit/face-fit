import math
import time
import sys
import cv2
from json import load as load_json, dumps, JSONEncoder
import base64
import os
import numpy as np
from shapely.geometry import MultiLineString
from shapely.ops import unary_union, polygonize
from scipy.spatial import Delaunay
from collections import Counter
from itertools import combinations
from flask import Flask, jsonify, request
from PIL import Image
import io
import Face_Maker as F_obj
from math import floor, ceil
from match_color import matching_color, find_noise_scratches

ref = []
ref_dict = []

if os.getenv('HOST'):
    HOST = os.getenv('HOST')
else:
    HOST = 'localhost'

ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))

# triangulation_json_path = os.path.join(ROOT_DIR, 'json', 'triangulation.json')
triangulation2_json_path = os.path.join(ROOT_DIR, 'json', 'triangulation2.json')

with open(triangulation2_json_path, 'r') as f:
    media_pipes_tris2 = load_json(f)


def get_concave_hull(points_list):  # points_list is a 2D numpy array
    # removed the Qbb option from the scipy defaults, it is much faster and equally precise without it.
    # unless your points_list are integers. see http://www.qhull.org/html/qh-optq.htm
    tri = Delaunay(points_list, qhull_options="Qc Qz Q12").simplices

    ia, ib, ic = tri[:, 0], tri[:, 1], tri[:, 2]  # indices of each of the triangles' points
    pa, pb, pc = points_list[ia], points_list[ib], points_list[ic]  # coordinates of each of the triangles' points

    a = np.sqrt((pa[:, 0] - pb[:, 0]) ** 2 + (pa[:, 1] - pb[:, 1]) ** 2)
    b = np.sqrt((pb[:, 0] - pc[:, 0]) ** 2 + (pb[:, 1] - pc[:, 1]) ** 2)
    c = np.sqrt((pc[:, 0] - pa[:, 0]) ** 2 + (pc[:, 1] - pa[:, 1]) ** 2)

    s = (a + b + c) * 0.5  # Semi-perimeter of triangle
    area = np.sqrt(s * (s - a) * (s - b) * (s - c))  # Area of triangle by Heron's formula
    edge_filter = (a * b * c / (4.0 * area) < 50)  # Radius Filter based
    edges = tri[edge_filter]  # Filter the edges

    edges = [tuple(sorted(combo)) for e in edges for combo in combinations(e, 2)]

    count = Counter(edges)  # count occurrences of each edge
    edges = [e for e, c in count.items() if c == 1]  # keep only edges that appear one time (concave hull edges)
    # coordinates of the edges that comprise the concave hull
    edges = [(points_list[e[0]], points_list[e[1]]) for e in edges]
    # use this only if you need to return your hull points in "order" (CCW)
    ml = MultiLineString(edges)
    poly = polygonize(ml)
    hull = unary_union(list(poly))
    hull_vertices = hull.exterior.coords.xy

    vertices = [[np.int32(hull_vertices[0][n]), np.int32(hull_vertices[1][n])] for n in range(len(hull_vertices[0]))]
    vertices = np.array(vertices)
    return vertices


def warp_triangle(img1, img2, t1, t2):
    x1, y1, w1, h1 = cv2.boundingRect(np.float32([t1]))
    x2, y2, w2, h2 = cv2.boundingRect(np.float32([t2]))
    # Offset points by left top corner of the respective rectangles
    t1_rect = [((t1[i][0] - x1), (t1[i][1] - y1)) for i in range(0, 3)]
    t2_rect = [((t2[i][0] - x2), (t2[i][1] - y2)) for i in range(0, 3)]
    # Get mask by filling triangle
    mask = np.zeros((h2, w2, 3), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(t2_rect), (1.0, 1.0, 1.0), 16, 0)
    # Apply warpImage to small rectangular patches
    img1_rect = img1[y1:y1 + h1, x1:x1 + w1]
    size = (w2, h2)
    # Affine Transformation
    warp_mat = cv2.getAffineTransform(np.float32(t1_rect), np.float32(t2_rect))
    img2_rect = cv2.warpAffine(img1_rect, warp_mat, (size[0], size[1]), None, flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REFLECT_101)
    img2_rect = img2_rect * mask
    # Copy triangular region of the rectangular patch to the output image
    img2[y2:y2 + h2, x2:x2 + w2] = img2[y2:y2 + h2, x2:x2 + w2] * ((1.0, 1.0, 1.0) - mask) + img2_rect


def sort_triangles_by_distance(triangles, viewer_pos, triangles_points):
    def calculate_distance(triangle):
        # Calculate the distance between the viewer and the centroid of the triangle
        point1, point2, point3 = [triangles_points[vertex] for vertex in triangle]
        centroid = (
            (point1[0] + point2[0] + point3[0]) / 3,
            (point1[1] + point2[1] + point3[1]) / 3,
            (point1[2] + point2[2] + point3[2]) / 3
        )
        return distance(viewer_pos, centroid)

    def is_clockwise(vertices):
        # Check if the vertices of a triangle are specified in clockwise order
        v0, v1, v2 = vertices
        return ((v1[0] - v0[0]) * (v2[1] - v0[1]) - (v1[1] - v0[1]) * (v2[0] - v0[0])) < 0

    def is_front_facing(vertices, eye_pos):
        # Check if the triangle is front-facing with respect to the viewer
        v0, v1, v2 = vertices
        normal = calculate_triangle_normal(v0, v1, v2)
        view_direction = (eye_pos[0] - v0[0], eye_pos[1] - v0[1], eye_pos[2] - v0[2])
        # dot_product = normal[0] * view_direction[0] + normal[1] * view_direction[1] + normal[2] * view_direction[2]
        return dot_product(normal, view_direction) >= 0

    def calculate_triangle_normal(v0, v1, v2):
        # Calculate the normal vector of a triangle
        ux, uy, uz = v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]
        vx, vy, vz = v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]
        nx = uy * vz - uz * vy
        ny = uz * vx - ux * vz
        nz = ux * vy - uy * vx
        return nx, ny, nz

    def is_inside_triangle(point, triangle):
        # Check if a point is inside a 2D triangle using barycentric coordinates
        v0, v1, v2 = triangle
        d00 = dot_product(v0, v0)
        d01 = dot_product(v0, v1)
        d11 = dot_product(v1, v1)
        d20 = dot_product(v2, v0)
        d21 = dot_product(v2, v1)
        inv_denom = 1 / (d00 * d11 - d01 * d01)
        u = (d11 * (point[0] - v0[0]) - d01 * (point[1] - v0[1])) * inv_denom
        v = (d00 * (point[1] - v0[1]) - d01 * (point[0] - v0[0])) * inv_denom
        return (u >= 0) and (v >= 0) and (u + v <= 1)

    # def get_triangle_vertices(triangle):
    #     # Get the vertices of a triangle using their indices
    #     return [triangles_points[vertex] for vertex in triangle]

    def triangle_covers(triangle1, triangle2):
        triangle1_vertices = [triangles_points[vertex] for vertex in triangle1] #get_triangle_vertices(triangle1)
        triangle2_vertices = [triangles_points[vertex] for vertex in triangle2] #get_triangle_vertices(triangle2)

        if is_clockwise(triangle1_vertices) or not is_front_facing(triangle1_vertices, viewer_pos):
            # Triangle1 is facing away from the viewer, it cannot visually cover triangle2
            return False

        for vertex in triangle2_vertices:
            if is_inside_triangle(vertex, triangle1_vertices):
                # One of the vertices of triangle2 is inside triangle1, triangle1 visually covers triangle2
                return True
        return False

    def check_adjacent_triangles(triangle1, triangle2):
        common_vertices = set(triangle1) & set(triangle2)
        if len(common_vertices) == 2:
            if triangle_covers(triangle1, triangle2):
                return -1
            else:
                return 1
        return 0

    sorted_triangles = sorted(triangles, key=calculate_distance, reverse=True)

    for i, triangle1 in enumerate(sorted_triangles):
        for j in range(i+1, len(sorted_triangles)):
            triangle2 = sorted_triangles[j]

            result = check_adjacent_triangles(triangle1, triangle2)
            if result == -1:
                sorted_triangles[i], sorted_triangles[j] = sorted_triangles[j], sorted_triangles[i]
                break
            elif result == 1:
                break

    return sorted_triangles


def morph(c_obj, r_obj):
    cam_image, cam_points, ref_points = c_obj.image, c_obj.pix_points, r_obj['pix_points']
    mask_dilate_iter, mask_erode_iter, blur_value, offset = 10, 15, 35, 5

    ref_image = cv2.imread(r_obj['src'])

    ref_smoothed, noise = find_noise_scratches(ref_image)

    r_roi = ref_smoothed[r_obj['bb']['yMin'] - offset:r_obj['bb']['yMax'] + offset,
            r_obj['bb']['xMin'] - offset:r_obj['bb']['xMax'] + offset]
    c_roi = cam_image[c_obj.bb_p1[1] - offset:c_obj.bb_p2[1] + offset,
            c_obj.bb_p1[0] - offset:c_obj.bb_p2[0] + offset]

    cam_cc = matching_color(r_roi, c_roi)

    cam_image[c_obj.bb_p1[1] - offset:c_obj.bb_p2[1] + offset,
    c_obj.bb_p1[0] - offset:c_obj.bb_p2[0] + offset] = cam_cc.astype('float64')
    # SWAP FACE
    ref_new_face = np.zeros(ref_image.shape, np.uint8)
    dt = media_pipes_tris2  # triangles

    new_dt = sort_triangles_by_distance(dt, (0, 0, -5), c_obj.points)
    tris1 = [[cam_points[new_dt[i][j]] for j in range(3)] for i in range(len(new_dt))]
    tris2 = [[ref_points[new_dt[i][j]] for j in range(3)] for i in range(len(new_dt))]
    for i in range(0, len(tris1)):  # Apply affine transformation to Delaunay triangles
        warp_triangle(cam_image, ref_new_face, tris1[i], tris2[i])

    # GENERATE FINAL IMAGE
    concave_hull = get_concave_hull(np.array(ref_points))
    ref_gray = cv2.cvtColor(ref_image, cv2.COLOR_BGR2GRAY)
    ref_img_mask = np.zeros_like(ref_gray)
    concave_mask = cv2.fillPoly(ref_img_mask, [concave_hull], 255)
    ref_face_mask = cv2.dilate(concave_mask, None, iterations=mask_dilate_iter)
    ref_face_mask = cv2.erode(ref_face_mask, None, iterations=mask_erode_iter)
    ref_face_mask = cv2.GaussianBlur(ref_face_mask, (blur_value, blur_value), sigmaX=0, sigmaY=0)
    mid3 = cv2.moments(concave_mask)  # Find Centroid
    center = (int(mid3['m10'] / mid3['m00']), int(mid3['m01'] / mid3['m00']))
    r_face_mask_3ch = cv2.cvtColor(ref_face_mask, cv2.COLOR_GRAY2BGR).astype('float') / 255.
    out_face = (ref_new_face.astype('float') / 255)
    out_bg = ref_smoothed.astype('float') / 255
    out = out_bg * (1 - r_face_mask_3ch) + out_face * r_face_mask_3ch
    out = (out * 255).astype('uint8')
    out = cv2.add(out, noise)
    out = cv2.add(out, noise)
    output = cv2.seamlessClone(out, ref_image, ref_face_mask, center, cv2.NORMAL_CLONE)

    return output


# ------------------ UTILS ------------------
def distance(point1, point2):
    return math.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2 + (point2[2] - point1[2]) ** 2)


def dot_product(v1, v2):
    # Compute the dot product of two vectors
    return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]


def readb64(base64_string):
    idx = base64_string.find('base64,')
    base64_string = base64_string[idx + 7:]

    sbuf = io.BytesIO()

    sbuf.write(base64.b64decode(base64_string, ' /'))
    pimg = Image.open(sbuf)
    return cv2.cvtColor(np.array(pimg), cv2.COLOR_RGB2BGR)


def round_num(value):
    x = floor(value)
    if (value - x) < .50:
        return x
    else:
        return ceil(value)


app = Flask(__name__)


# ------------------ ROUTES ------------------
@app.route('/INIT_PAINTINGS', methods=['POST'])
def init():
    # POST request
    if request.method == 'POST':
        global ref_dict
        print('INIT Incoming..')
        data = request.get_json()
        ref_dict = []
        for idx, file in enumerate(data):
            ref_img = cv2.imread(os.path.join(ROOT_DIR, file))
            p_face = F_obj.Face('ref')
            p_face.get_landmarks(ref_img)
            face_dict = {'which': p_face.which, 'id': idx, 'src': file, 'points': p_face.points,
                         'expression': [p_face.status['l_e'], p_face.status['r_e'], p_face.status['lips']],
                         'pix_points': p_face.pix_points,
                         'angles': [round_num(p_face.alpha) + 90, round_num(p_face.beta) + 90, round_num(p_face.gamma)],
                         'bb': {'xMin': p_face.bb_p1[0], 'xMax': p_face.bb_p2[0], 'yMin': p_face.bb_p1[1],
                                'yMax': p_face.bb_p2[1], 'width': p_face.delta_x, 'height': p_face.delta_y,
                                'center': [p_face.bb_p1[0] + round_num(p_face.delta_x / 2),
                                           p_face.bb_p2[0] + round_num(p_face.delta_y / 2)]}}
            ref_dict.append(face_dict)
        print('REFERENCES INIT DONE')
        return jsonify(ref_dict), 200

    else:
        message = {'greeting': 'Hello from Face-Fit Flask server!'}
        return jsonify(message)


@app.route('/DATAtoPY', methods=['POST'])
def sendData():
    # POST request
    if request.method == 'POST':
        print('Incoming..')
        data = request.get_json()
        data_img = data['objs']
        user = data['user_id']
        c_image = readb64(data_img["c_face"])
        selected = data_img["selected"]

        r_obj = ref_dict[selected]
        c_image = cv2.flip(c_image, 1)
        # Create Frame Face Object
        c_obj = F_obj.Face('cam')
        c_obj.get_landmarks(c_image)
        # Select Reference Image Face Object
        head, file_name = os.path.split(r_obj['src'])
        r_obj['src'] = os.path.join(ROOT_DIR, 'images', file_name)
        # Morph the faces
        output = morph(c_obj, r_obj)
        numb = "0" + str(selected + 1) if selected <= 8 else str(selected + 1)
        morphed_file_name = 'morph_' + numb + '.png'
        path = os.path.join(ROOT_DIR, 'temp', user, 'morphs', morphed_file_name)
        write = cv2.imwrite(path, output)
        if write:
            return path, 200
    # GET request
    else:
        message = {'greeting': 'Hello from Flask!'}
        return jsonify(message)


if __name__ == "__main__":
    app.run(host=HOST, port=8050, debug=True)
