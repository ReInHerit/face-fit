import math
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
import Face as F_obj
from math import floor, ceil

ref = []
ref_dict = []

if os.getenv('HOST'):
    HOST = os.getenv('HOST')
else:
    HOST = 'localhost'

ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))

triangulation_json_path = os.path.join(ROOT_DIR, 'json', 'triangulation.json')
triangulation2_json_path = os.path.join(ROOT_DIR, 'json', 'triangulation2.json')

with open(triangulation_json_path, 'r') as f:
    media_pipes_tris = load_json(f)
with open(triangulation2_json_path, 'r') as f:
    media_pipes_tris2 = load_json(f)

def find_noise_scratches(img):  # De-noising
    dst = cv2.fastNlMeansDenoisingColored(img, None, 5, 5, 5, 15)
    noise = cv2.subtract(img, dst)
    return dst, noise


# COLOR CORRECTION functions
def calculate_cdf(histogram):
    """ This method calculates the cumulative distribution function """
    # Get the cumulative sum of the elements
    cdf = histogram.cumsum()
    # Normalize the cdf
    normalized_cdf = cdf / float(cdf.max())
    return normalized_cdf


def calculate_lookup(src_cdf, ref_cdf):
    """ This method creates the lookup table """
    lookup_table = np.zeros(256)
    lookup_val = 0
    for src_pixel_val in range(len(src_cdf)):
        for ref_pixel_val in range(len(ref_cdf)):
            if ref_cdf[ref_pixel_val] >= src_cdf[src_pixel_val]:
                lookup_val = ref_pixel_val
                break
        print(src_pixel_val, ref_pixel_val)
        lookup_table[src_pixel_val] = lookup_val
    return lookup_table


def match_histograms(src_image, ref_image):
    """ This method matches the source image histogram to the reference signal """
    # Split the images into the different color channels
    src_b, src_g, src_r = cv2.split(src_image)
    ref_b, ref_g, ref_r = cv2.split(ref_image)
    # Compute the b, g, and r histograms separately
    src_hist_blue, bin_0 = np.histogram(src_b.flatten(), 256, [0, 256])
    src_hist_green, bin_1 = np.histogram(src_g.flatten(), 256, [0, 256])
    src_hist_red, bin_2 = np.histogram(src_r.flatten(), 256, [0, 256])
    ref_hist_blue, bin_3 = np.histogram(ref_b.flatten(), 256, [0, 256])
    ref_hist_green, bin_4 = np.histogram(ref_g.flatten(), 256, [0, 256])
    ref_hist_red, bin_5 = np.histogram(ref_r.flatten(), 256, [0, 256])
    # Compute the normalized cdf for the source and reference image
    src_cdf_blue = calculate_cdf(src_hist_blue)
    src_cdf_green = calculate_cdf(src_hist_green)
    src_cdf_red = calculate_cdf(src_hist_red)
    ref_cdf_blue = calculate_cdf(ref_hist_blue)
    ref_cdf_green = calculate_cdf(ref_hist_green)
    ref_cdf_red = calculate_cdf(ref_hist_red)
    # Make a separate lookup table for each color
    blue_lookup_table = calculate_lookup(src_cdf_blue, ref_cdf_blue)
    green_lookup_table = calculate_lookup(src_cdf_green, ref_cdf_green)
    red_lookup_table = calculate_lookup(src_cdf_red, ref_cdf_red)
    # Use the lookup function to transform the colors of the original source image
    blue_after_transform = cv2.LUT(src_b, blue_lookup_table)
    green_after_transform = cv2.LUT(src_g, green_lookup_table)
    red_after_transform = cv2.LUT(src_r, red_lookup_table)
    # Put the image back together
    image_after_matching = cv2.merge([blue_after_transform, green_after_transform, red_after_transform])
    image_after_matching = cv2.convertScaleAbs(image_after_matching)
    return image_after_matching


# Morph Functions
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
    # in the list below both (i, j) and (j, i) pairs are counted. The reasoning is that boundary edges appear only once
    # while interior edges twice
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


def distance(point1, point2):
    # Calculate the distance between two 3D points
    return math.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2 + (point2[2] - point1[2])**2)

def sort_triangles_by_distance(triangles, viewer_pos, triangles_points):
    # Calculate the distance between the viewer and each triangle
    distances = []
    for triangle in triangles:
        point1 = triangles_points[triangle[0]]
        point2 = triangles_points[triangle[1]]
        point3 = triangles_points[triangle[2]]
        # Calculate the distance between the viewer and the centroid of the triangle
        centroid = ((point1[0] + point2[0] + point3[0]) / 3,
                    (point1[1] + point2[1] + point3[1]) / 3,
                    (point1[2] + point2[2] + point3[2]) / 3)
        distances.append(distance(viewer_pos, centroid))

    # Sort the triangles by distance
    sorted_triangles = [triangle for _, triangle in sorted(zip(distances, triangles), reverse=True)]

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

    cam_cc = match_histograms(c_roi, r_roi)

    cam_image[c_obj.bb_p1[1] - offset:c_obj.bb_p2[1] + offset,
               c_obj.bb_p1[0] - offset:c_obj.bb_p2[0] + offset] = cam_cc.astype('float64')
    # SWAP FACE
    ref_new_face = np.zeros(ref_image.shape, np.uint8)
    dt = media_pipes_tris2  # triangles

    new_dt = sort_triangles_by_distance(dt, (0, 0, -20), c_obj.points)
    tris1 = [[cam_points[new_dt[i][j]] for j in range(3)]for i in range(len(new_dt))]
    tris2 = [[ref_points[new_dt[i][j]] for j in range(3)]for i in range(len(new_dt))]
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
    # center = [x - y for x, y in zip(center, painting_data[file_name]["center_delta"])]
    r_face_mask_3ch = cv2.cvtColor(ref_face_mask, cv2.COLOR_GRAY2BGR).astype('float') / 255.
    out_face = (ref_new_face.astype('float') / 255)
    out_bg = ref_smoothed.astype('float') / 255

    out = out_bg * (1 - r_face_mask_3ch) + out_face * r_face_mask_3ch
    out = (out * 255).astype('uint8')
    out = cv2.add(out, noise)
    out = cv2.add(out, noise)
    output = cv2.seamlessClone(out, ref_image, ref_face_mask, center, cv2.NORMAL_CLONE)

    return output


def readb64(base64_string):
    idx = base64_string.find('base64,')
    base64_string = base64_string[idx+7:]

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


@app.route('/INIT_PAINTINGS', methods=['POST'])
def init():
    # POST request
    if request.method == 'POST':
        global ref_dict
        print('INIT Incoming..')
        data = request.get_json()
        ref_dict = []
        for idx, file in enumerate(data):

            path_on_server = os.path.join(ROOT_DIR, file)

            ref_img = cv2.imread(path_on_server)
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
        #Morph the faces
        output = morph(c_obj, r_obj)
        numb = "0" + str(selected + 1) if selected <= 8 else str(selected + 1)
        morphed_file_name = 'morph_' + numb + '.png'

        path = os.path.join(ROOT_DIR, 'temp', user, 'morphs', morphed_file_name)  # path_to["morphs"] + 'morph_' + numb + '.png'

        write = cv2.imwrite(path, output)
        if write:
            return path, 200

    # GET request
    else:
        message = {'greeting': 'Hello from Flask!'}
        return jsonify(message)  # serialize and use JSON headers


if __name__ == "__main__":
    app.run(host=HOST, port=8050, debug=True)
