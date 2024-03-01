import os
from collections import Counter
from itertools import combinations
from json import load as load_json

import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import Delaunay
from shapely.geometry import MultiLineString
from shapely.ops import unary_union, polygonize

from FaceFit.static.assets.py.utils import dot_product, distance
from FaceFit.static.assets.py.match_color import matching_color, find_noise_scratches

BG_COLOR = (0, 0, 0)
MASK_COLOR = (255, 255, 255)
ref = []
ref_dict = []

if os.getenv('HOST'):
    HOST = os.getenv('HOST')
else:
    HOST = 'localhost'

ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))

triangulation2_json_path = os.path.join(ROOT_DIR, 'json', 'triangulation2.json')

with open(triangulation2_json_path, 'r') as f:
    media_pipes_tris2 = load_json(f)

# Create the options that will be used for ImageSegmenter
BaseOptions = mp.tasks.BaseOptions
ImageSegmenter = mp.tasks.vision.ImageSegmenter
ImageSegmenterOptions = mp.tasks.vision.ImageSegmenterOptions
VisionRunningMode = mp.tasks.vision.RunningMode
model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'hair_segmenter.tflite'))

base_options = BaseOptions(model_asset_path=model_path)
options = ImageSegmenterOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.IMAGE, output_category_mask=True)


def get_hair_mask(images):
    masks = []
    with ImageSegmenter.create_from_options(options) as segmenter:
        print('Segmenter created')
        for file in images:
            rgb_image = cv2.cvtColor(file, cv2.COLOR_BGR2RGB)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

            # Retrieve the masks for the segmented image
            segmentation_result = segmenter.segment(image)
            category_mask = segmentation_result.category_mask
            image_data = image.numpy_view()
            fg_image = np.zeros(image_data.shape, dtype=np.uint8)
            fg_image[:] = MASK_COLOR
            bg_image = np.zeros(image_data.shape, dtype=np.uint8)
            bg_image[:] = BG_COLOR
            condition2 = np.stack((category_mask.numpy_view(),) * 3, axis=-1) > 0.2
            output_image = np.where(condition2, fg_image, bg_image)
            masks.append(output_image)
        return masks


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
        # d20 = dot_product(v2, v0)
        # d21 = dot_product(v2, v1)
        inv_denom = 1 / (d00 * d11 - d01 * d01)
        u = (d11 * (point[0] - v0[0]) - d01 * (point[1] - v0[1])) * inv_denom
        v = (d00 * (point[1] - v0[1]) - d01 * (point[0] - v0[0])) * inv_denom
        return (u >= 0) and (v >= 0) and (u + v <= 1)

    def triangle_covers(triangle1, triangle2):
        triangle1_vertices = [triangles_points[vertex] for vertex in triangle1]
        triangle2_vertices = [triangles_points[vertex] for vertex in triangle2]

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
    mask_dilate_iter, mask_erode_iter, blur_value, offset = 10, 15, 35, 20

    ref_image = cv2.imread(r_obj['src'])
    images = [cam_image, ref_image]
    hair_masks = get_hair_mask(images)

    ref_smoothed, noise = find_noise_scratches(ref_image)
    cam_smoothed, _ = find_noise_scratches(cam_image)
    r_roi = ref_smoothed[r_obj['bb']['yMin'] - offset:r_obj['bb']['yMax'] + offset,
            r_obj['bb']['xMin'] - offset:r_obj['bb']['xMax'] + offset]
    c_roi = cam_smoothed[c_obj.bb_p1[1] - offset:c_obj.bb_p2[1] + offset,
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


    r_face_mask_3ch = cv2.cvtColor(ref_face_mask, cv2.COLOR_GRAY2BGR).astype('float') / 255.
    out_face = (ref_new_face.astype('float') / 255)
    out_bg = ref_smoothed.astype('float') / 255
    out = out_bg * (1 - r_face_mask_3ch) + out_face * r_face_mask_3ch
    out = (out * 255).astype('uint8')

    out = cv2.add(out, noise)
    out = cv2.add(out, noise)

    # Find Center of the polygon to place the face
    brect = cv2.boundingRect(concave_mask)
    center_of_brect = (int(brect[0] + brect[2] / 2), int(brect[1] + brect[3] / 2))

    output = cv2.seamlessClone(out, ref_image, ref_face_mask, center_of_brect, cv2.NORMAL_CLONE)
    # place ref hairs on top of the output
    hair_mask_gray = cv2.cvtColor(hair_masks[1], cv2.COLOR_BGR2GRAY)
    ret, hair_mask_gray = cv2.threshold(hair_mask_gray, 127, 255, cv2.THRESH_BINARY)
    br = cv2.boundingRect(hair_mask_gray)  # bounding rect (x,y,width,height)
    centerOfBR = (br[0] + br[2] // 2, br[1] + br[3] // 2)
    refined_output = cv2.seamlessClone(ref_image, output, hair_mask_gray, centerOfBR, cv2.NORMAL_CLONE)

    return refined_output

