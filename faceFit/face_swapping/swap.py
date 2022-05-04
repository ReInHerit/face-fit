import math
import scipy.spatial as spatial
import logging
import cv2
import mediapipe as mp
import numpy as np
from operator import itemgetter
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh
LEFT_EYE = mp_face_mesh.FACEMESH_LEFT_EYE
LEFT_EYEBROW = mp_face_mesh.FACEMESH_LEFT_EYEBROW
LEFT_IRIS = mp_face_mesh.FACEMESH_LEFT_IRIS
RIGHT_EYE = mp_face_mesh.FACEMESH_RIGHT_EYE
RIGHT_EYEBROW = mp_face_mesh.FACEMESH_RIGHT_EYEBROW
RIGHT_IRIS = mp_face_mesh.FACEMESH_RIGHT_IRIS
LIPS = mp_face_mesh.FACEMESH_LIPS
FACE_OVAL = mp_face_mesh.FACEMESH_FACE_OVAL
img1 = cv2.imread("bradley_cooper.jpg")
img_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
mask = np.zeros_like(img_gray)
img2 = cv2.imread("jim_carrey.jpg")
img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
class FacePart:
    def __init__(self, part_group):
        self.part_group = part_group
        self.idx = []
        self.pts = []
        self.raw_pts = []
        self.get_idx()

    def get_idx(self):
        part = list(self.part_group)
        for index in part:
            self.idx.append(index[0])
            self.idx.append(index[1])
        self.idx = sorted(set(self.idx))

    def calc_pts(self, points_array):
        temp_array = []
        for i in self.idx:
            temp_array.append(points_array[i])
        self.raw_pts = temp_array
        v = np.array(temp_array)
        new_points = scale_numpy_array(v, 0, 1)
        self.pts = new_points.tolist()

class Face:
    def __init__(self, which):
        self.which = which
        self.image = []
        self.np_image = []
        self.f_lmrks = []
        self.landmarks = []
        self.points = []
        self.pix_points = []
        self.where_looks = ''
        self.alpha = 0
        self.beta = 0
        self.tilt = {'where':'', 'angle': 0}
        self.status = {'l_e':'', 'r_e':'', 'lips':''}
        self.centers = {'l_e':(0,0), 'r_e':(0,0), 'lips':(0,0)}
        self.delta_x = 0
        self.delta_y = 0
        self.bb_p1 = (0, 0)
        self.bb_p2 = (0, 0)
        self.bb_center = (0, 0)
        self.l_e = FacePart(LEFT_EYE)
        self.r_e = FacePart(RIGHT_EYE)
        self.lips = FacePart(LIPS)

    def get_landmarks(self, image):
        with  mp_face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5) as face_m:

            self.image = image
            picture = image#.astype('uint8')
            # Convert the BGR image to RGB before processing.
            result = face_m.process(cv2.cvtColor(picture, cv2.COLOR_BGR2RGB))
            if result.multi_face_landmarks:
                self.np_image = picture.copy()
                w, h, c = picture.shape
                for face_landmarks in result.multi_face_landmarks:

                    self.points = []
                    self.pix_points = []
                    self.f_lmrks = face_landmarks
                    self.landmarks = face_landmarks.landmark
                    # print('len lm:', len(face_landmarks.landmark))
                    for i in range(0, len(face_landmarks.landmark)):
                        x = face_landmarks.landmark[i].x
                        y = face_landmarks.landmark[i].y
                        z = face_landmarks.landmark[i].z
                        self.points.append([x, y, z])
                        self.pix_points.append([int(x * h), int(y * w)])
                    # calc expression
                    expression = check_expression(image, self.landmarks)
                    self.status['l_e'] = expression[0]
                    self.status['r_e'] = expression[1]
                    self.status['lips'] = expression[2]
                    self.centers['l_e'] = expression[3]
                    self.centers['r_e'] = expression[4]
                    self.centers['lips'] = expression[5]
                    # calc BBOX
                    cx_min = h
                    cy_min = w
                    cx_max = 0
                    cy_max = 0
                    for lm in self.points:
                        cx, cy = int(lm[0] * h), int(lm[1] * w)
                        if cx < cx_min:
                            cx_min = cx
                        if cy < cy_min:
                            cy_min = cy
                        if cx > cx_max:
                            cx_max = cx
                        if cy > cy_max:
                            cy_max = cy
                    self.bb_p1 = (cx_min, cy_min)
                    self.bb_p2 = (cx_max, cy_max)
                    self.delta_x = cx_max - cx_min
                    self.delta_y = cy_max - cy_min
                    self.bb_center = (int(cx_min + self.delta_x / 2), int(cy_min + self.delta_y / 2))

                    # where is looking
                    look = where_is_looking(image, self.f_lmrks, self.which)
                    self.where_looks = look[0]
                    self.alpha = look[2]
                    self.beta = look[1]
                    self.l_e.calc_pts(self.points)
                    self.r_e.calc_pts(self.points)
                    self.lips.calc_pts(self.points)
                    # tilt
                    min_a = min(self.l_e.raw_pts, key=itemgetter(1))[1]
                    max_a = max(self.l_e.raw_pts, key=itemgetter(1))[1]
                    min_b = min(self.r_e.raw_pts, key=itemgetter(1))[1]
                    max_b = max(self.r_e.raw_pts, key=itemgetter(1))[1]
                    if max_a < min_b:
                        text = 'left'
                    elif max_b < min_a:
                        text = 'right'
                    else:
                        text = 'even'
                    point1 = self.l_e.raw_pts[1]
                    point2 = self.r_e.raw_pts[1]
                    angle = math.degrees(math.atan2(-(point2[1]-point1[1]), point2[0]-point1[0])) % 360
                    self.tilt = {'where':text, 'angle': angle}
                self.np_image = np.asarray(self.np_image)

    def self_hud_mask(self):
        img1_points = self.pix_points
        # Find convex hull
        hull_index = cv2.convexHull(np.array(img1_points), returnPoints=False)
        # Create convex hull lists
        hull = []
        for i in range(0, len(hull_index)):
            hull.append(img1_points[hull_index[i][0]])
        # Calculate Mask for Seamless cloning
        hull8U = []
        for i in range(0, len(hull)):
            hull8U.append((hull[i][0], hull[i][1]))
        mask = np.zeros(self.image.shape, dtype=self.image.dtype)
        cv2.fillConvexPoly(mask, np.int32(hull8U), (255, 255, 255))
        return mask


def factor_and_center(img, landmark_a, id1, id2, id3, id4):
    p1 = (int(landmark_a[id1].x * img.shape[1]), int(landmark_a[id1].y * img.shape[0]), 0)
    p2 = (int(landmark_a[id2].x * img.shape[1]), int(landmark_a[id2].y * img.shape[0]), 0)
    p3 = (int(landmark_a[id3].x * img.shape[1]), int(landmark_a[id3].y * img.shape[0]), 0)
    p4 = (int(landmark_a[id4].x * img.shape[1]), int(landmark_a[id4].y * img.shape[0]), 0)
    division = calc_distance(p3, p4) / calc_distance(p1, p2)
    center = find_center(np.array([p1, p2, p3, p4]))
    return division, center

def calc_distance(p1, p2):
    return ((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2 + (p2[2] - p1[2]) ** 2) ** 0.5

def find_center(points_array):
    length = points_array.shape[0]
    sum_x = np.sum(points_array[:, 0])
    sum_y = np.sum(points_array[:, 1])
    return int(sum_x / length), int(sum_y / length)


def check_expression(img, landmarks):
    # l_eye
    l_division, l_center = factor_and_center(img, landmarks, 362, 263, 386, 374)
    if l_division <= 0.1:
        l_e = 'closed'
    else:
        l_e = 'opened'

    # r_eye
    r_division, r_center = factor_and_center(img, landmarks, 33, 133, 159, 145)
    if r_division <= 0.1:
        r_e = 'closed'
    else:
        r_e = 'opened'

    # Mouth
    lips_division, lips_center = factor_and_center(img, landmarks, 78, 308, 13, 14)
    if lips_division < 0.15:
        lips = 'closed'
    elif 0.15 <= lips_division < 0.4:
        lips = 'opened'
    else:
        lips = 'full opened'

    # lips_center = (int(landmarks[13].x * img.shape[1]), int(landmarks[13].y * img.shape[0]))

    return l_e, r_e, lips, l_center, r_center, lips_center


def scale_numpy_array(arr, min_v, max_v):
    new_range = (min_v, max_v)
    max_range = max(new_range)
    min_range = min(new_range)
    scaled_unit = (max_range - min_range) / (np.max(arr) - np.min(arr))
    return arr * scaled_unit - np.min(arr) * scaled_unit + min_range
def where_is_looking(img, f_landmarks, what):
    hr, wr, cr = img.shape
    face2d = []
    face3d = []
    for indx, lm in enumerate(f_landmarks.landmark):
        if indx == 33 or indx == 263 or indx == 1 or indx == 61 or indx == 291 or indx == 199:
            if indx == 1:
                nose_2d = (lm.x * wr, lm.y * hr)
                nose_3d = (lm.x * wr, lm.y * hr, lm.z * 8000)

            x1, y1 = int(lm.x * wr), int(lm.y * hr)

            # Get the 2D Coordinates
            face2d.append([x1, y1])

            # Get the 3D Coordinates
            face3d.append([x1, y1, lm.z])

    # Convert to the NumPy array
    face_2d = np.array(face2d, dtype=np.float64)
    face_3d = np.array(face3d, dtype=np.float64)
    # The camera matrix
    focal_length = 1 * wr
    cam_matrix = np.array([[focal_length, 0, hr / 2],
                           [0, focal_length, wr / 2],
                           [0, 0, 1]])
    dist_matrix = np.zeros((4, 1), dtype=np.float64)  # The Distance Matrix
    succ, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)  # Solve PnP
    r_mat, jac = cv2.Rodrigues(rot_vec)  # Get rotational matrix
    # Get angles
    angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(r_mat)
    alpha = angles[0] * 360
    beta = angles[1] * 360
    if what == 'ref':  # if reference
        alpha = angles[0] * 360
        beta = angles[1] * 360
        # gamma = angles[2] * 360
    else:  ## 'lower': -25, 'upper': 25}, 'desired': {'lower': -35, 'upper': 48}}
        alpha = int(normalize(alpha, {'actual': {'lower': -40, 'upper': 40}, 'desired': {'lower': -40, 'upper': 40}}))
        beta = int(normalize(beta, {'actual': {'lower': -25, 'upper': 25}, 'desired': {'lower': -60, 'upper': 60}}))

    # See where the user's head tilting
    if beta < -5:
        if alpha < 0:
            text = 'Looking Down Left'
        elif alpha > 15:
            text = 'Looking Up Left'
        else:
            text = "Looking Left"
    elif beta > 5:
        if alpha < 0:
            text = 'Looking Down Right'
        elif alpha > 15:
            text = 'Looking Up Right'
        else:
            text = "Looking Right"
    else:
        if alpha < 0:
            text = "Looking Down"
        elif alpha > 15:
            text = 'Looking Up'
        else:
            text = "Looking Forward"

    return [text, beta, alpha]
def hud_mask(mask_obj, masked_obj):
    # img1_warped = np.copy(masked_obj)
    img1_points = mask_obj.pix_points
    img2_points = masked_obj.pix_points
    # Find convex hull
    hull_index = cv2.convexHull(np.array(img1_points), returnPoints=False)

    # Create convex hull lists
    hull1 = []
    hull2 = []
    for i in range(0, len(hull_index)):
        hull1.append(img1_points[hull_index[i][0]])
        hull2.append(img2_points[hull_index[i][0]])

    # Calculate Mask for Seamless cloning
    hull8U = []
    for i in range(0, len(hull2)):
        hull8U.append((hull2[i][0], hull2[i][1]))

    mask = np.zeros(masked_obj.image.shape, dtype=masked_obj.image.dtype)
    cv2.fillConvexPoly(mask, np.int32(hull8U), (255, 255, 255))
    # print('qua')
    # cv2.imshow('',mask)
    # cv2.waitKey(1)
    return mask
def normalize(value, bounds):
    return bounds['desired']['lower'] + (value - bounds['actual']['lower']) * \
           (bounds['desired']['upper'] - bounds['desired']['lower']) / \
           (bounds['actual']['upper'] - bounds['actual']['lower'])
def extract_index_nparray(nparray):
    index = None
    for num in nparray[0]:
        index = num
        break
    return index

def morph2(c_obj, r_obj):

    img1_points = np.array(c_obj.pix_points, np.int32)
    img2_points = np.array(r_obj.pix_points, np.int32)


    img1 = c_obj.image
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    mask = np.zeros_like(img1_gray)
    img2 = r_obj.image
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)


    # detector = dlib.get_frontal_face_detector()
    # predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    height, width, channels = img2.shape
    img2_new_face = np.zeros((height, width, channels), np.uint8)
    convexhull1 = cv2.convexHull(img1_points)
    # cv2.polylines(img, [convexhull], True, (255, 0, 0), 3)
    cv2.fillConvexPoly(mask, convexhull1, 255)


    face_image_1 = cv2.bitwise_and(img1, img1, mask=mask)

    # Delaunay triangulation
    rect = cv2.boundingRect(convexhull1)
    subdiv = cv2.Subdiv2D(rect)
    subdiv.insert(c_obj.pix_points)
    triangles = subdiv.getTriangleList()
    triangles = np.array(triangles, dtype=np.int32)

    indexes_triangles = []
    for t in triangles:
        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])

        index_pt1 = np.where((img1_points == pt1).all(axis=1))
        index_pt1 = extract_index_nparray(index_pt1)

        index_pt2 = np.where((img1_points == pt2).all(axis=1))
        index_pt2 = extract_index_nparray(index_pt2)

        index_pt3 = np.where((img1_points == pt3).all(axis=1))
        index_pt3 = extract_index_nparray(index_pt3)

        if index_pt1 is not None and index_pt2 is not None and index_pt3 is not None:
            triangle = [index_pt1, index_pt2, index_pt3]
            indexes_triangles.append(triangle)

    # Face 2

    convexhull2 = cv2.convexHull(img2_points)


    for triangle_index in indexes_triangles:
        # Triangulation of the first face
        tr1_pt1 = c_obj.pix_points[triangle_index[0]]
        tr1_pt2 = c_obj.pix_points[triangle_index[1]]
        tr1_pt3 = c_obj.pix_points[triangle_index[2]]
        triangle1 = np.array([tr1_pt1, tr1_pt2, tr1_pt3], np.int32)

        rect1 = cv2.boundingRect(triangle1)
        (x, y, w, h) = rect1
        cropped_triangle = img1[y: y + h, x: x + w]
        cropped_tr1_mask = np.zeros((h, w), np.uint8)

        img1_points = np.array([[tr1_pt1[0] - x, tr1_pt1[1] - y],
                           [tr1_pt2[0] - x, tr1_pt2[1] - y],
                           [tr1_pt3[0] - x, tr1_pt3[1] - y]], np.int32)

        cv2.fillConvexPoly(cropped_tr1_mask, img1_points, 255)

        # # Lines space
        # cv2.line(lines_space_mask, tr1_pt1, tr1_pt2, 255)
        # cv2.line(lines_space_mask, tr1_pt2, tr1_pt3, 255)
        # cv2.line(lines_space_mask, tr1_pt1, tr1_pt3, 255)
        # lines_space = cv2.bitwise_and(img1, img1, mask=lines_space_mask)

        # Triangulation of second face
        tr2_pt1 = r_obj.pix_points[triangle_index[0]]
        tr2_pt2 = r_obj.pix_points[triangle_index[1]]
        tr2_pt3 = r_obj.pix_points[triangle_index[2]]
        triangle2 = np.array([tr2_pt1, tr2_pt2, tr2_pt3], np.int32)

        rect2 = cv2.boundingRect(triangle2)
        (x, y, w, h) = rect2

        cropped_tr2_mask = np.zeros((h, w), np.uint8)

        img2_points = np.array([[tr2_pt1[0] - x, tr2_pt1[1] - y],
                            [tr2_pt2[0] - x, tr2_pt2[1] - y],
                            [tr2_pt3[0] - x, tr2_pt3[1] - y]], np.int32)

        cv2.fillConvexPoly(cropped_tr2_mask, img2_points, 255)

        # Warp triangles
        img1_points = np.float32(img1_points)
        img2_points = np.float32(img2_points)
        M = cv2.getAffineTransform(img1_points, img2_points)
        warped_triangle = cv2.warpAffine(cropped_triangle, M, (w, h))
        warped_triangle = cv2.bitwise_and(warped_triangle, warped_triangle, mask=cropped_tr2_mask)

        # Reconstructing destination face
        img2_new_face_rect_area = img2_new_face[y: y + h, x: x + w]
        img2_new_face_rect_area_gray = cv2.cvtColor(img2_new_face_rect_area, cv2.COLOR_BGR2GRAY)
        _, mask_triangles_designed = cv2.threshold(img2_new_face_rect_area_gray, 1, 255, cv2.THRESH_BINARY_INV)
        warped_triangle = cv2.bitwise_and(warped_triangle, warped_triangle, mask=mask_triangles_designed)

        img2_new_face_rect_area = cv2.add(img2_new_face_rect_area, warped_triangle)
        img2_new_face[y: y + h, x: x + w] = img2_new_face_rect_area

    # Face swapped (putting 1st face into 2nd face)

    img2_face_mask = np.zeros_like(img2_gray)

    img2_head_mask = cv2.fillConvexPoly(img2_face_mask, convexhull2, 255)
    img2_head_mask = cv2.GaussianBlur(img2_head_mask, (35, 35), sigmaX=0, sigmaY=0)

    img2_face_mask = cv2.bitwise_not(img2_head_mask)  ##inversa
    head_mask_3chan = cv2.cvtColor(img2_head_mask, cv2.COLOR_GRAY2BGR).astype('float') / 255.
    face_mask_3chan = cv2.cvtColor(img2_face_mask, cv2.COLOR_GRAY2BGR).astype('float') / 255.
    img2_face = img2_new_face.astype('float')/255
    img2_bg = img2.astype('float')/255
    out = img2_bg * (1-head_mask_3chan) + img2_face * head_mask_3chan
    out = (out * 255).astype('uint8')
    cv2.imshow('-', out)
    # img2_new_face = cv2.bitwise_and(img2_new_face, img2_new_face, mask=img2_head_mask)

    img2_head_noface = cv2.bitwise_and(img2, img2, mask=img2_face_mask)

    result = cv2.add(img2_head_noface, img2_new_face)
    (x, y, w, h) = cv2.boundingRect(convexhull2)
    center_face2 = (int((x + x + w) / 2), int((y + y + h) / 2))
    seamlessclone = cv2.seamlessClone(out, img2, img2_head_mask, center_face2, cv2.NORMAL_CLONE)

    cv2.imshow('0', head_mask_3chan)
    cv2.imshow('1', result)
    cv2.imshow('2', img2_bg)
    cv2.imshow('3', img2_face)

    cv2.imshow('4', seamlessclone)
    cv2.waitKey(0)

    # return seamlessclone

im1 = Face('ref')
im2 = Face('ref')
im1.get_landmarks(img1)
im2.get_landmarks(img2)
morph = morph2(im1, im2)
cv2.imshow('', morph)
cv2.waitKey(0)
