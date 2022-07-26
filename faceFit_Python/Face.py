import math
from operator import itemgetter

import cv2
import mediapipe as mp
import numpy as np
# Mediapipe

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
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)


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
        self.tilt = {'where': '', 'angle': 0}
        self.status = {'l_e': '', 'r_e': '', 'lips': ''}
        self.centers = {'l_e': (0, 0), 'r_e': (0, 0), 'lips': (0, 0)}
        self.delta_x = 0
        self.delta_y = 0
        self.bb_p1 = (0, 0)
        self.bb_p2 = (0, 0)
        self.bb_center = (0, 0)
        self.l_e = FacePart(LEFT_EYE)
        self.r_e = FacePart(RIGHT_EYE)
        self.lips = FacePart(LIPS)

    def get_landmarks(self, image):
        with mp_face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5) as face_m:

            self.image = image
            picture = image
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
                    self.tilt = {'where': text, 'angle': angle}
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
        hull_8u = []
        for i in range(0, len(hull)):
            hull_8u.append((hull[i][0], hull[i][1]))
        mask = np.zeros(self.image.shape, dtype=self.image.dtype)
        cv2.fillConvexPoly(mask, np.int32(hull_8u), (255, 255, 255))
        return mask

    def draw(self, part):
        conn = ''
        dr_spec = ''
        if part == 'iris':
            conn = mp_face_mesh.FACEMESH_IRISES
            dr_spec = mp_drawing_styles.get_default_face_mesh_iris_connections_style()
        elif part == 'contours':
            conn = mp_face_mesh.FACEMESH_CONTOURS
            dr_spec = mp_drawing_styles.get_default_face_mesh_contours_style()
        elif part == 'tessellation':
            conn = mp_face_mesh.FACEMESH_TESSELATION
            dr_spec = mp_drawing_styles.get_default_face_mesh_tesselation_style()
        else:
            print('WRONG PART DESCRIPTOR')
        mp_drawing.draw_landmarks(
            image=self.image,
            landmark_list=self.f_lmrks,
            connections=conn,
            landmark_drawing_spec=None,
            connection_drawing_spec=dr_spec)

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
        new_range = (0, 1)
        max_range = max(new_range)
        min_range = min(new_range)
        scaled_unit = (max_range - min_range) / (np.max(v) - np.min(v))
        new_points = v * scaled_unit - np.min(v) * scaled_unit + min_range
        self.pts = new_points.tolist()

def factor_and_center(img, landmark_a, id1, id2, id3, id4):
    p1 = (int(landmark_a[id1].x * img.shape[1]), int(landmark_a[id1].y * img.shape[0]), 0)
    p2 = (int(landmark_a[id2].x * img.shape[1]), int(landmark_a[id2].y * img.shape[0]), 0)
    p3 = (int(landmark_a[id3].x * img.shape[1]), int(landmark_a[id3].y * img.shape[0]), 0)
    p4 = (int(landmark_a[id4].x * img.shape[1]), int(landmark_a[id4].y * img.shape[0]), 0)
    p4_p3 = ((p4[0] - p3[0]) ** 2 + (p4[1] - p3[1]) ** 2 + (p4[2] - p3[2]) ** 2) ** 0.5
    p2_p1 = ((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2 + (p2[2] - p1[2]) ** 2) ** 0.5
    division = p4_p3 / p2_p1
    center = find_center(np.array([p1, p2, p3, p4]))
    return division, center


def normalize(value, bounds):
    return bounds['desired']['lower'] + (value - bounds['actual']['lower']) * \
           (bounds['desired']['upper'] - bounds['desired']['lower']) / \
           (bounds['actual']['upper'] - bounds['actual']['lower'])





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

    return l_e, r_e, lips, l_center, r_center, lips_center


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
    angles, mtx_r, mtx_q, qx, qy, qz = cv2.RQDecomp3x3(r_mat)
    alpha = angles[0] * 360
    beta = angles[1] * 360
    if what == 'ref':  # if reference
        alpha = angles[0] * 360
        beta = angles[1] * 360
    else:
        alpha = int(normalize(alpha, {'actual': {'lower': -40, 'upper': 40}, 'desired': {'lower': -40, 'upper': 40}}))
        beta = int(normalize(beta, {'actual': {'lower': -15, 'upper': 12}, 'desired': {'lower': -65, 'upper': 55}}))

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