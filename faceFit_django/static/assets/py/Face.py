
from math import degrees, atan2
import cv2
from mediapipe import solutions as mp_solutions
import numpy as np

# Mediapipe

mp_drawing = mp_solutions.drawing_utils
mp_drawing_styles = mp_solutions.drawing_styles
mp_face_mesh = mp_solutions.face_mesh
LEFT_EYE = mp_face_mesh.FACEMESH_LEFT_EYE
RIGHT_EYE = mp_face_mesh.FACEMESH_RIGHT_EYE
LIPS = mp_face_mesh.FACEMESH_LIPS
IRISES = mp_face_mesh.FACEMESH_IRISES
CONTOURS = mp_face_mesh.FACEMESH_CONTOURS
TESSELATION = mp_face_mesh.FACEMESH_TESSELATION
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)


class Face:
    def __init__(self, which):
        self.which = which
        self.image = np.array([])
        self.f_landmarks = []
        self.landmarks = []
        self.points = []
        self.pix_points = []
        self.alpha = 0
        self.beta = 0
        self.gamma = 0
        self.status = {'l_e': '', 'r_e': '', 'lips': ''}
        self.delta_x = 0
        self.delta_y = 0
        self.bb_p1 = (0, 0)
        self.bb_p2 = (0, 0)
        self.l_e = FacePart(LEFT_EYE)
        self.r_e = FacePart(RIGHT_EYE)
        self.lips = FacePart(LIPS)

    def get_landmarks(self, image):
        with mp_face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.50) as face_m:
            self.image = image
            picture = image
            # Convert the BGR image to RGB before processing.
            result = face_m.process(cv2.cvtColor(picture, cv2.COLOR_BGR2RGB))
            if result.multi_face_landmarks:
                w, h, c = self.image.shape
                for landmarks in result.multi_face_landmarks:
                    self.points, self.pix_points = [], []
                    self.f_landmarks = landmarks
                    self.landmarks = landmarks.landmark
                    for i in range(0, len(landmarks.landmark)):
                        x, y, z = landmarks.landmark[i].x, landmarks.landmark[i].y, landmarks.landmark[i].z
                        self.points.append([x, y, z])
                        self.pix_points.append([int(x * h), int(y * w)])
                    # calc expression
                    expression = check_expression(self.pix_points)
                    self.status['l_e'] = expression[0]
                    self.status['r_e'] = expression[1]
                    self.status['lips'] = expression[2]
                    # calc BBOX
                    cx_min, cy_min, cx_max, cy_max = h, w, 0, 0
                    for point in self.points:
                        cx, cy = int(point[0] * h), int(point[1] * w)
                        cx_min = cx if cx < cx_min else cx_min
                        cx_max = cx if cx > cx_max else cx_max
                        cy_min = cy if cy < cy_min else cy_min
                        cy_max = cy if cy > cy_max else cy_max
                    self.bb_p1 = (cx_min, cy_min)
                    self.bb_p2 = (cx_max, cy_max)
                    self.delta_x = cx_max - cx_min
                    self.delta_y = cy_max - cy_min
                    # where is looking
                    self.l_e.calc_pts(self.points)
                    self.r_e.calc_pts(self.points)
                    self.lips.calc_pts(self.points)
                    self.where_is_looking()

    def where_is_looking(self):
        hr, wr, cr = self.image.shape

        face2d = [self.pix_points[index] for index in [33, 263, 1, 61, 291, 199]]
        face3d = [[self.pix_points[index][0], self.pix_points[index][1], self.points[index][2]]
                  for index in [33, 263, 1, 61, 291, 199]]
        face_2d = np.array(face2d, dtype=np.float64)
        face_3d = np.array(face3d, dtype=np.float64)
        # The camera matrix
        focal_length = 1 * wr
        cam_matrix = np.array([[focal_length, 0, hr / 2],
                               [0, focal_length, wr / 2],
                               [0, 0, 1]])
        dist_matrix = np.zeros((4, 1), dtype=np.float64)  # The Distance Matrix
        success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)  # Solve PnP
        r_mat, jac = cv2.Rodrigues(rot_vec)  # Get rotational matrix
        # Get angles alpha (pitch) and beta (yaw)
        angles, mtx_r, mtx_q, qx, qy, qz = cv2.RQDecomp3x3(r_mat)
        alpha, beta, tilt = tuple(angle * 360 for angle in angles)
        if self.which == 'cam':  # if camera
            alpha = int(normalize(alpha, {'actual': {'lower': -40, 'upper': 40}, 'desired': {'lower': -40, 'upper': 40}}))
            beta = int(normalize(beta, {'actual': {'lower': -15, 'upper': 12}, 'desired': {'lower': -65, 'upper': 55}}))
        # get gamma (roll)
        point1 = self.l_e.raw_pts[1]
        point2 = self.r_e.raw_pts[1]
        gamma = degrees(atan2(-(point2[1] - point1[1]), point2[0] - point1[0])) % 360 - 180
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def draw(self, part):
        conn, dr_spec = '', ''
        if part == 'iris':
            conn = IRISES
            dr_spec = mp_drawing_styles.get_default_face_mesh_iris_connections_style()
        elif part == 'contours':
            conn = CONTOURS
            dr_spec = mp_drawing_styles.get_default_face_mesh_contours_style()
        elif part == 'tessellation':
            conn = TESSELATION
            dr_spec = mp_drawing_styles.get_default_face_mesh_tesselation_style()
        else:
            print('WRONG PART DESCRIPTOR')
        mp_drawing.draw_landmarks(
            image=self.image,
            landmark_list=self.f_landmarks,
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
        self.raw_pts = [points_array[i] for i in self.idx]
        v = np.array(self.raw_pts)
        new_range = (0, 1)
        scaled_unit = (max(new_range) - min(new_range)) / (np.max(v) - np.min(v))
        new_points = v * scaled_unit - np.min(v) * scaled_unit + min(new_range)
        self.pts = new_points.tolist()


def aperture(pixel_points, id1, id2, id3, id4):
    p1 = (pixel_points[id1][0], pixel_points[id1][1])
    p2 = (pixel_points[id2][0], pixel_points[id2][1])
    p3 = (pixel_points[id3][0], pixel_points[id3][1])
    p4 = (pixel_points[id4][0], pixel_points[id4][1])
    p4_p3 = ((p4[0] - p3[0]) ** 2 + (p4[1] - p3[1]) ** 2) ** 0.5
    p2_p1 = ((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2) ** 0.5
    division = p4_p3 / p2_p1
    return division


def normalize(value, bounds):
    return bounds['desired']['lower'] + (value - bounds['actual']['lower']) * \
           (bounds['desired']['upper'] - bounds['desired']['lower']) / \
           (bounds['actual']['upper'] - bounds['actual']['lower'])


def check_expression(face_points):
    # l_eye
    l_gap = aperture(face_points, 362, 263, 386, 374)
    if l_gap <= 0.1:
        l_e = 'closed'
    else:
        l_e = 'opened'
    # r_eye
    r_gap = aperture(face_points, 33, 133, 159, 145)
    if r_gap <= 0.1:
        r_e = 'closed'
    else:
        r_e = 'opened'
    # Mouth
    lips_gap = aperture(face_points, 78, 308, 13, 14)
    if lips_gap < 0.15:
        lips = 'closed'
    elif 0.15 <= lips_gap < 0.4:
        lips = 'opened'
    else:
        lips = 'full opened'

    return l_e, r_e, lips
