import os
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
import numpy as np

# Mediapipe

mp_drawing = solutions.drawing_utils
mp_drawing_styles = solutions.drawing_styles
mp_face_mesh = solutions.face_mesh
LEFT_EYE = mp_face_mesh.FACEMESH_LEFT_EYE
RIGHT_EYE = mp_face_mesh.FACEMESH_RIGHT_EYE
LIPS = mp_face_mesh.FACEMESH_LIPS
IRISES = mp_face_mesh.FACEMESH_IRISES
CONTOURS = mp_face_mesh.FACEMESH_CONTOURS
TESSELATION = mp_face_mesh.FACEMESH_TESSELATION

drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'face_landmarker.task'))
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.FaceLandmarkerOptions(base_options=base_options,
                                       output_face_blendshapes=True,
                                       output_facial_transformation_matrixes=True,
                                       num_faces=1)
detector = vision.FaceLandmarker.create_from_options(options)

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

    def get_landmarks(self, n_image):
        w, h, c = n_image.shape
        self.image = n_image
        rgb_image = cv2.cvtColor(n_image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        detection_result = detector.detect(mp_image)
        face_landmarks = detection_result.face_landmarks
        blend_shapes = detection_result.face_blendshapes
        transformation_matrix = detection_result.facial_transformation_matrixes[0]
        if face_landmarks[0]:
            length = len(face_landmarks[0])
            self.points, self.pix_points = [], []
            for i in range(0, length):
                landmark = face_landmarks[0][i]
                x, y, z, = landmark.x, landmark.y, landmark.z
                self.points.append([x, y, z])
                self.pix_points.append([int(x * h), int(y * w)])
                # print('x, y, z', x, y, z)
            # calc expression
            self.status['l_e'], self.status['r_e'], self.status['lips'] = self.check_expression(self.pix_points)
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
            delta = (cx_max - cx_min, cy_max - cy_min)
            self.delta_x = delta[0]
            self.delta_y = delta[1]
            self.alpha, self.beta, self.gamma = self.get_face_angles(transformation_matrix)


    def get_face_angles(self, transformation_matrix):
        rotation_matrix = transformation_matrix[:3, :3]

        # Perform Euler angle decomposition
        pitch = np.arcsin(rotation_matrix[1, 2])
        roll = np.arctan2(-rotation_matrix[0, 2], rotation_matrix[2, 2])
        jaw = np.arctan2(-rotation_matrix[1, 0], rotation_matrix[1, 1])

        # Convert angles from radians to degrees if needed
        pitch_deg = round(np.degrees(pitch))
        roll_deg = round(np.degrees(roll))
        jaw_deg = round(np.degrees(jaw))
        # print('pitch_deg, roll_deg, jaw_deg', pitch_deg, roll_deg, jaw_deg)
        return pitch_deg, roll_deg, jaw_deg

    def check_expression(self, face_points):
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
