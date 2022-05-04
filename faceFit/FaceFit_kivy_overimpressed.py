import cv2
import mediapipe as mp
import numpy as np

from kivy.core.window import Window
from kivy.graphics import Color, Rectangle
from kivy.properties import Clock, BooleanProperty, NumericProperty, ListProperty, ObjectProperty
from kivy.uix.progressbar import ProgressBar

from pynput.keyboard import Key, Controller
import glob
from operator import itemgetter
import math
import json

# import kivy module
import kivy
from kivy.metrics import dp
kivy.require("1.9.1")
import os
from kivy.app import App
from kivy.uix.label import Label
from kivy.uix.image import Image
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.scrollview import ScrollView
from kivy.uix.behaviors import ToggleButtonBehavior
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from color_transfer import color_transfer
Window.maximize()

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
keyboard = Controller()
font = cv2.FONT_HERSHEY_SIMPLEX
final_morphs = {}
# For static images:
ref_files = []
ref_images = []
project_path = 'C:/Users/arkfil/Desktop/FITFace/faceFit'
ref_path = project_path + '/images/'
buttons = []
result_buttons = []
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
selected = -1
raw_image = []
labels = []
pbars = []
view = {}
out = []
curr = 0
delta = 5
sm = {}

# OBJECTS
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

class MyButton(ToggleButtonBehavior, Image):
    def __init__(self, **kwargs):
        super(MyButton, self).__init__(**kwargs)
        #Stores the image name of the image button
        self.source = kwargs["source"]
        #Treat the image as a texture so you can edit it
        self.texture = self.button_texture(self.source)

    #The image changes depending on the state of the toggle button and the state.
    def on_state(self, widget, value):
        if value == 'down':
            self.texture = self.button_texture(self.source, off=True)
        else:
            self.texture = self.button_texture(self.source)

    #Change the image, rectangular when pressed+Darken the color
    def button_texture(self, data, off=False):
        im = cv2.imread(data)
        # im = self.square_image(im)
        if off:
            # im = self.adjust(im)
            im = cv2.rectangle(im, (1, 1), (im.shape[1]-1, im.shape[0]-1), (255, 255, 255), 10)

        #flip upside down
        buf = cv2.flip(im, 0)
        image_texture = Texture.create(size=(im.shape[1], im.shape[0]), colorfmt='bgr')
        image_texture.blit_buffer(buf.tobytes(), colorfmt='bgr', bufferfmt='ubyte')
        return image_texture

    #Make the image square
    def square_image(self, img):
        h, w = img.shape[:2]
        if h > w:
            x = int((h-w)/2)
            img = img[x:x + w, :, :]
        elif h < w:
            x = int((w - h) / 2)
            img = img[:, x:x + h, :]

        return img

    #Darken the color of the image
    def adjust(self, img):
        #Performs a product-sum operation.
        print('adjust')
        # dst = cv2.resize(img,dsize=(200,200),interpolation=cv2.INTER_LINEAR)
        # [0, 255]Clip with to make uint8 type.
        # return np.clip(dst, 0, 255).astype(np.uint8)


class Camera(Image):
    def __init__(self, **kwargs):
        super(Camera, self).__init__(**kwargs)
        # Connect to 0th camera
        self.capture = cv2.VideoCapture(0)
        self.reference = selected

        # Set drawing interval
        Clock.schedule_interval(self.update, 1.0 / 30)

    # Drawing method to execute at intervals
    def update(self, dt):
        global raw_image, selected, view
        self.reference = selected
        success, self.frame = self.capture.read()
        image = cv2.flip(self.frame, 1)
        self.texture = view
        if success and self.reference != -1:
            image.flags.writeable = True
            cam_obj.get_landmarks(image)
            # load the images

            raw_image = cam_obj.image.copy()

            if cam_obj.beta >= ref[self.reference].beta + delta:
                text1 = 'left'
            elif cam_obj.beta <= ref[self.reference].beta - delta:
                text1 = 'right'
            else:
                text1 = 'ok'

            if cam_obj.alpha >= ref[self.reference].alpha + delta:
                text2 = 'down'
            elif cam_obj.alpha <= ref[self.reference].alpha - delta:
                text2 = 'up'
            else:
                text2 = 'ok'
            if ref[self.reference].tilt['angle'] >= cam_obj.tilt['angle'] + delta:
                text3 = 'left'
            elif ref[self.reference].tilt['angle'] <= cam_obj.tilt['angle'] - delta:
                text3 = 'right'
            else:
                text3 = 'ok'
            labels[3].__setattr__('text', str(int(cam_obj.beta)))
            labels[4].__setattr__('text', str(int(cam_obj.alpha)))
            labels[5].__setattr__('text', str(int(cam_obj.tilt['angle'])))
            pbars[0].__setattr__('source', int(cam_obj.beta-ref[self.reference].beta))
            overimpressed = cut_paste(ref[self.reference], cam_obj)

            # Convert to Kivy Texture

            buf = cv2.flip(overimpressed, 0).tobytes()
            texture = Texture.create(size=(overimpressed.shape[1], overimpressed.shape[0]), colorfmt='bgr')
            texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            # print(self.texture, overimpressed.shape)
            view.__setattr__('texture', texture)

            self.texture = texture
            if match():
                path = 'images/final_morphs/morph_' + str(self.reference) + '.png'
                cv2.imwrite(path, final_morphs[self.reference])
                result_buttons[self.reference].source = path
                buttons[self.reference].state = 'normal'
                buttons[self.reference].height = 150
                for i in range(0, 6):
                    labels[i].__setattr__('text', '-')
                view.__setattr__('source', path)

                selected = -1

        elif success and self.reference == -1:
            buf = cv2.flip(image, 0).tobytes()
            texture = Texture.create(size=(self.frame.shape[1], self.frame.shape[0]), colorfmt='bgr')
            texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            self.texture = texture

    def on_play(self, instance, value):
        if not self._camera:
            return
        if value:
            self._camera.start()
        else:
            self._camera.stop()


class BoxLayoutApp(App):  # class in which we are creating the button
    def __init__(self):
        super(BoxLayoutApp, self).__init__()
        global labels, view, pbars
        self.super_box = BoxLayout(orientation='horizontal')
        # ###LEFT PART### #
        self.l_box = BoxLayout(orientation='vertical', size=(dp(400), dp(1000)), size_hint=(0.2, 1))
        self.title_l = Label(text='Seleziona un quadro', size_hint=(1, 0.1),
                             pos_hint={'center_x': 0.5, 'center_y': 0.5})
        self.sc_view = ScrollView(size_hint=(1, 0.9))  # Definition of scroll view to place image buttons
        self.box = GridLayout(padding=[0, 25, 0, 0], cols=1, spacing=20, size_hint_y=None)

        # ###CENTRAL PART### #
        self.c_box = BoxLayout(orientation='vertical', size=(dp(800), dp(1000)), size_hint=(.6, 1), padding=[5,5,5,5])
        self.my_camera = Camera(allow_stretch=True, keep_ratio=True, size_hint=(1, 1), width=self.c_box.size[0],
                                height=self.c_box.size[1] / (self.c_box.size[0] / self.c_box.size[1]))
        self.view = Image(source='', allow_stretch=True, keep_ratio=True, size_hint=(1, 1),
                          width=self.c_box.size[0], height=self.c_box.size[1] / (self.c_box.size[0] / self.c_box.size[1]))
        view = self.view

        self.title2 = Label(text='Statistiche', size_hint=(1, .1))
        self.val1 = '-'
        self.riferimenti = BoxLayout(size_hint=(1, .3), orientation='horizontal')
        self.picture_box = BoxLayout(size_hint=(.3, 1), orientation='vertical')
        self.reference_title = Label(text='', size_hint=(1, .1))
        self.rot_x = Label(text='rot x = ', text_size=(dp(50), dp(20)),
                           size_hint=(1, .3), halign='left', valign='middle')
        self.rot_y = Label(text='rot y = ', text_size=(dp(50), dp(20)),
                           size_hint=(1, .3), halign='left', valign='middle')
        self.rot_z = Label(text='rot z = ', text_size=(dp(50), dp(20)),
                           size_hint=(1, .3), halign='left', valign='middle')

        self.c_value_x = Label(text=self.val1, text_size=(dp(50), dp(20)),
                               size_hint=(1, .3), halign='left', valign='middle')
        self.c_value_y = Label(text=self.val1, text_size=(dp(50), dp(20)),
                               size_hint=(1, .3), halign='left', valign='middle')
        self.c_value_z = Label(text=self.val1, text_size=(dp(50), dp(20)),
                               size_hint=(1, .3), halign='left', valign='middle')
        self.r_value_box = BoxLayout(size_hint=(.3, 1), orientation='vertical')
        self.reference_values = Label(text='valori quadro', size_hint=(1, .1))
        self.r_value_x = Label(text=self.val1, text_size=(dp(50), dp(20)),
                               size_hint=(1, .3), halign='left', valign='middle')
        self.r_value_y = Label(text=self.val1, text_size=(dp(50), dp(20)),
                               size_hint=(1, .3), halign='left', valign='middle')
        self.r_value_z = Label(text=self.val1, text_size=(dp(50), dp(20)),
                               size_hint=(1, .3), halign='left', valign='middle')
        self.c_value_box = BoxLayout(size_hint=(.3, 1), orientation='vertical')
        self.cam_values = Label(text='valori camera', size_hint=(1, .1))
        labels = [self.r_value_x, self.r_value_y, self.r_value_z, self.c_value_x, self.c_value_y, self.c_value_z]
        self.hints = BoxLayout(size_hint=(1,1), orientation='vertical', padding=[dp(40),self.cam_values.height*.1, dp(40), 0])
        self.prog_x = ProgressBar(max=100,size_hint=(1, .3), pos=(dp(50),dp(65)))
        self.prog_y = ProgressBar(size_hint=(1, .3), pos=(dp(70),dp(100)))
        self.prog_z = ProgressBar(size_hint=(1, .3) , pos=(dp(30),dp(25)))
        pbars = [self.prog_x, self.prog_y, self.prog_z]
        # ##RIGHT PART## #
        self.r_box = BoxLayout(orientation='vertical', size_hint=(0.2, 1))
        self.title_r = Label(text='Questo è il titolo', size_hint=(1, 0.1),
                             bold=True, pos_hint={'center_x': 0.5, 'center_y': 0.5})
        self.sc_view_results = ScrollView(size_hint=(1, 0.9))  # Definition of scroll view to place image buttons
        self.box_results = GridLayout(padding=[0, 25, 0, 0], cols=1, spacing=20, size_hint_y=None)

    def build(self):

        image_dir = "images/"  # Directory to read

        # ###LEFT PART### #
        self.c_box.bind(size=self._update_rect, pos=self._update_rect)
        with self.c_box.canvas.before:
            Color(.1, .2, .2, 1)  # green; colors range from 0-1 not 0-255
            self.rect = Rectangle(size=self.c_box.size, pos=self.c_box.pos)

        # self.title_l = Label(text='Seleziona un quadro', size_hint=(1, 0.1), bold=True , pos_hint={'center_x': 0.5, 'center_y': 0.5})
        self.box.bind(minimum_height=self.box.setter('height'))
        self.box = self.image_load(image_dir, self.box)  # Batch definition of image buttons, arranged in grid layout

        self.sc_view.add_widget(self.box)
        self.l_box.add_widget(self.title_l)
        self.l_box.add_widget(self.sc_view)

        # ###CENTRAL PART### #
        title = Label(text='FACE FIT', bold=True , size_hint=(1, .2))
        self.riferimenti.add_widget(self.picture_box)
        self.riferimenti.add_widget(self.r_value_box)
        self.riferimenti.add_widget(self.c_value_box)
        self.riferimenti.add_widget(self.hints)
        self.hints.add_widget(self.prog_x)
        self.hints.add_widget(self.prog_y)
        self.hints.add_widget(self.prog_z)
        self.picture_box.add_widget(self.reference_title)
        self.picture_box.add_widget(self.rot_x)
        self.picture_box.add_widget(self.rot_y)
        self.picture_box.add_widget(self.rot_z)
        self.r_value_box.add_widget(self.reference_values)
        self.r_value_box.add_widget(self.r_value_x)
        self.r_value_box.add_widget(self.r_value_y)
        self.r_value_box.add_widget(self.r_value_z)
        self.c_value_box.add_widget(self.cam_values)
        self.c_value_box.add_widget(self.c_value_x)
        self.c_value_box.add_widget(self.c_value_y)
        self.c_value_box.add_widget(self.c_value_z)
        self.c_box.add_widget(title)
        self.c_box.add_widget(self.view)
        self.c_box.add_widget(self.title2)
        self.c_box.add_widget(self.riferimenti)

        # # ##RIGHT PART## #
        self.box_results.bind(minimum_height=self.box_results.setter('height'))
        self.box_results = self.image_load("results_images/", self.box_results)
        self.sc_view_results.add_widget(self.box_results)
        self.r_box.add_widget(self.title_r)
        self.r_box.add_widget(self.sc_view_results)

        self.super_box.add_widget(self.l_box)
        self.super_box.add_widget(self.c_box)
        self.super_box.add_widget(self.r_box)
        # self.first_screen.add_widget(self.progressbar)
        # self.sm.add_widget(self.first_screen)
        # self.main.add_widget(self.super_box)
        # self.sm.add_widget(self.main)

        return self.super_box

    def _update_rect(self, instance, value):
        self.rect.pos = instance.pos
        self.rect.size = instance.size

    def image_load(self, im_dir, grid):
        if im_dir == "images/" :
            # images = ref_files  # sorted(os.listdir(im_dir))
            for idx, file in enumerate(ref_files):
                ref_img = cv2.imread(file)
                ref.append(Face('ref'))
                ref[idx].get_landmarks(ref_img)
                # DRAW LANDMARKS
                # ref[idx].draw('contours')
                # ref[idx].draw('tessellation')
            # for image in ref_files:
                button = MyButton(size_hint_y=None,
                                  height=150,
                                  source=os.path.join(im_dir, file),
                                  group="g1")
                buttons.append(button)
                ref_images.append(file)
                button.bind(on_press=self.select)
                grid.add_widget(button)

        elif im_dir == "results_images/":
            # images = final_morphs  # sorted(os.listdir(im_dir))
            for idx, file in enumerate(ref_files):
                im_dir = 'images/Thumbs/'
                thumb = 'morph_thumb.jpg'
                button = MyButton(size_hint_y=None,
                                  height=150,
                                  disabled=True,
                                  source=os.path.join(im_dir, thumb),
                                  group="g2")
                result_buttons.append(button)
                button.bind(on_press=self.select)
                grid.add_widget(button)

        return grid

    def select(self, btn):
        global selected

        for b in range(0, len(buttons)):
            if buttons[b] == btn and btn.state == 'down' :
                self.view.source = ref_images[b]
                btn.__setattr__('height', 200)
                if selected > -1 and selected != b:
                    buttons[selected].__setattr__('height', 150)
                labels[0].__setattr__('text', str(int(ref[b].beta)))
                labels[1].__setattr__('text', str(int(ref[b].alpha)))
                labels[2].__setattr__('text', str(int(ref[b].tilt['angle'])))

                selected = b
            elif buttons[b] == btn and btn.state == 'normal':
                buttons[b].__setattr__('height', 150)
                self.view.source = ''
                for i in range(0, 6):
                    labels[i].__setattr__('text', '-')
                selected = -1


        return btn



# CALCULATORS
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


def apply_affine_transform(src, src_tri, dst_tri, siz):
    warp_mat = cv2.getAffineTransform(np.float32(src_tri), np.float32(dst_tri))
    # Apply the Affine Transform just found to the src image
    dst = cv2.warpAffine(src, warp_mat, (siz[0], siz[1]), None, flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REFLECT_101)
    return dst


def warp_triangle(img1, img2, t1, t2):
    r1 = cv2.boundingRect(np.float32([t1]))
    r2 = cv2.boundingRect(np.float32([t2]))
    # Offset points by left top corner of the respective rectangles
    t1_rect = []
    t2_rect = []
    t2_rect_int = []
    for i in range(0, 3):
        t1_rect.append(((t1[i][0] - r1[0]), (t1[i][1] - r1[1])))
        t2_rect.append(((t2[i][0] - r2[0]), (t2[i][1] - r2[1])))
        t2_rect_int.append(((t2[i][0] - r2[0]), (t2[i][1] - r2[1])))

    # Get mask by filling triangle
    mask = np.zeros((r2[3], r2[2], 3), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(t2_rect_int), (1.0, 1.0, 1.0), 16, 0)
    # mask = cv2.GaussianBlur(mask, (3,3), sigmaX=0, sigmaY=0)

    # Apply warpImage to small rectangular patches
    img1_rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
    siz = (r2[2], r2[3])
    img2_rect = apply_affine_transform(img1_rect, t1_rect, t2_rect, siz)

    img2_rect = img2_rect * mask

    # Copy triangular region of the rectangular patch to the output image
    img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] * ((1.0, 1.0, 1.0) - mask)
    img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] + img2_rect


def find_center(points_array):
    length = points_array.shape[0]
    sum_x = np.sum(points_array[:, 0])
    sum_y = np.sum(points_array[:, 1])
    return int(sum_x / length), int(sum_y / length)





# MATCHING FUNCTION
def match():
    global cam_obj

    if len(cam_obj.points) != 0:
        # CHECK HEAD ORIENTATION
        if ref[selected].alpha - delta <= cam_obj.alpha <= ref[selected].alpha + delta and \
                ref[selected].beta - delta <= cam_obj.beta <= ref[selected].beta + delta and\
                ref[selected].tilt['angle'] - delta <= cam_obj.tilt['angle'] <= ref[selected].tilt['angle'] + delta:
            print('match_angles')
            # CHECK EXPRESSION
            cam_exp = (cam_obj.status['l_e'], cam_obj.status['r_e'], cam_obj.status['lips'])
            ref_exp = (ref[selected].status['l_e'], ref[selected].status['r_e'], ref[selected].status['lips'])
            if cam_exp == ref_exp:
                print('MATCH')
                morphed = morph(cam_obj, ref[selected])

                final_morphs[selected] = morphed
                return True
            else:
                return False


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


# def txt_multiline(img, end_point, increment, txt, color):
#     y_start = (end_point[1] - 25)
#     y_increment = increment[1]
#     for i, line in enumerate(txt.split('\n')):
#         y = y_start + i * y_increment
#         cv2.putText(img, line, ((end_point[0] + increment[0]), y), font, 1, color, 2)


# def rotate_hud(origin, point, _angle):
#     """
#     Rotate a point counterclockwise by a given angle around a given origin.
#     The angle should be given in radians.
#     """
#     ox, oy = origin
#     px, py = point
#
#     qx = ox + np.cos(_angle) * (px - ox) - np.sin(_angle) * (py - oy)
#     qy = oy + np.sin(_angle) * (px - ox) + np.cos(_angle) * (py - oy)
#     return int(qx), int(qy)
# def draw_hud(img, center_point, b_box, up_down, r_l, turn_z, ref_id):
#     hud = np.zeros_like(img, np.uint8)
#     j_arrow = [int(i * 1.3) for i in b_box]
#     color = (0, 255, 0)
#     thick = 10
#     thick_tilt = 10
#     x = center_point[0]
#     y = center_point[1] - b_box[1]
#     a = 25
#     b = 35
#     sector = 0
#
#     # BIG ARROW
#     if r_l == 'right':
#         if up_down == 'up':
#             angle = 45
#             sector = 1
#         elif up_down == 'down':
#             angle = 135
#             sector = 2
#         else:
#             angle= 90
#             sector = 2
#     elif r_l == 'left':
#         if up_down == 'up':
#             angle = -45
#             sector = 1
#         elif up_down == 'down':
#             angle = -135
#             sector = 2
#         else:
#             angle = -90
#             sector = 2
#     else:
#         if up_down == 'up':
#             angle = 0
#             sector = 1
#         elif up_down == 'down':
#             angle = 180
#             sector = 2
#         else:
#             sector = 1
#             angle = 0
#             x = -200
#             y = -300
#     big_arrow = np.array([(x, y), (x - a, y), (x - a, y - b), (x - 2 * a, y - b), (x, y - 2 * b), (x + 2 * a, y - b),
#                       (x + a, y - b), (x + a, y)], np.float32)
#     arrow_rotated = big_arrow.copy()
#
#     for i, p in enumerate(big_arrow):
#         arrow_rotated[i] = rotate_hud(center_point, p, np.deg2rad(angle))
#
#     cv2.polylines(hud, [np.int32(arrow_rotated)], True, color, thick)
#
#     # TILT
#     if turn_z == 'right':
#         if sector != 2:
#             alpha = 100
#             beta = 170
#             tilt_p1 = [center_point[0] - j_arrow[0], center_point[1] + 10]
#             tilt_p2 = [center_point[0] - j_arrow[0] - 10, center_point[1] + 25]
#             tilt_p3 = [center_point[0] - j_arrow[0] + 15, center_point[1] + 20]
#         else:
#             alpha = 190
#             beta = 260
#             tilt_p1 = [center_point[0] - 5, center_point[1] - j_arrow[1]-1]
#             tilt_p2 = [center_point[0] - 20, center_point[1] - j_arrow[1]-10]
#             tilt_p3 = [center_point[0] - 15, center_point[1] - j_arrow[1]+15]
#     elif turn_z == 'left':
#         if sector != 2:
#             alpha = 10
#             beta = 80
#             tilt_p1 = [center_point[0] + j_arrow[0], center_point[1] + 10]
#             tilt_p2 = [center_point[0] + j_arrow[0] + 10, center_point[1] + 25]
#             tilt_p3 = [center_point[0] + j_arrow[0] - 15, center_point[1] + 20]
#         else:
#             alpha = 280
#             beta = 350
#             tilt_p1 = [center_point[0] + 5, center_point[1] - j_arrow[1]-1]
#             tilt_p2 = [center_point[0] + 20, center_point[1] - j_arrow[1]-10]
#             tilt_p3 = [center_point[0] + 15, center_point[1] - j_arrow[1]+15]
#     else:
#         alpha = 0
#         beta = 0
#         tilt_p1 = [-300, -300]
#         tilt_p2 = [-300, -300]
#         tilt_p3 = [-300, -300]
#         thick_tilt = 0
#     cv2.ellipse(hud, center_point, j_arrow, 0, alpha, beta, color, thick_tilt)
#     pts0 = np.array([tilt_p1, tilt_p2, tilt_p3], np.int32)
#     pts0 = pts0.reshape((-1, 1, 2))
#     cv2.polylines(hud, [pts0], True, color, thick_tilt)
#
#     # L Eye
#
#     if ref[ref_id].status['l_e'] != cam_obj.status['l_e']:
#         l_e_start = cam_obj.centers['l_e']
#         l_e_end = (cam_obj.centers['l_e'][0] + j_arrow[0]//2, cam_obj.centers['l_e'][1] - j_arrow[1])
#         txt_l = "check\nleft\neye"
#     else:
#         l_e_start = (-300, -300)
#         l_e_end = (-300, -300)
#         txt_l = ""
#     cv2.line(hud, l_e_start, l_e_end, color, 2)
#     txt_multiline(hud, l_e_end, [5, 25], txt_l, color)
#
#     # R Eye
#     if ref[ref_id].status['r_e'] != cam_obj.status['r_e']:
#         r_e_start = cam_obj.centers['r_e']
#         r_e_end = (cam_obj.centers['r_e'][0] - j_arrow[0]//2, cam_obj.centers['r_e'][1] - j_arrow[1])
#         txt_r = "check\n right\n  eye"
#     else:
#         r_e_start = (-300, -300)
#         r_e_end = (-300, -300)
#         txt_r = ""
#     cv2.line(hud, r_e_start, r_e_end, color, 2)
#     txt_multiline(hud, r_e_end, [-95, 25], txt_r, color)
#
#     # Mouth
#     if ref[ref_id].status['lips'] != cam_obj.status['lips']:
#         mouth_start = cam_obj.centers['lips']
#         mouth_end = (cam_obj.centers['lips'][0] - 100), (cam_obj.centers['lips'][1] + 125)
#         txt_mouth = "check mouth"
#     else:
#         mouth_start = (-300, -300)
#         mouth_end = (-300, -300)
#         txt_mouth = ""
#     cv2.line(hud, mouth_start, mouth_end, color, 2)
#     cv2.putText(hud, txt_mouth, ((mouth_end[0] - 100), (mouth_end[1] + 20)), font, 1, color, 2)
#     d = 20
#     # ref[selected].image = cv2.cvtColor(ref[selected].image, cv2.COLOR_BGR2RGB)
#     temp_ref = ref[selected].image.copy()
#     rx = (cam_obj.delta_x + 2 * d) / (ref[selected].delta_x + 2 * d)
#     ry = (cam_obj.delta_y + 2 * d) / (ref[selected].delta_y + 2 * d)
#     media_scale = round((rx + ry) / 2, 2)
#     r_min_x, r_min_y = ref[selected].bb_p1
#     r_max_x, r_max_y = ref[selected].bb_p2
#     center_ref = ref[selected].pix_points[168]
#     center_cam = cam_obj.pix_points[168]
#     delta_r_min = [r_min_x - d, r_min_y - d]
#     delta_r_max = [r_max_x + d, r_max_y + d]
#     cropped_ref = temp_ref[delta_r_min[1]:delta_r_max[1], delta_r_min[0]:delta_r_max[0]]
#     print(center_cam, center_ref, delta_r_min, delta_r_max)
#     print(r_max_y-r_min_y+2*d, r_max_x-r_min_x+2*d, 'picture')
#     # print('rx', rx, 'ry', ry, 'media', media_scale)
#     new_min_x = center_cam[0] - int((center_ref[0] - delta_r_min[0]) )
#     new_min_y = center_cam[1] - int((center_ref[1] - delta_r_min[1]) )
#     new_max_x = center_cam[0] - int((center_ref[0] - delta_r_max[0]) )
#     new_max_y = center_cam[1] - int((center_ref[1] - delta_r_max[1]) )
#     print(new_min_x, new_max_x, new_min_y, new_max_y )
#     print(selected)
#     temp_cam = img.copy()
#     # print(exp_bb_reference.shape, exp_bb_cam.shape, img.shape)
#     print(temp_cam[new_min_y:new_max_y, new_min_x:new_max_x].shape, cropped_ref.shape)
#     temp_cam[new_min_y:new_max_y, new_min_x:new_max_x] = \
#         cv2.addWeighted(temp_cam[new_min_y:new_max_y, new_min_x:new_max_x], 1, cropped_ref, .9, 1)
#     mask = hud.astype(bool)
#     # print(new_max_x - new_min_x, new_max_y - new_min_y, cropped_ref.shape)
#     out_image = temp_cam.copy()
#     # out_image = cv2.addWeighted(img, 1, exp_bb_cam, 0.7, 1)
#     out_image[mask] = cv2.addWeighted(img, 1, hud, 0.9, 1)[mask]
#     return out_image


# def draw(part, img, face_l):
#     conn = ''
#     dr_spec = ''
#     if part == 'iris':
#         conn = mp_face_mesh.FACEMESH_IRISES
#         dr_spec = mp_drawing_styles.get_default_face_mesh_iris_connections_style()
#     elif part == 'contours':
#         conn = mp_face_mesh.FACEMESH_CONTOURS
#         dr_spec = mp_drawing_styles.get_default_face_mesh_contours_style()
#     elif part == 'tessellation':
#         conn = mp_face_mesh.FACEMESH_TESSELATION
#         dr_spec = mp_drawing_styles.get_default_face_mesh_tesselation_style()
#     else:
#         print('WRONG PART DESCRIPTOR')
#     mp_drawing.draw_landmarks(
#         image=img,
#         landmark_list=face_l,
#         connections=conn,
#         landmark_drawing_spec=None,
#         connection_drawing_spec=dr_spec)


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


def cut_paste(obj1, obj2):
    d = 10
    img1 = obj1.image
    img2 = obj2.image
    # mask1 = obj1.self_hud_mask()
    mask2 = obj2.self_hud_mask()
    mask2gray = cv2.cvtColor(mask2, cv2.COLOR_BGR2GRAY)
    # cut roi face from cam
    temp_1 = obj1.image.copy()
    temp_2 = obj2.image.copy()
    masked_2 = cv2.bitwise_and(temp_2, temp_2, mask=mask2gray)


    rx = (obj1.delta_x + 2 * d) / (obj2.delta_x + 2 * d)
    ry = (obj1.delta_y + 2 * d) / (obj2.delta_y + 2 * d)
    media_scale = round((rx + ry) / 2, 2)
    min_x_2, min_y_2 = obj2.bb_p1
    max_x_2, max_y_2 = obj2.bb_p2

    center_2 = obj2.pix_points[168]
    center_1 = obj1.pix_points[168]

    delta_2_min = [min_x_2 - d, min_y_2 - d]
    delta_2_max = [max_x_2 + d, max_y_2 + d]
    if delta_2_min[0] < 0:
        delta_2_min[0] = 0
    elif delta_2_max[0] > img2.shape[1]:
        delta_2_max[0] = img2.shape[1]
    if delta_2_min[1] < 0:
        delta_2_min[1] = 0
    elif delta_2_max[1] > img2.shape[0]:
        delta_2_max[1] = img2.shape[0]

    # # cut roi face from ref
    # min_x_1, min_y_1 = obj1.bb_p1
    # max_x_1, max_y_1 = obj1.bb_p2
    # delta_1_min = [min_x_1 - d, min_y_1 - d]
    # delta_1_max = [max_x_1 + d, max_y_1 + d]
    # cropped_1 = temp_1[delta_1_min[1]:delta_1_max[1], delta_1_min[0]:delta_1_max[0]]

    new_min_x = center_1[0] - int((center_2[0] - delta_2_min[0])*media_scale)
    new_min_y = center_1[1] - int((center_2[1] - delta_2_min[1])*media_scale)
    new_max_x = center_1[0] + int((delta_2_max[0] - center_2[0])*media_scale)
    new_max_y = center_1[1] + int((delta_2_max[1] - center_2[1])*media_scale)
    if new_min_x < 0:
        new_min_x = 0
    elif new_min_y < 0:
        new_min_y = 0
    elif new_max_x > img1.shape[1]:
        new_max_x = img1.shape[1]
    elif new_max_y > img1.shape[0]:
        new_max_y = img1.shape[0]
    cropped_2 = masked_2[delta_2_min[1]:delta_2_max[1], delta_2_min[0]:delta_2_max[0]]

    cropped_2 = cv2.resize(cropped_2,((new_max_x - new_min_x),(new_max_y - new_min_y)),interpolation=cv2.INTER_LINEAR)
    blurred_2 = cv2.GaussianBlur(cropped_2, (3,3), sigmaX=0, sigmaY=0)
    # Sobel Edge Detection
    print(blurred_2.shape, blurred_2.dtype, cropped_2.shape, cropped_2.dtype)
    sobelx = cv2.Sobel(src=blurred_2, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=3)  # Sobel Edge Detection on the X axis
    sobely = cv2.Sobel(src=blurred_2, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=3)  # Sobel Edge Detection on the Y axis
    # sobelxy = cv2.Sobel(src=blurred_2, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5)  # Combined X and Y Sobel Edge Detection
    abs_grad_x = cv2.convertScaleAbs(sobelx)
    abs_grad_y = cv2.convertScaleAbs(sobely)
    edged_2 = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    temp_1 = img1.copy()
    copied = temp_1[new_min_y:new_max_y, new_min_x:new_max_x]
    copied = cv2.addWeighted(temp_1[new_min_y:new_max_y, new_min_x:new_max_x], 1, edged_2, .9, 1)
    temp_1 [new_min_y:new_max_y, new_min_x:new_max_x] = copied
    return temp_1


def morph(c_obj, r_obj):
    source = ref[selected].image
    target = cam_obj.image

    # transfer the color distribution from the source image to the target image
    color_transfer(source, target, clip=False, preserve_paper=False)

    img1_warped = np.copy(r_obj.image)
    img1_points = c_obj.pix_points
    img2_points = r_obj.pix_points
    convexhull2 = cv2.convexHull(np.array(img2_points))
    mask = hud_mask(c_obj, r_obj)
    # Find Centroid
    mid = cv2.moments(mask[:, :, 1])
    center = (int(mid['m10']/mid['m00']), int(mid['m01']/mid['m00']))

    # triangles
    dt = media_pipes_tris

    height, width, channels = r_obj.image.shape
    img2_new_face = np.zeros((height, width, channels), np.uint8)
    convexhull1 = cv2.convexHull(np.array(img1_points))
    cv2.fillConvexPoly(mask, convexhull1, 255)
    # If no Delaunay Triangles were found, quit
    if len(dt) == 0:
        quit()

    tris1 = []
    tris2 = []
    for i in range(0, len(dt)):
        tri1 = []
        tri2 = []
        for j in range(0, 3):
            tri1.append(img1_points[dt[i][j]])
            tri2.append(img2_points[dt[i][j]])
        tris1.append(tri1)
        tris2.append(tri2)
    # Apply affine transformation to Delaunay triangles
    for i in range(0, len(tris1)):
        warp_triangle(c_obj.image, img2_new_face, tris1[i], tris2[i])

    # Clone seamlessly.

    gray = cv2.cvtColor(r_obj.image, cv2.COLOR_BGR2GRAY)
    img2_face_mask = np.zeros_like(gray)

    img2_head_mask = cv2.fillConvexPoly(img2_face_mask, convexhull2, 255)
    img2_head_mask = cv2.GaussianBlur(img2_head_mask, (15, 15), sigmaX=0, sigmaY=0)
    head_mask_3chan = cv2.cvtColor(img2_head_mask, cv2.COLOR_GRAY2BGR).astype('float') / 255.
    img2_face = img2_new_face.astype('float') / 255
    img2_bg = r_obj.image.astype('float') / 255
    out = img2_bg * (1 - head_mask_3chan) + img2_face * head_mask_3chan
    out = (out * 255).astype('uint8')

    output = cv2.seamlessClone(out, r_obj.image, img2_head_mask, center, cv2.NORMAL_CLONE)
    return output


def extract_index_nparray(nparray):
    index = None
    for num in nparray[0]:
        index = num
        break
    return index


for filename in glob.iglob(f'{ref_path}*'):
    if 'FACE_' in filename:
        ref_files.append(filename)

with open('triangles_reduced2.json', 'r') as f:
    media_pipes_tris = json.load(f)

ref = []
cam_obj = Face('cam')
app = BoxLayoutApp()
app.run()
