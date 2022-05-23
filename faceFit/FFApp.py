import cv2
import numpy as np
import glob
import json
import os

from kivy.base import EventLoop
from kivy.lang import builder
from kivy.properties import ObjectProperty, StringProperty, NumericProperty
from kivy.uix.button import Button
from kivy.uix.widget import Widget
from skimage import filters as filters
# import kivy module
import kivy
from kivy.app import App
from kivy.metrics import dp
import Face as F_obj
from CustomModules import CustomGraphics
from kivy.core.window import Window
from kivy.graphics import Color, Rectangle
from kivy.uix.progressbar import ProgressBar
from kivy.uix.label import Label
from kivy.uix.image import Image
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.scrollview import ScrollView
from kivy.uix.behaviors import ToggleButtonBehavior
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from color_transfer import color_transfer
import blend_modes
import neural_style_transfer as st

kivy.require("1.9.1")
ref_files = []
buttons = []
result_buttons = []
ids = {}
view = {}
view_source = ''
selected = -1
morph_selected = -1
last_match = -1
r_rot = []
c_rot = []
prog_bars = []
pb_rots = []
delta = 5
final_morphs = {}
default_texture = []
morph_texture = {}
capture = None
filled=[]

try:
    root = os.path.dirname(os.path.abspath(__file__))
except NameError:
    import sys
    root = os.path.dirname(os.path.abspath(sys.argv[0]))

project_path = root
ref_path = project_path + '/images/'
img_path = 'images/'
thumbs_path = img_path + 'Thumbs/'
morph_path = ref_path + 'final_morphs/'
view_default = thumbs_path + 'view_default.jpg'
view_base_image = cv2.imread(view_default)
buf = cv2.flip(view_base_image, 0).tobytes()
default_texture = Texture.create(size=(view_base_image.shape[1], view_base_image.shape[0]), colorfmt='bgr')
default_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')

r_ch, g_ch, b_ch = cv2.split(view_base_image)
with open('triangles_reduced2.json', 'r') as f:
    media_pipes_tris = json.load(f)

class MyButton(ToggleButtonBehavior, Image):
    def __init__(self, **kwargs):
        super(MyButton, self).__init__(**kwargs)
        self.source = kwargs["source"]  # Stores the image name of the image button
        self.texture = self.button_texture(self.source)  # Treat the image as a texture, so you can edit it

    # The image changes depending on the state of the toggle button and the state.
    def on_state(self, widget, value):
        global view
        if value == 'down':
            self.texture = self.button_texture(self.source, off=True)
            self.__setattr__('height', 200)
        else:
            self.texture = self.button_texture(self.source)
            self.__setattr__('height', 150)

    def button_texture(self, data, off=False):
        im = cv2.imread(data)
        if off:
            im = cv2.rectangle(im, (1, 1), (im.shape[1]-1, im.shape[0]-1), (255, 255, 255), 10)

        buf_butt = cv2.flip(im, 0)  # flip upside down
        image_texture = Texture.create(size=(im.shape[1], im.shape[0]), colorfmt='bgr')
        image_texture.blit_buffer(buf_butt.tobytes(), colorfmt='bgr', bufferfmt='ubyte')
        return image_texture


class MyCamera(Image):
    def __init__(self, **kwargs):
        super(MyCamera, self).__init__(**kwargs)
        self.capture = cv2.VideoCapture(0)  # Connect to 0th camera
        self.reference = selected
        self.texture = default_texture
        self.source = view_default
        Clock.schedule_interval(self.update, 1.0 / 30)  # Set drawing interval

    def update(self, dt):
        global selected, view, pb_rots, view_source, last_match, morph_texture
        self.reference = selected
        if self.reference != -1:
            success, frame = self.capture.read()
            image = cv2.flip(frame, 1)
            self.texture = view.texture
            if success:
                image.flags.writeable = True
                cam_obj.get_landmarks(image)
                # # DRAW LANDMARKS
                # cam_obj.draw('contours')
                # cam_obj.draw('tessellation')
                c_rot[0] = str(int(cam_obj.beta))
                c_rot[1] = str(int(cam_obj.alpha))
                c_rot[2] = str(int(cam_obj.tilt['angle']))

                perc_x = 100 - abs(ref[self.reference].beta - cam_obj.beta)
                perc_y = 100 - abs(ref[self.reference].alpha - cam_obj.alpha)
                perc_z = 100 - abs(ref[self.reference].tilt['angle'] - cam_obj.tilt['angle'])
                pb_rots = [perc_x, perc_y, perc_z]

                overlaid = self.cut_paste(ref[self.reference], cam_obj)
                buf_overlaid = cv2.flip(overlaid, 0).tobytes()
                texture = Texture.create(size=(overlaid.shape[1], overlaid.shape[0]), colorfmt='bgr')
                texture.blit_buffer(buf_overlaid, colorfmt='bgr', bufferfmt='ubyte')

                self.texture = texture
                if match():
                    path = morph_path + 'morph_' + str(self.reference) + '.png'
                    cv2.imwrite(path, final_morphs[self.reference])
                    buttons[self.reference].state = 'normal'
                    buttons[self.reference].height = 150
                    last_morphed = cv2.imread(path)
                    buf_morph = cv2.flip(last_morphed, 0).tobytes()
                    morph_texture[self.reference] = Texture.create(size=(last_morphed.shape[1], last_morphed.shape[0]), colorfmt='bgr')
                    morph_texture[self.reference].blit_buffer(buf_morph, colorfmt='bgr', bufferfmt='ubyte')
                    for i in range(0, 3):
                        c_rot[i] = '-'
                        r_rot[i] = '-'
                        pb_rots[i] = 0

                    self.texture = morph_texture[self.reference]
                    # ids.view.texture = self.texture
                    result_buttons[self.reference].texture = morph_texture[self.reference]
                    last_match = selected
                    filled.append(last_match)
                    selected = -1

        else:
            self.texture = None
            if last_match != -1 and morph_texture[last_match]:
                self.texture = morph_texture[last_match]
                if morph_selected != -1:
                    self.texture = morph_texture[morph_selected]
            else:
                self.texture = default_texture


    def on_play(self, instance, value):
        if not self._camera:
            return
        if value:
            self._camera.start()
        else:
            self._camera.stop()

    def cut_paste(self, obj1, obj2):
        offset = 10
        img1 = obj1.image
        img2 = obj2.image
        mask2 = obj2.self_hud_mask()
        mask2gray = cv2.cvtColor(mask2, cv2.COLOR_BGR2GRAY)
        # cut roi face from cam
        temp_1 = img1.copy()
        temp_2 = img2.copy()
        masked_2 = cv2.bitwise_and(temp_2, temp_2, mask=mask2gray)

        rx = (obj1.delta_x + 2 * offset) / (obj2.delta_x + 2 * offset)
        ry = (obj1.delta_y + 2 * offset) / (obj2.delta_y + 2 * offset)
        media_scale = round((rx + ry) / 2, 2)
        min_x_2, min_y_2 = obj2.bb_p1
        max_x_2, max_y_2 = obj2.bb_p2

        center_1 = obj1.pix_points[168]
        center_2 = obj2.pix_points[168]

        delta_2_min = [min_x_2 - offset, min_y_2 - offset]
        delta_2_max = [max_x_2 + offset, max_y_2 + offset]
        if delta_2_min[0] < 0:
            delta_2_min[0] = 0
        elif delta_2_max[0] > img2.shape[1]:
            delta_2_max[0] = img2.shape[1]
        if delta_2_min[1] < 0:
            delta_2_min[1] = 0
        elif delta_2_max[1] > img2.shape[0]:
            delta_2_max[1] = img2.shape[0]

        new_min_x = center_1[0] - int((center_2[0] - delta_2_min[0]) * media_scale)
        new_min_y = center_1[1] - int((center_2[1] - delta_2_min[1]) * media_scale)
        new_max_x = center_1[0] + int((delta_2_max[0] - center_2[0]) * media_scale)
        new_max_y = center_1[1] + int((delta_2_max[1] - center_2[1]) * media_scale)
        if new_min_x < 0:
            new_min_x = 0
        elif new_min_y < 0:
            new_min_y = 0
        elif new_max_x > img1.shape[1]:
            new_max_x = img1.shape[1]
        elif new_max_y > img1.shape[0]:
            new_max_y = img1.shape[0]
        cropped_2 = masked_2[delta_2_min[1]:delta_2_max[1], delta_2_min[0]:delta_2_max[0]]

        cropped_2 = cv2.resize(cropped_2, ((new_max_x - new_min_x), (new_max_y - new_min_y)),
                               interpolation=cv2.INTER_LINEAR)

        edged_2 = find_edges(cropped_2, 3, 1, 1, 3)

        copied = temp_1[new_min_y:new_max_y, new_min_x:new_max_x]
        copied = cv2.addWeighted(copied, 1, edged_2, .99, 1)
        temp_1[new_min_y:new_max_y, new_min_x:new_max_x] = copied
        return temp_1


class MainLayout(Widget):
    source = StringProperty('')
    ref_x = StringProperty('-')
    ref_y = StringProperty('-')
    ref_z = StringProperty('-')
    cam_x = StringProperty('-')
    cam_y = StringProperty('-')
    cam_z = StringProperty('-')
    pb_x = NumericProperty(0)
    pb_y = NumericProperty(0)
    pb_z = NumericProperty(0)
    scroll = ObjectProperty(None)
    def __init__(self, **kwargs):
        super(MainLayout, self).__init__(**kwargs)
        self.source = view_default
        Clock.schedule_once(self.verify_ids, 0)
        self.event = Clock.schedule_interval(self.update, 0.1)

    def verify_ids(self, widget):
        global ids
        ids = self.ids
        self.build()

    def build(self):
        global view, r_rot, c_rot, prog_bars, pb_rots, view_source
        grid1 = ids['l_scroll']
        grid1.bind(minimum_height=grid1.setter('height'))
        grid1 = self.image_load(img_path, grid1)
        grid2 = ids['r_box_grid']
        grid2.bind(minimum_height=grid2.setter('height'))
        grid2 = self.image_load(thumbs_path, grid2)
        view = ids['view']
        r_rot = [ids['ref_x'].text, ids['ref_y'].text, ids['ref_z'].text]
        c_rot = [ids['cam_x'].text, ids['cam_y'].text, ids['cam_z'].text]
        prog_bars = [ids['pb_x'], ids['pb_y'], ids['pb_z']]
        pb_rots = [self.pb_x, self.pb_y, self.pb_z]
        view_source = self.source


    def image_load(self, im_dir, grid):
        if im_dir == "images/":
            for idx, file in enumerate(ref_files):
                ref_img = cv2.imread(file)
                ref.append(F_obj.Face('ref'))
                ref[idx].get_landmarks(ref_img)
                # DRAW LANDMARKS
                # ref[idx].draw('contours')
                # ref[idx].draw('tessellation')
                button = MyButton(size_hint_y=None,
                                  height=150,
                                  source=os.path.join(im_dir, file),
                                  group="g1")
                buttons.append(button)
                button.bind(on_press=self.select)
                grid.add_widget(button)

        elif im_dir == 'images/Thumbs/':
            for idx, file in enumerate(ref_files):
                thumb = 'morph_thumb.jpg'
                button = MyButton(size_hint_y=None,
                                  height=150,
                                  disabled=False,
                                  source=os.path.join(im_dir, thumb),
                                  group="g2")
                result_buttons.append(button)
                button.bind(on_press=self.select_morph_button)
                grid.add_widget(button)

        return grid

    def select(self, btn):
        global selected, morph_selected
        for b in range(0, len(buttons)):
            if buttons[b] == btn and btn.state == 'down':
                r_rot[0] = str(int(ref[b].beta))
                r_rot[1] = str(int(ref[b].alpha))
                r_rot[2] = str(int(ref[b].tilt['angle']))
                selected = b
                if morph_selected != -1:
                    result_buttons[morph_selected].height = 150
                    result_buttons[morph_selected].state = 'normal'
                    morph_selected = -1
            elif buttons[b] == btn and btn.state == 'normal':
                view.source = ''
                for i in range(0, 3):
                    c_rot[i] = '-'
                    r_rot[i] = '-'
                    pb_rots[i] = 0
                selected = -1
                view.source = view_default
        return btn

    def select_morph_button(self, btn):
        global selected, morph_selected
        id = result_buttons.index(btn)
        if selected == -1 and id in filled:
            print(btn.texture, type(btn.texture))
            for b in range(0, len(result_buttons)):
                if result_buttons[b] == btn and btn.state == 'down':
                    morph_selected = b
                    ids.view.texture = morph_texture[morph_selected]
                    result_buttons[b].texture = morph_texture[morph_selected]

                elif result_buttons[b] == btn and btn.state == 'normal':
                    result_buttons[b].texture = morph_texture[morph_selected]
                    morph_selected = -1
                    ids.view.texture = default_texture
        else:
            btn.height = 150
            btn.state == 'normal'

    def update(self, dt):
        view.texture = ids.view.texture
        if len(morph_texture) >> 0:
            for m_tex in morph_texture.keys():
                result_buttons[m_tex].texture = morph_texture[m_tex]
        self.pb_x = pb_rots[0]
        self.pb_y = pb_rots[1]
        self.pb_z = pb_rots[2]
        self.ref_x = r_rot[0]
        self.ref_y = r_rot[1]
        self.ref_z = r_rot[2]
        self.cam_x = c_rot[0]
        self.cam_y = c_rot[1]
        self.cam_z = c_rot[2]


class MainApp(App):
    def on_start(self, **kwargs):

        return MainLayout()


# Match functions
def match():
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


def find_edges(img, blur_size, dx, dy, ksize):
    blurred = cv2.GaussianBlur(img, (blur_size, blur_size), sigmaX=0, sigmaY=0)
    grayed = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    # Laplacian Edge Detection
    laplacian = cv2.Laplacian(grayed, cv2.CV_64F)
    abs_laplacian = cv2.convertScaleAbs(laplacian)
    # Sobel Edge Detection
    sobel_x = cv2.Sobel(src=grayed, ddepth=cv2.CV_64F, dx=dx, dy=0, ksize=ksize)  # Sobel Edge Detection on the X axis
    sobel_y = cv2.Sobel(src=grayed, ddepth=cv2.CV_64F, dx=0, dy=dy, ksize=ksize)  # Sobel Edge Detection on the Y axis
    abs_grad_x = cv2.convertScaleAbs(sobel_x)
    abs_grad_y = cv2.convertScaleAbs(sobel_y)

    edged = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    edged = cv2.addWeighted(edged, 1, abs_laplacian, 0.5, 1)
    edged = cv2.cvtColor(edged, cv2.COLOR_GRAY2BGR)
    return edged


# Morph Functions
def hud_mask(mask_obj, masked_obj):
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
    hull_8u = []
    for i in range(0, len(hull2)):
        hull_8u.append((hull2[i][0], hull2[i][1]))

    mask = np.zeros(masked_obj.image.shape, dtype=masked_obj.image.dtype)
    cv2.fillConvexPoly(mask, np.int32(hull_8u), (255, 255, 255))
    return mask


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

    # Apply warpImage to small rectangular patches
    img1_rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
    siz = (r2[2], r2[3])
    img2_rect = apply_affine_transform(img1_rect, t1_rect, t2_rect, siz)

    img2_rect = img2_rect * mask

    # Copy triangular region of the rectangular patch to the output image
    img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] * ((1.0, 1.0, 1.0) - mask)
    img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] + img2_rect


def morph(c_obj, r_obj):
    source = ref[selected].image
    target = cam_obj.image

    img1_points = c_obj.pix_points
    img2_points = r_obj.pix_points

    mask = hud_mask(c_obj, r_obj)

    mid = cv2.moments(mask[:, :, 1])  # Find Centroid
    center = (int(mid['m10']/mid['m00']), int(mid['m01']/mid['m00']))

    cc_image = cv2.cvtColor(color_correct(target, source), cv2.COLOR_BGRA2BGR)
    c_obj.image[c_obj.bb_p1[1]:c_obj.bb_p2[1], c_obj.bb_p1[0]:c_obj.bb_p2[0]] = cc_image

    height, width, channels = r_obj.image.shape
    img2_new_face = np.zeros((height, width, channels), np.uint8)
    convexhull1 = cv2.convexHull(np.array(img1_points))
    cv2.fillConvexPoly(mask, convexhull1, 255)
    convexhull2 = cv2.convexHull(np.array(img2_points))

    dt = media_pipes_tris  # triangles

    if len(dt) == 0:  # If no Delaunay Triangles were found, quit
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


# Color Correction Functions
def sharpen(img):
    # convert to gray
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # blur
    smooth = cv2.GaussianBlur(gray, (99, 99), 0)

    # divide gray by morphology image
    division = cv2.divide(gray, smooth, scale=255)

    # sharpen using unsharp masking
    sharp = filters.unsharp_mask(division, radius=20, amount=2, preserve_range=False)
    sharp = (255 * sharp).clip(0, 255).astype(np.uint8)

    # threshold
    thresh = cv2.threshold(sharp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    return thresh


def color_correct(cam_img, ref_img):
    roi1 = cam_img[cam_obj.bb_p1[1]:cam_obj.bb_p2[1], cam_obj.bb_p1[0]:cam_obj.bb_p2[0]]
    roi2 = ref_img[ref[selected].bb_p1[1]:ref[selected].bb_p2[1], ref[selected].bb_p1[0]:ref[selected].bb_p2[0]]
    sharp = sharpen(cam_img)
    sharp_roi1 = sharp[cam_obj.bb_p1[1]:cam_obj.bb_p2[1], cam_obj.bb_p1[0]:cam_obj.bb_p2[0]]
    sharp_roi1 = cv2.GaussianBlur(sharp_roi1, (11, 11), 0)

    # transfer the color distribution from the source image to the target image
    roi1 = st.print_style_transfer(cv2.cvtColor(roi1, cv2.COLOR_BGR2RGB), cv2.cvtColor(roi2, cv2.COLOR_BGR2RGB))
    # roi1 = color_transfer(roi2, roi1, clip=True, preserve_paper=False)
    roi1 = cv2.normalize(roi1, None, 0 , 255, cv2.NORM_MINMAX, cv2.CV_8U)
    cv2.imshow('1', roi1)
    b_channel, g_channel, r_channel = cv2.split(roi1)
    alpha_channel = np.ones(b_channel.shape, dtype='uint8')  # b_channel.dtype)  # creating a dummy alpha channel image.
    roi1_4ch = cv2.merge((b_channel, g_channel, r_channel, alpha_channel))
    sharp_channel, = cv2.split(sharp_roi1)
    sharp_roi_4ch = cv2.merge((sharp_channel, sharp_channel, sharp_channel, alpha_channel))
    info_roi1 = np.iinfo(roi1_4ch.dtype)
    info_sharp = np.iinfo(sharp_roi_4ch.dtype)
    roi1_norm_float = roi1_4ch.astype(np.float64) / info_roi1.max
    sharp_norm_float = sharp_roi_4ch.astype(np.float64) / info_sharp.max
    blended = blend_modes.darken_only(roi1_norm_float, sharp_norm_float, .5)
    cc_out = (blended * 255).astype('uint8')
    return cc_out

def color_correct2(cam_img, ref_img):
    roi1 = cam_img[cam_obj.bb_p1[1]:cam_obj.bb_p2[1], cam_obj.bb_p1[0]:cam_obj.bb_p2[0]]
    roi2 = ref_img[ref[selected].bb_p1[1]:ref[selected].bb_p2[1], ref[selected].bb_p1[0]:ref[selected].bb_p2[0]]
    sharp = sharpen(cam_img)
    sharp_roi1 = sharp[cam_obj.bb_p1[1]:cam_obj.bb_p2[1], cam_obj.bb_p1[0]:cam_obj.bb_p2[0]]
    sharp_roi1 = cv2.GaussianBlur(sharp_roi1, (11, 11), 0)

    # transfer the color distribution from the source image to the target image
    roi1 = color_transfer(roi2, roi1, clip=True, preserve_paper=False)

    b_channel, g_channel, r_channel = cv2.split(roi1)
    alpha_channel = np.ones(b_channel.shape, dtype=b_channel.dtype)  # creating a dummy alpha channel image.
    roi1_4ch = cv2.merge((b_channel, g_channel, r_channel, alpha_channel))
    sharp_channel, = cv2.split(sharp_roi1)
    sharp_roi_4ch = cv2.merge((sharp_channel, sharp_channel, sharp_channel, alpha_channel))
    info_roi1 = np.iinfo(roi1_4ch.dtype)
    info_sharp = np.iinfo(sharp_roi_4ch.dtype)
    roi1_norm_float = roi1_4ch.astype(np.float64) / info_roi1.max
    sharp_norm_float = sharp_roi_4ch.astype(np.float64) / info_sharp.max
    blended = blend_modes.darken_only(roi1_norm_float, sharp_norm_float, .5)
    cc_out = (blended * 255).astype('uint8')
    return cc_out

for filename in glob.iglob(f'{ref_path}*'):
    if 'FACE_' in filename:
        ref_files.append(filename)
ref = []
cam_obj = F_obj.Face('cam')
MainApp().run()

