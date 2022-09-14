import cv2
import numpy as np
import glob
import json
import os

from kivy.properties import ObjectProperty, StringProperty, NumericProperty

from kivy.uix.widget import Widget

import kivy
from kivy.app import App
from kivy.uix.image import Image
from kivy.uix.behaviors import ToggleButtonBehavior
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.core.window import Window

import Face as F_obj

kivy.require("1.9.1")
ref_files = []
ref = []
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
progress_bars = []
pb_rots = []
delta = 7
final_morphs = {}
morph_texture = {}
# capture = None
filled = []
valid_images = [".jpg", ".gif", ".png", ".tga"]
cam_obj = F_obj.Face('cam')

try:
    project_path = os.path.dirname(os.path.abspath(__file__))
except NameError:
    import sys

    project_path = os.path.dirname(os.path.abspath(sys.argv[0]))

ref_path = project_path + '/images/'
img_path = 'images/'
thumbs_path = img_path + 'Thumbs/'
morph_path = ref_path + 'final_morphs/'
view_default = thumbs_path + 'view_default.jpg'
view_base_image = cv2.imread(view_default)
buf = cv2.flip(view_base_image, 0).tobytes()
default_texture = Texture.create(size=(view_base_image.shape[1], view_base_image.shape[0]), colorfmt='bgr')
default_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')

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
            im = cv2.rectangle(im, (1, 1), (im.shape[1] - 1, im.shape[0] - 1), (255, 255, 255), 10)

        buf_butt = cv2.flip(im, 0)  # flip upside down
        image_texture = Texture.create(size=(im.shape[1], im.shape[0]), colorfmt='bgr')
        image_texture.blit_buffer(buf_butt.tobytes(), colorfmt='bgr', bufferfmt='ubyte')
        if self:
            return image_texture


class MyCamera(Image):
    def __init__(self, **kwargs):
        super(MyCamera, self).__init__(**kwargs)
        self.capture = cv2.VideoCapture(0)  # Connect to 0th camera
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        self.selected = selected
        self.texture = default_texture
        self.source = view_default
        Clock.schedule_interval(self.update, 1.0/ 30)  # Set drawing interval

    def update(self, dt):
        global selected, view, pb_rots, view_source, last_match, morph_texture, cam_obj
        self.selected = selected
        if self.selected != -1:

            success, frame = self.capture.read()
            image = cv2.flip(frame, 1)

            self.texture = view.texture
            if success:

                image.flags.writeable = True
                cam_obj.get_landmarks(image)
                # # DRAW LANDMARKS
                # cam_obj.draw('contours')
                # cam_obj.draw('tessellation')
                if cam_obj.pix_points:
                    c_rot[0] = str(int(cam_obj.beta))
                    c_rot[1] = str(int(cam_obj.alpha))
                    c_rot[2] = str(int(cam_obj.tilt['angle']))

                    perc_x = 100 - abs(ref[self.selected].beta - cam_obj.beta)
                    perc_y = 100 - abs(ref[self.selected].alpha - cam_obj.alpha)
                    perc_z = 100 - abs(ref[self.selected].tilt['angle'] - cam_obj.tilt['angle'])
                    pb_rots = [perc_x, perc_y, perc_z]
                    overlaid = self.cut_paste(ref[self.selected], cam_obj)

                    buf_overlaid = cv2.flip(overlaid, 0).tobytes()
                    texture = Texture.create(size=(overlaid.shape[1], overlaid.shape[0]), colorfmt='bgr')
                    texture.blit_buffer(buf_overlaid, colorfmt='bgr', bufferfmt='ubyte')

                    self.texture = texture
                    if match():
                        path = morph_path + 'morph_' + str(self.selected) + '.png'
                        cv2.imwrite(path, final_morphs[self.selected])
                        cam_obj = F_obj.Face('cam')
                        buttons[self.selected].state = 'normal'
                        buttons[self.selected].height = 150
                        last_morphed = cv2.imread(path)
                        print(last_morphed.dtype)
                        buf_morph = cv2.flip(last_morphed, 0).tobytes()
                        morph_texture[self.selected] = Texture.create(size=(last_morphed.shape[0],last_morphed.shape[1]), colorfmt='bgr')
                        morph_texture[self.selected].blit_buffer(buf_morph, colorfmt='bgr', bufferfmt='ubyte')
                        for i in range(0, 3):
                            c_rot[i] = '-'
                            r_rot[i] = '-'
                            pb_rots[i] = 0

                        self.texture = morph_texture[self.selected]
                        # ids.view.texture = self.texture
                        result_buttons[self.selected].texture = morph_texture[self.selected]
                        last_match = selected
                        filled.append(last_match)
                        selected = -1
            else:
                print('camera not ready')
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

    def cut_paste(self, r_obj, c_obj):
        offset = 10
        img1 = r_obj.image
        img2 = c_obj.image
        mask2 = c_obj.self_hud_mask()
        mask2gray = cv2.cvtColor(mask2, cv2.COLOR_BGR2GRAY)

        temp1 = img1.copy()
        temp2 = img2.copy()
        masked2 = cv2.bitwise_and(temp2, temp2, mask=mask2gray)

        center1 = r_obj.pix_points[168]
        center2 = c_obj.pix_points[168]

        rx = (r_obj.delta_x + 2 * offset) / (c_obj.delta_x + 2 * offset)
        ry = (r_obj.delta_y + 2 * offset) / (c_obj.delta_y + 2 * offset)
        media_scale = round((rx + ry) / 2, 2)
        min_x, min_y = c_obj.bb_p1
        max_x, max_y = c_obj.bb_p2

        delta_2_min = [min_x - offset, min_y - offset]
        delta_2_max = [max_x + offset, max_y + offset]
        if delta_2_min[0] < 0:
            delta_2_min[0] = 0
        elif delta_2_max[0] > img2.shape[1]:
            delta_2_max[0] = img2.shape[1]
        if delta_2_min[1] < 0:
            delta_2_min[1] = 0
        elif delta_2_max[1] > img2.shape[0]:
            delta_2_max[1] = img2.shape[0]

        new_min_x = center1[0] - int((center2[0] - delta_2_min[0]) * media_scale)
        new_min_y = center1[1] - int((center2[1] - delta_2_min[1]) * media_scale)
        new_max_x = center1[0] + int((delta_2_max[0] - center2[0]) * media_scale)
        new_max_y = center1[1] + int((delta_2_max[1] - center2[1]) * media_scale)
        if new_min_x < 0:
            new_min_x = 0
        elif new_min_y < 0:
            new_min_y = 0
        elif new_max_x > img1.shape[1]:
            new_max_x = img1.shape[1]
        elif new_max_y > img1.shape[0]:
            new_max_y = img1.shape[0]
        cropped2 = masked2[delta_2_min[1]:delta_2_max[1], delta_2_min[0]:delta_2_max[0]]
        cropped2 = cv2.resize(cropped2, ((new_max_x - new_min_x), (new_max_y - new_min_y)),
                              interpolation=cv2.INTER_LINEAR)

        edged = find_edges(cropped2, 3, 1, 1, 3)

        copied = temp1[new_min_y:new_max_y, new_min_x:new_max_x]
        print(copied.shape, edged.shape)
        if copied.shape == edged.shape:
            copied = cv2.addWeighted(copied, 1, edged, .99, 1)
        temp1[new_min_y:new_max_y, new_min_x:new_max_x] = copied
        return temp1


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
        global view, r_rot, c_rot, progress_bars, pb_rots, view_source
        grid1 = ids['l_scroll']
        grid1.bind(minimum_height=grid1.setter('height'))
        grid1 = self.image_load(img_path, grid1)
        grid2 = ids['r_box_grid']
        grid2.bind(minimum_height=grid2.setter('height'))
        grid2 = self.image_load(thumbs_path, grid2)
        view = ids['view']
        r_rot = [ids['ref_x'].text, ids['ref_y'].text, ids['ref_z'].text]
        c_rot = [ids['cam_x'].text, ids['cam_y'].text, ids['cam_z'].text]
        progress_bars = [ids['pb_x'], ids['pb_y'], ids['pb_z']]
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
            btn.state = 'normal'

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
                ref[selected].beta - delta <= cam_obj.beta <= ref[selected].beta + delta and \
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


def calculate_cdf(histogram):
    """
    This method calculates the cumulative distribution function
    :param array histogram: The values of the histogram
    :return: normalized_cdf: The normalized cumulative distribution function
    :rtype: array
    """
    # Get the cumulative sum of the elements
    cdf = histogram.cumsum()

    # Normalize the cdf
    normalized_cdf = cdf / float(cdf.max())

    return normalized_cdf


def calculate_lookup(src_cdf, ref_cdf):
    """
    This method creates the lookup table
    :param array src_cdf: The cdf for the source image
    :param array ref_cdf: The cdf for the reference image
    :return: lookup_table: The lookup table
    :rtype: array
    """
    lookup_table = np.zeros(256)
    lookup_val = 0
    for src_pixel_val in range(len(src_cdf)):
        # lookup_val
        for ref_pixel_val in range(len(ref_cdf)):
            if ref_cdf[ref_pixel_val] >= src_cdf[src_pixel_val]:
                lookup_val = ref_pixel_val
                break
        lookup_table[src_pixel_val] = lookup_val
    return lookup_table


def match_histograms(src_image, ref_image):
    """
    This method matches the source image histogram to the reference signal
    :param image src_image: The original source image
    :param image  ref_image: The reference image
    :return: image_after_matching
    :rtype: image (array)
    """
    # Split the images into the different color channels b means blue, g means green and r means red
    src_b, src_g, src_r = cv2.split(src_image)
    ref_b, ref_g, ref_r = cv2.split(ref_image)

    # Compute the b, g, and r histograms separately
    # The flatten() Numpy method returns a copy of the array c collapsed into one dimension.
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

def apply_affine_transform(src, src_tri, dst_tri, siz):
    warp_mat = cv2.getAffineTransform(np.float32(src_tri), np.float32(dst_tri))
    # Apply the Affine Transform just found to the src image
    dst = cv2.warpAffine(src, warp_mat, (siz[0], siz[1]), None, flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REFLECT_101)
    return dst


def warp_triangle(img1, img2, t1, t2):
    x1, y1, w1, h1 = cv2.boundingRect(np.float32([t1]))
    x2, y2, w2, h2 = cv2.boundingRect(np.float32([t2]))
    # Offset points by left top corner of the respective rectangles
    t1_rect = []
    t2_rect = []
    t2_rect_int = []
    for i in range(0, 3):
        t1_rect.append(((t1[i][0] - x1), (t1[i][1] - y1)))
        t2_rect.append(((t2[i][0] - x2), (t2[i][1] - y2)))
        t2_rect_int.append(((t2[i][0] - x2), (t2[i][1] - y2)))

    # Get mask by filling triangle
    mask = np.zeros((h2, w2, 3), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(t2_rect_int), (1.0, 1.0, 1.0), 16, 0)

    # Apply warpImage to small rectangular patches
    img1_rect = img1[y1:y1 + h1, x1:x1 + w1]
    size = (w2, h2)
    img2_rect = apply_affine_transform(img1_rect, t1_rect, t2_rect, size)

    img2_rect = img2_rect * mask

    # Copy triangular region of the rectangular patch to the output image
    img2[y2:y2 + h2, x2:x2 + w2] = img2[y2:y2 + h2, x2:x2 + w2] * ((1.0, 1.0, 1.0) - mask)
    img2[y2:y2 + h2, x2:x2 + w2] = img2[y2:y2 + h2, x2:x2 + w2] + img2_rect


def morph(c_obj, r_obj):
    ref_image = r_obj.image
    cam_image = c_obj.image
    MASK_DILATE_ITER = 15
    MASK_ERODE_ITER = 20
    BLUR = 35
    cam_points = c_obj.pix_points
    ref_points = r_obj.pix_points
    offset = 5
    cc_r_roi = ref_image[r_obj.bb_p1[1] - offset:r_obj.bb_p2[1] + offset,
               r_obj.bb_p1[0] - offset:r_obj.bb_p2[0] + offset]
    cc_c_roi = cam_image[c_obj.bb_p1[1] - offset:c_obj.bb_p2[1] + offset,
               c_obj.bb_p1[0] - offset:c_obj.bb_p2[0] + offset]
    cam_cc = match_histograms(cc_c_roi, cc_r_roi)
    cam_image[c_obj.bb_p1[1] - offset:c_obj.bb_p2[1] + offset, c_obj.bb_p1[0] - offset:c_obj.bb_p2[0] + offset] = \
        cam_cc.astype('float64')
    # Find convex hull
    cam_hull = cv2.convexHull(np.array(cam_points), returnPoints=False)
    ref_hull = cv2.convexHull(np.array(ref_points))
    ref_hull_ids = cv2.convexHull(np.array(ref_points), returnPoints=False)
    # Create convex hull lists
    hull1 = []
    hull2 = []
    for i in range(0, len(cam_hull)):
        hull1.append(ref_points[cam_hull[i][0]])
    for i in range(0, len(ref_hull_ids)):
        hull2.append(ref_points[ref_hull_ids[i][0]])
    # Calculate Mask for Seamless cloning
    hull_1u = []
    hull_2u = []
    for i in range(0, len(hull1)):
        hull_1u.append((hull1[i][0], hull1[i][1]))
    for i in range(0, len(hull2)):
        hull_2u.append((hull2[i][0], hull2[i][1]))
    # mask = np.zeros(ref_image.shape, dtype=ref_image.dtype)
    # cv2.fillConvexPoly(mask, np.int32(hull_1u), (255, 255, 255))
    # mask2 = np.zeros(ref_image.shape, dtype=ref_image.dtype)
    # cv2.fillConvexPoly(mask2, np.int32(hull_2u), (255, 255, 255))
    # cv2.imshow('', mask)
    # cv2.imshow('2', mask2)
    # # cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    # #############
    # mid = cv2.moments(mask[:, :, 1])  # Find Centroid
    # center = (int(mid['m10'] / mid['m00']), int(mid['m01'] / mid['m00']))
    # mid2 = cv2.moments(mask2[:, :, 1])  # Find Centroid
    # center2 = (int(mid2['m10'] / mid2['m00']), int(mid2['m01'] / mid2['m00']))
    # print(center , center2)
    ref_new_face = np.zeros(ref_image.shape, np.uint8)
    dt = media_pipes_tris  # triangles
    tris1 = []
    tris2 = []
    for i in range(0, len(dt)):
        tri1 = []
        tri2 = []
        for j in range(0, 3):
            tri1.append(cam_points[dt[i][j]])
            tri2.append(ref_points[dt[i][j]])
        tris1.append(tri1)
        tris2.append(tri2)
    # Apply affine transformation to Delaunay triangles
    for i in range(0, len(tris1)):
        warp_triangle(cam_image, ref_new_face, tris1[i], tris2[i])
    # GENERATE FINAL IMAGE
    cv2.imshow('1', ref_new_face)
    ref_gray = cv2.cvtColor(ref_image, cv2.COLOR_BGR2GRAY)
    ref_img_mask = np.zeros_like(ref_gray)
    ref_face_mask = cv2.fillConvexPoly(ref_img_mask, ref_hull, 255)
    ref_face_mask = cv2.dilate(ref_face_mask, None, iterations=MASK_DILATE_ITER)
    ref_face_mask = cv2.erode(ref_face_mask, None, iterations=MASK_ERODE_ITER)
    ref_face_mask = cv2.GaussianBlur(ref_face_mask, (BLUR, BLUR), sigmaX=0, sigmaY=0)
    mid3 = cv2.moments(ref_face_mask)  # Find Centroid
    center = (int(mid3['m10'] / mid3['m00']), int(mid3['m01'] / mid3['m00']))
    cv2.imshow('3', ref_face_mask)
    r_face_mask_3ch = cv2.cvtColor(ref_face_mask, cv2.COLOR_GRAY2BGR).astype('float') / 255.
    out_face = ref_new_face.astype('float') / 255
    out_bg = ref_image.astype('float') / 255
    out = out_bg * (1 - r_face_mask_3ch) + out_face * r_face_mask_3ch
    out = (out * 255).astype('uint8')
    # center = int(ref_image.shape[0]/2), int(ref_image.shape[1]/2)
    output = cv2.seamlessClone(out, ref_image, ref_face_mask, center, cv2.NORMAL_CLONE)
    # output = cv2.addWeighted(out,.5,output,.5,0)
    return output


for filename in glob.iglob(f'{ref_path}*'):
    ext = os.path.splitext(filename)[1]
    if ext.lower() not in valid_images:
        continue
    ref_files.append(filename)

# cam_obj = F_obj.Face('cam')
MainApp().run()
