import os
from collections import Counter
from glob import iglob
from itertools import combinations
from json import load as load_json
from re import search

import cv2
import numpy as np
os.environ['KIVY_GL_BACKEND'] = 'sdl2'
# from kivy import Config
# Config.set('graphics', 'multisamples', '0')
from kivy.app import App
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.properties import ObjectProperty, StringProperty, NumericProperty
from kivy.uix.behaviors import ToggleButtonBehavior
from kivy.uix.image import Image
from kivy.uix.widget import Widget
from scipy.spatial import Delaunay
from shapely.geometry import MultiLineString
from shapely.ops import unary_union, polygonize

import Face as F_obj
from send_mail import send as send_mail

selected = reset = morph_selected = last_match = -1
delta = 7
ref, ref_files, buttons, morphed_buttons = [], [], [], []
r_rot, c_rot, pb_rots, filled = [], [], [], []
valid_images = [".jpg", ".gif", ".png", ".tga"]
ids, view, final_morphs, morph_texture = {}, {}, {}, {}
send_to, email_alert, view_source = '', '', ''

cam_obj = F_obj.Face('cam')

# Window.fullscreen = 'auto'
# Window.maximize()
# PATHS
try:
    project_path = os.path.dirname(os.path.abspath(__file__))
except NameError:
    import sys
    project_path = os.path.dirname(os.path.abspath(sys.argv[0]))

path_to = {"project": project_path, "images": 'images/'}
path_to["ref"] = ''.join([project_path, '/', path_to["images"]])
path_to["thumbs"] = ''.join([path_to["images"], 'Thumbs/'])
path_to["morphs"] = ''.join([path_to["ref"], 'final_morphs/'])
if not os.path.exists(path_to["morphs"]):
    os.makedirs(path_to["morphs"])

view_default = ''.join([path_to["thumbs"], 'view_default.jpg'])
morph_default = ''.join([path_to["thumbs"], 'morph_thumb.jpg'])
view_base_image = cv2.imread(view_default)
morph_base_image = cv2.imread(morph_default)

with open('triangulation.json', 'r') as f:
    media_pipes_tris = load_json(f)

with open('painting_data.json', 'r') as f:
    painting_data = load_json(f)


def images_in_folder(folder):
    files_list = []
    for filename in iglob(f'{folder}*'):
        ext = os.path.splitext(filename)[1]
        if ext.lower() not in valid_images:
            continue
        files_list.append(filename)
        files_list.sort()
    return files_list


def create_texture(image):
    b, g, r = cv2.split(image)
    image = cv2.merge((r, g, b))
    buf = cv2.flip(image, 0).tobytes()
    my_texture = Texture.create(size=(image.shape[1], image.shape[0]), colorfmt='rgb')
    my_texture.blit_buffer(buf, colorfmt='rgb', bufferfmt='ubyte')
    return my_texture


def btn_change(btn, state, height, texture):
    btn.state = state
    btn.height = height
    if texture != 'same':
        btn.texture = texture
    else:
        return


class MyButton(ToggleButtonBehavior, Image):
    def __init__(self, **kwargs):
        super(MyButton, self).__init__(**kwargs)
        self.source = kwargs["source"]  # Stores the image name of the image button
        self.texture = self.button_texture(self.source)  # Treat the image as a texture, so you can edit it

    def button_texture(self, data, off=False):
        im = cv2.imread(data)
        if off:
            im = cv2.rectangle(im, (1, 1), (im.shape[1] - 1, im.shape[0] - 1), (255, 255, 255), 10)
        image_texture = create_texture(im)
        if self:
            return image_texture


class MyCamera(Image):
    intro_images = ObjectProperty(Image(source='intro.zip', anim_delay=0.1, anim_loop=1))

    def __init__(self, **kwargs):
        super(MyCamera, self).__init__(**kwargs)
        self.capture = cv2.VideoCapture(0)  # Connect to 0th camera
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        self.fps = self.capture.get(cv2.CAP_PROP_FPS)
        Clock.schedule_interval(self.update, 1.0 / 30)  # Set drawing interval / 30

    def update(self, dt):
        global selected, view, pb_rots, view_source, last_match, morph_texture, cam_obj, morph_selected

        if selected != -1:
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
                    c_rot[0] = str(int(cam_obj.alpha))
                    c_rot[1] = str(int(cam_obj.beta))
                    c_rot[2] = str(int(cam_obj.gamma))
                    # calc Rotation's Hints
                    percent_x = 100 - abs(ref[selected].alpha - cam_obj.alpha)
                    percent_y = 100 - abs(ref[selected].beta - cam_obj.beta)
                    percent_z = 100 - abs(ref[selected].gamma - cam_obj.gamma)
                    pb_rots = [percent_x, percent_y, percent_z]
                    # apply user's mask
                    overlaid = cut_paste_user_mask(ref[selected], cam_obj)
                    over_texture = create_texture(overlaid)
                    self.texture = over_texture

                    if match():
                        # save result
                        numb = "0" + str(selected + 1) if selected <= 8 else str(selected + 1)

                        path = path_to["morphs"] + 'morph_' + numb + '.png'
                        cv2.imwrite(path, final_morphs[selected])
                        # reset values
                        cam_obj = F_obj.Face('cam')
                        btn_change(buttons[selected], 'normal', 150, 'same')
                        last_morphed = cv2.imread(path)  # FIX create directories if that does not exist
                        for i in range(0, 3):
                            c_rot[i], r_rot[i], pb_rots[i] = '-', '-', 0
                        # create result texture and apply to view and morph button
                        morph_texture[selected] = create_texture(last_morphed)
                        self.texture = morph_texture[selected]
                        btn_change(morphed_buttons[selected], 'down', 200, morph_texture[selected])
                        # set trackers
                        last_match = morph_selected = selected
                        filled.append(last_match)                        
                        selected = -1  # reset selected
            else:
                print('camera not ready')
        else:  # selected = -1
            self.texture = None
            if reset == 0:
                if last_match != -1 and morph_texture and morph_texture[last_match]:
                    if morph_selected == -1:
                        self.texture = morph_texture[last_match]
                    else:
                        self.texture = morph_texture[morph_selected]
                else:
                    if morph_selected == -1:
                        self.texture = view_default_texture
                    elif morph_texture:
                        self.texture = morph_texture[morph_selected]
            else:
                self.texture = self.intro_images.texture

    def on_play(self, instance, value):
        if not self._camera:
            return
        if value:
            self._camera.start()
        else:
            self._camera.stop()


class MainLayout(Widget):
    source = StringProperty('')
    email_alert = StringProperty('-')
    ref_x = StringProperty('-')
    ref_y = StringProperty('-')
    ref_z = StringProperty('-')
    cam_x = StringProperty('-')
    cam_y = StringProperty('-')
    cam_z = StringProperty('-')
    pb_x = NumericProperty(0)
    pb_y = NumericProperty(0)
    pb_z = NumericProperty(0)
    scroll = ObjectProperty()

    def __init__(self, **kwargs):
        super(MainLayout, self).__init__(**kwargs)
        self.email_alert = email_alert
        self.source = view_default
        Clock.schedule_once(self.verify_ids, 0)
        self.event = Clock.schedule_interval(self.update, 0.1)

    def verify_ids(self, widget):
        global ids
        ids = self.ids
        self.build()

    def build(self):
        global view, r_rot, c_rot, pb_rots, view_source, email_alert
        grid1 = ids['l_box_grid']
        grid2 = ids['r_box_grid']
        grid1.bind(minimum_height=grid1.setter('height'))
        grid2.bind(minimum_height=grid2.setter('height'))
        for idx, file in enumerate(ref_files):
            ref_img = cv2.imread(file)
            ref.append(F_obj.Face('ref'))
            ref[idx].get_landmarks(ref_img)
            # DRAW LANDMARKS
            # ref[idx].draw('contours')
            # ref[idx].draw('tessellation')
            button_ref = MyButton(source=os.path.join(path_to["images"], file), group="g1")
            button_morphs = MyButton(disabled=False, source=morph_default, group="g2")
            buttons.append(button_ref)
            button_ref.bind(on_press=self.select)
            grid1.add_widget(button_ref)
            morphed_buttons.append(button_morphs)
            button_morphs.bind(on_press=self.select_morph_button)
            grid2.add_widget(button_morphs)
        view = ids['view']
        email_alert = ids['email_alert']
        r_rot = [ids['ref_x'].text, ids['ref_y'].text, ids['ref_z'].text]
        c_rot = [ids['cam_x'].text, ids['cam_y'].text, ids['cam_z'].text]
        pb_rots = [self.pb_x, self.pb_y, self.pb_z]
        view_source = self.source

    @staticmethod
    def select(btn):
        global selected, morph_selected, reset
        btn_change(buttons[selected], 'normal', 150, texture='same') if selected >= 0 else print('no')
        for b in range(0, len(buttons)):
            if buttons[b] == btn and btn.state == 'down':
                r_rot[0] = str(int(ref[b].alpha))
                r_rot[1] = str(int(ref[b].beta))
                r_rot[2] = str(int(ref[b].gamma))
                selected = b
                print(selected)
                reset = 0
                btn_change(buttons[selected], 'down', 200, texture='same')
                if morph_selected != -1:
                    btn_change(morphed_buttons[morph_selected], 'normal', 150, 'same')
                    morph_selected = -1
            elif buttons[b] == btn and btn.state == 'normal':
                view.source = ''
                btn_change(buttons[selected], 'normal', 150, texture='same')
                for i in range(0, 3):
                    c_rot[i] = '-'
                    r_rot[i] = '-'
                    pb_rots[i] = 0
                selected = -1
                reset = 0
                view.source = view_default

        return btn

    @staticmethod
    def select_morph_button(btn):
        global selected, morph_selected, last_match, reset
        m_id = morphed_buttons.index(btn)
        if m_id not in filled:
            btn_change(morphed_buttons[m_id], 'normal', 150, 'same')
            btn_change(morphed_buttons[morph_selected], 'down', 200, 'same')
            # return
        elif selected == -1:
            if btn.state == 'down':
                last_match = morph_selected
                morph_selected = m_id
                btn_change(morphed_buttons[morph_selected], 'down', 200, 'same')
                btn_change(morphed_buttons[last_match], 'normal', 150, 'same')
            elif btn.state == 'normal':
                last_match = -1
                morph_selected = -1
                btn_change(morphed_buttons[m_id], 'normal', 150, 'same')
        else:
            btn_change(buttons[selected], 'normal', 150, 'same')
            btn_change(morphed_buttons[m_id], 'down', 200, 'same')
            for i in range(0, 3):
                c_rot[i], r_rot[i], pb_rots[i] = '-', '-', 0
            selected = -1
            last_match = -1
            morph_selected = m_id
            reset = 0

    def update(self, dt):
        if len(morph_texture) >> 0:
            for m_tex in morph_texture.keys():
                morphed_buttons[m_tex].texture = morph_texture[m_tex]
        self.email_alert = email_alert.text
        self.pb_x, self.pb_y, self.pb_z = pb_rots
        self.ref_x, self.ref_y, self.ref_z = r_rot
        self.cam_x, self.cam_y, self.cam_z = c_rot

    def on_new_text(self):
        global send_to
        if self.ids.input.text != '':
            send_to = self.ids.input.text
            check = self.check_email_address(send_to)
            if check:
                self.checkout()
        else:
            email_alert.text = 'insert email'

    def on_reset(self):
        global selected, morph_selected, morph_texture, filled, reset
        morphed_files = images_in_folder(path_to["morphs"])
        for m_tex in morph_texture.keys():
            btn_change(morphed_buttons[m_tex], 'normal', 150, morphs_default_texture)
        selected, reset, morph_selected = -1, -1, -1
        morph_texture = {}
        filled = []
        email_alert.text, self.ids.input.text = '', ''
        # delete attached images
        for file in morphed_files:
            os.remove(file)

    @staticmethod
    def check_email_address(address):
        global email_alert
        # Checks if the address match regular expression
        is_valid = search("""^\w+([-+."]\w+)*@\w+([-.]\w+)*\.\w+([-.]\w+)*$""", address)
        if is_valid:  # If there is a matching group
            email_alert.text = 'email format is valid'
            return True
        else:
            email_alert.text = "email format isn't valid"
            return False

    def checkout(self):
        global selected, morph_selected, morph_texture, filled, reset
        morphed_files = images_in_folder(path_to["morphs"])
        if send_to != '':
            # send mail with attachments
            send_mail(morphed_files, send_to, painting_data)
            self.on_reset()
            # # reset GUI and variables
            # for m_tex in morph_texture.keys():
            #     btn_change(morphed_buttons[m_tex], 'normal', 150, morphs_default_texture)
            # selected, reset, morph_selected = -1, -1, -1
            # morph_texture = {}
            # filled = []
            # email_alert.text, self.ids.input.text = '', ''
            # # delete attached images
            # for file in morphed_files:
            #     os.remove(file)
        else:
            print('cannot send email')
            return


class MainApp(App):
    def on_start(self, **kwargs):
        return MainLayout()


# Match functions
def match():
    if len(cam_obj.points) != 0:
        # CHECK HEAD ORIENTATION
        if abs(cam_obj.alpha - ref[selected].alpha) <= delta and abs(cam_obj.beta - ref[selected].beta) <= delta and \
                abs(cam_obj.gamma - ref[selected].gamma) <= delta:
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


def cut_paste_user_mask(r_obj, c_obj):
    img1, img2 = r_obj.image, c_obj.image
    r_img, c_img = img1.copy(), img2.copy()
    # create masks
    concave_hull = get_concave_hull(np.int32(c_obj.pix_points))
    mask = np.zeros(cam_obj.image.shape, dtype=cam_obj.image.dtype)
    mask = cv2.fillPoly(mask, [concave_hull], (255, 255, 255))
    mask2gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    c_masked = cv2.bitwise_and(c_img, c_img, mask=mask2gray)
    # calc BBs scale factor
    center1 = r_obj.pix_points[168]
    center2 = c_obj.pix_points[168]
    offset = 10
    ratio_x = (r_obj.delta_x + 2 * offset) / (c_obj.delta_x + 2 * offset)
    ratio_y = (r_obj.delta_y + 2 * offset) / (c_obj.delta_y + 2 * offset)
    media_scale = round((ratio_x + ratio_y) / 2, 2)
    # calc Mask resize
    min_x, min_y = c_obj.bb_p1
    max_x, max_y = c_obj.bb_p2

    delta_2_min = [clamp(min_x - offset, 0, img2.shape[1]), clamp(min_y - offset, 0, img2.shape[0])]
    delta_2_max = [clamp(max_x + offset, 0, img2.shape[1]), clamp(max_y + offset, 0, img2.shape[0])]

    new_min_x = clamp(center1[0] - int((center2[0] - delta_2_min[0]) * media_scale), 0, img1.shape[1])
    new_min_y = clamp(center1[1] - int((center2[1] - delta_2_min[1]) * media_scale), 0, img1.shape[0])
    new_max_x = clamp(center1[0] + int((delta_2_max[0] - center2[0]) * media_scale), 0, img1.shape[1])
    new_max_y = clamp(center1[1] + int((delta_2_max[1] - center2[1]) * media_scale), 0, img1.shape[0])

    c_new_roi = c_masked[delta_2_min[1]:delta_2_max[1], delta_2_min[0]:delta_2_max[0]]
    c_new_roi = cv2.resize(c_new_roi, ((new_max_x - new_min_x), (new_max_y - new_min_y)),
                           interpolation=cv2.INTER_LINEAR)
    # find Mask edges and apply
    edged = find_edges(c_new_roi, 3, 1, 1, 3)
    copied = r_img[new_min_y:new_max_y, new_min_x:new_max_x]
    # if copied.shape == edged.shape:
    #     copied = cv2.addWeighted(copied, 1, edged, .99, 1)
    copied = cv2.addWeighted(copied, 1, edged, .99, 1) if copied.shape == edged.shape else copied
    r_img[new_min_y:new_max_y, new_min_x:new_max_x] = copied
    return r_img


def find_edges(img, blur_size, dx, dy, k_size):
    blurred = cv2.GaussianBlur(img, (blur_size, blur_size), sigmaX=0, sigmaY=0)
    grayed = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    # Laplacian Edge Detection
    laplacian = cv2.Laplacian(grayed, cv2.CV_64F)
    abs_laplacian = cv2.convertScaleAbs(laplacian)
    # Sobel Edge Detection
    sobel_x = cv2.Sobel(src=grayed, ddepth=cv2.CV_64F, dx=dx, dy=0, ksize=k_size)  # Sobel Edge Detection on the X axis
    sobel_y = cv2.Sobel(src=grayed, ddepth=cv2.CV_64F, dx=0, dy=dy, ksize=k_size)  # Sobel Edge Detection on the Y axis
    abs_grad_x = cv2.convertScaleAbs(sobel_x)
    abs_grad_y = cv2.convertScaleAbs(sobel_y)

    edged = cv2.addWeighted(abs_grad_x, 1, abs_grad_y, 1, 0)
    edged = cv2.addWeighted(edged, 1, abs_laplacian, 0.5, 1)
    edged = cv2.cvtColor(edged, cv2.COLOR_GRAY2BGR)
    return edged


def clamp(num, min_value, max_value):
    clamped = max(min(num, max_value), min_value)
    return clamped


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


def find_noise_scratches(img):  # De-noising
    dst = cv2.fastNlMeansDenoisingColored(img, None, 5, 5, 5, 15)
    noise = cv2.subtract(img, dst)
    return dst, noise


def morph(c_obj, r_obj):
    cam_image, cam_points, ref_image, ref_points = c_obj.image, c_obj.pix_points, r_obj.image, r_obj.pix_points
    mask_dilate_iter, mask_erode_iter, blur_value, offset = 10, 15, 35, 5
    head, file_name = os.path.split(ref_files[selected])
    # COLOR CORRECTION
    ref_smoothed, noise = find_noise_scratches(ref_image)
    r_roi = ref_smoothed[r_obj.bb_p1[1] - offset:r_obj.bb_p2[1] + offset,
               r_obj.bb_p1[0] - offset:r_obj.bb_p2[0] + offset]
    c_roi = cam_image[c_obj.bb_p1[1] - offset:c_obj.bb_p2[1] + offset,
               c_obj.bb_p1[0] - offset:c_obj.bb_p2[0] + offset]
    cam_cc = match_histograms(c_roi, r_roi)
    cam_image[c_obj.bb_p1[1] - offset:c_obj.bb_p2[1] + offset, c_obj.bb_p1[0] - offset:c_obj.bb_p2[0] + offset] = \
        cam_cc.astype('float64')

    # SWAP FACE
    ref_new_face = np.zeros(ref_image.shape, np.uint8)
    dt = media_pipes_tris  # triangles

    tris1 = [[cam_points[dt[i][j]] for j in range(3)]for i in range(len(dt))]
    tris2 = [[ref_points[dt[i][j]] for j in range(3)]for i in range(len(dt))]
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
    center = [x - y for x, y in zip(center, painting_data[file_name]["center_delta"])]
    r_face_mask_3ch = cv2.cvtColor(ref_face_mask, cv2.COLOR_GRAY2BGR).astype('float') / 255.
    out_face = (ref_new_face.astype('float') / 255)
    out_bg = ref_smoothed.astype('float') / 255
    out = out_bg * (1 - r_face_mask_3ch) + out_face * r_face_mask_3ch
    out = (out * 255).astype('uint8')
    out = cv2.add(out, noise)
    out = cv2.add(out, noise)
    output = cv2.seamlessClone(out, ref_image, ref_face_mask, center, cv2.NORMAL_CLONE)
    
    return output


ref_files = images_in_folder(path_to["ref"])

view_default_texture = create_texture(view_base_image)
morphs_default_texture = create_texture(morph_base_image)

MainApp().run()
