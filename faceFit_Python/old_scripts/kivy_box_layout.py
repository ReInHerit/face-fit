import numpy as np
import cv2

# import kivy module
import kivy
from kivy.metrics import dp
import FaceFit_kivy as ff
kivy.require("1.9.1")
import random

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
from kivy.config import Config
from kivy.uix.widget import Widget
from kivy.lang import Builder

Builder.load_file('interface_settings.kv')
Config.set('graphics', 'width', '1620')
Config.set('graphics', 'height', '1000')
project_path = '/faceFit_Python'
path = project_path + '/images/'
lst = []
results_lst = []
buttons = []
result_buttons = []
for file in os.listdir(path):
    if file.lower().endswith(('.png', '.jpg', '.gif')):
        lst.append(file)

# declaring the colours you can use directly also
red = [1, 0, 0, 1]
green = [0, 1, 0, 1]
blue = [0, 0, 1, 1]
purple = [1, 0, 1, 1]


class KivyCamera(Image):
    def __init__(self, capture, fps, **kwargs):
        super(KivyCamera, self).__init__(**kwargs)
        self.capture = capture
        Clock.schedule_interval(self.update, 1.0 / fps)

    def update(self, dt):
        ret, frame = self.capture.read()
        if ret:
            # convert it to texture
            buf1 = cv2.flip(frame, 0)
            buf = buf1.tostring()
            image_texture = Texture.create(
                size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
            image_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            # display image from the texture
            self.texture = image_texture


# Image button class
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
        image_texture.blit_buffer(buf.tostring(), colorfmt='bgr', bufferfmt='ubyte')
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


# class in which we are creating the button
class BoxLayoutApp(App):
    def build(self):
        colors = [red, green, blue, purple]
        image_dir = "../images/"  # Directory to read
        self.image_name = ""  # Manage image file names

        super_box = BoxLayout(orientation='horizontal')
        l_box = BoxLayout(orientation='vertical', size=(dp(400),dp(1000)), size_hint=(0.2, 1))

        title_l = Label(text='Questo è il titolo', size_hint=(1, 0.1), pos_hint={'center_x': 0.5, 'center_y': 0.5})
        sc_view = ScrollView(size_hint=(1, 0.9))  # Definition of scroll view to place image buttons
        box = GridLayout(padding=[0,25,0,0] ,cols=1, spacing=20, size_hint_y=None)
        box.bind(minimum_height=box.setter('height'))
        # Batch definition of image buttons, arranged in grid layout
        box = self.image_load(image_dir, box)

        sc_view.add_widget(box)
        l_box.add_widget(title_l)
        l_box.add_widget(sc_view)

        c_box = BoxLayout(orientation='vertical', size=(dp(800),dp(1000)), size_hint=(0.6, 1))
        title = Label(text='Questo è il titolo', size_hint=(1, 0.2) )
        self.capture = cv2.VideoCapture(0)
        self.my_camera = KivyCamera(capture=self.capture, fps=30, size_hint=(1, 1))
        title2 = Label(text='Questo è il titolo',size_hint=(1, 0.1))
        feed = Image(color=random.choice(colors),
                     opacity=1,
                     size_hint=(1, 0.3))
        c_box.add_widget(title)
        c_box.add_widget(self.my_camera)
        c_box.add_widget(title2)
        c_box.add_widget(feed)

        r_box = BoxLayout(orientation='vertical', size_hint=(0.2, 1))
        title_r = Label(text='Questo è il titolo', size_hint=(1, 0.1), pos_hint={'center_x': 0.5, 'center_y': 0.5})
        sc_view_results = ScrollView(size_hint=(1, 0.9))  # Definition of scroll view to place image buttons
        box_results = GridLayout(padding=[0, 25, 0, 0], cols=1, spacing=20, size_hint_y=None)
        box_results.bind(minimum_height=box_results.setter('height'))
        # Batch definition of image buttons, arranged in grid layout
        box_results = self.image_load(image_dir, box_results)
        sc_view_results.add_widget(box_results)
        r_box.add_widget(title_r)
        r_box.add_widget(sc_view_results)

        # superbox used to again align the oriented widgets
        super_box.add_widget(l_box)
        super_box.add_widget(c_box)
        super_box.add_widget(r_box)

        return super_box

        # Load image button
    def image_load(self, im_dir, grid):
        if im_dir == "images/" :
            images = lst  # sorted(os.listdir(im_dir))

            for image in images:
                button = MyButton(size_hint_y=None,
                                  height=150,
                                  source=os.path.join(im_dir, image),
                                  group="g1")
                buttons.append(button)
                button.bind(on_press=self.select)
                grid.add_widget(button)

        elif im_dir == "results_images/":
            images = results_lst  # sorted(os.listdir(im_dir))

            for image in images:
                button = MyButton(size_hint_y=None,
                                  height=150,
                                  source=os.path.join(im_dir, image),
                                  group="g2")
                result_buttons.append(button)
                button.bind(on_press=self.select)
                grid.add_widget(button)

        return grid

    def select(self, btn):
        for b in range(0, len(buttons)):
            buttons[b].__setattr__('height', 150)
            if buttons[b] == btn and btn.state == 'down':
                btn.__setattr__('height', 200)

        return btn

    # When you press the image button, the image is displayed in the image widget
    def set_image(self, btn):
        if btn.state == "down":
            self.image_name = btn.source
            # Update screen
            Clock.schedule_once(self.update)

    # Screen update
    def update(self, t):
        self.image.source = self.image_name

# creating the object root for BoxLayoutApp() class
root = BoxLayoutApp()

# run function runs the whole program
# i.e run() method which calls the
# target function passed to the constructor.
root.run()
