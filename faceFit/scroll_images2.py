import os
import cv2
import numpy as np

from kivy.app import App
from kivy.uix.image import Image
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.scrollview import ScrollView
from kivy.uix.behaviors import ToggleButtonBehavior
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.config import Config
Config.set('graphics', 'width', '200')
Config.set('graphics', 'height', '600')
project_path = 'C:/Users/arkfil/Desktop/FITFace/faceFit'
path = project_path + '/images/'
lst = []
buttons = []
for file in os.listdir(path):
    if file.lower().endswith(('.png', '.jpg', '.gif')):
        lst.append(file)
#Image button class
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
        im = self.square_image(im)
        if off:
            im = self.adjust(im, alpha=0.6, beta=0.0)
            im = cv2.rectangle(im, (2, 2), (im.shape[1]-2, im.shape[0]-2), (255, 255, 0), 10)

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
    def adjust(self, img, alpha=1.0, beta=0.0):
        #Performs a product-sum operation.
        dst = alpha * img + beta
        # [0, 255]Clip with to make uint8 type.
        return np.clip(dst, 0, 255).astype(np.uint8)


class Test(BoxLayout):
    def __init__(self, **kwargs):
        super(Test, self).__init__(**kwargs)
        # Directory to read
        image_dir = "images/"
        #
        #Vertical arrangement
        self.orientation = 'vertical'

        #Manage image file names
        self.image_name = ""

        #Preparing a widget to display an image
        self.image = Image(size_hint=(0.5, 0.5))
        self.add_widget(self.image)

        #Definition of scroll view to place image buttons
        sc_view = ScrollView(size_hint=(1, None), size=(self.width, self.height*5))

        #Because only one widget can be placed in the scroll view
        box = GridLayout(cols=1, spacing=10, size_hint_y=None)
        box.bind(minimum_height=box.setter('height'))

        #Batch definition of image buttons, arranged in grid layout
        box = self.image_load(image_dir, box)

        sc_view.add_widget(box)
        self.add_widget(sc_view)

    #Load image button
    def image_load(self, im_dir, grid):
        images = lst #sorted(os.listdir(im_dir))

        for image in images:
            button = MyButton(size_hint_y=None,
                              height=200,
                              source=os.path.join(im_dir, image),
                              group="g1")
            button.bind(on_press=self.set_image)
            grid.add_widget(button)

        return grid

    #When you press the image button, the image is displayed in the image widget
    def set_image(self, btn):
        if btn.state=="down":
            self.image_name = btn.source
            #Update screen
            Clock.schedule_once(self.update)

    #Screen update
    def update(self, t):
        self.image.source = self.image_name


class SampleApp(App):
    def build(self):
        return Test()


SampleApp().run()