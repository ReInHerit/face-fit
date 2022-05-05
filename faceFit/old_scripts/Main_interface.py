import kivy
import cv2
from kivy.app import App
from kivy.base import EventLoop
from kivy.core.image import Image
from kivy.factory import Factory
from kivy.graphics.texture import Texture
from kivy.properties import Clock, StringProperty, BooleanProperty
from kivy.uix.behaviors import ToggleButtonBehavior
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.widget import Widget
import os
project_path = '/faceFit'
path = project_path + '/images/'
lst = []
buttons = []
for file in os.listdir(path):
    if file.lower().endswith(('.png', '.jpg', '.gif')):
        lst.append(file)


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
    active = BooleanProperty(False)
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

class mainLayout(Widget):

    def __init__(self, **kwargs):
        super(mainLayout, self).__init__(**kwargs)
        print(self.ids)
        image_dir = "../images/"
        for image in lst:
            print(image)
            button = MyButton(source=os.path.join(image_dir, image))
            print(button.source)
            buttons.append(button)
            button.bind(on_press=self.select)
            self.ids.box_grid.add_widget(button)

    # def build(self):
    #     image_dir = "images/"
    #     self.image_load(image_dir, self.ids.box_grid)
    #     print('qui')
    # def image_load(self, im_dir, grid):
    #     images = lst  # sorted(os.listdir(im_dir))
    #
    #     for image in images:
    #         button = MyButton(size_hint_y=None,
    #                           height=150,
    #                           source=os.path.join(im_dir, image),
    #                           group="g1")
    #         buttons.append(button)
    #         button.bind(on_press=self.select)
    #         grid.add_widget(button)
    #
    #     print(buttons[0].height)
    #     return grid





class FaceFitApp(App):
    def build(self):

        return mainLayout()



FaceFitApp().run()