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
from kivy.uix.button import Button
from kivy.uix.widget import Widget
from kivy.lang import Builder
import os
project_path = '/faceFit_Python'
path = project_path + '/images/'
lst = []
buttons = []
for file in os.listdir(path):
    if file.lower().endswith(('.png', '.jpg', '.gif')):
        lst.append(file)

print(lst)
# class KivyCamera(Image):
#     def __init__(self, capture, fps, **kwargs):
#         super(KivyCamera, self).__init__(**kwargs)
#         self.capture = capture
#         Clock.schedule_interval(self.update, 1.0 / fps)
#
#     def update(self, dt):
#         ret, frame = self.capture.read()
#         if ret:
#             # convert it to texture
#             buf1 = cv2.flip(frame, 0)
#             buf = buf1.tostring()
#             image_texture = Texture.create(
#                 size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
#             image_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
#             # display image from the texture
#             self.texture = image_texture
#

# Image button class
class MyButton(Image):
    pass
class mainLayout(Widget):

    def __init__(self, **kwargs):
        super(mainLayout, self).__init__(**kwargs)
        print(self.ids)


        for image in lst:
            print(image)
            button = MyButton(source=os.path.join(path, image))
            print(button.source)
            buttons.append(button)
            # button.bind(on_press=self.select)
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