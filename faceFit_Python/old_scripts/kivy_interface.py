import kivy
from kivy.app import App
from kivy.uix.label import Label
from kivy.uix.image import Image
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.scrollview import ScrollView
from kivy.uix.behaviors import ToggleButtonBehavior
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.config import Config
from kivy.properties import ObjectProperty
from kivy.uix.widget import Widget
from kivy.lang import Builder
import cv2
import os

Builder.load_file('interface.kv')
Config.set('graphics', 'width', '1620')
Config.set('graphics', 'height', '1000')
project_path = '/faceFit_Python'
path = project_path + '/images/'
lst = []
buttons = []
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


class MyLayout(BoxLayout):
    myText = ObjectProperty(Label(text='ecco'))

    def build(self):
        print('ecce')
        image_dir = "../images/"  # Directory to read
        self.image_name = ""  # Manage image file names

        super_box = self.ids.super_box
        l_box = self.ids.l_box

        title_l = self.ids.title_l
        myText = 'cambio'

        sc_view = self.ids.sc_view  # Definition of scroll view to place image buttons
        box = self.ids.box
        box.bind(minimum_height=box.setter('height'))
        # Batch definition of image buttons, arranged in grid layout
        box = self.image_load(image_dir, box)

        sc_view.add_widget(box)
        l_box.add_widget(title_l)
        l_box.add_widget(sc_view)

        c_box = self.ids.c_box
        title = self.ids.title
        # self.capture = cv2.VideoCapture(0)
        # self.my_camera = self.ids.camera

        # camera = Image(color=random.choice(colors), opacity=1, size_hint=(1, None), size=(640, 480))
        title2 = self.ids.title2
        feed = self.ids.feed
        c_box.add_widget(title)
        # c_box.add_widget(self.my_camera)
        c_box.add_widget(title2)
        c_box.add_widget(feed)

        r_box = self.ids.r_box
        r_title = self.ids.title3
        result = self.ids.result

        r_box.add_widget(r_title)
        r_box.add_widget(result)

        # superbox used to again align the oriented widgets
        super_box.add_widget(l_box)
        super_box.add_widget(c_box)
        super_box.add_widget(r_box)

        return super_box

    # Load image button
    def image_load(self, im_dir, grid):
        images = lst  # sorted(os.listdir(im_dir))

        for image in images:
            button = MyButton(size_hint_y=None,
                              height=150,
                              source=os.path.join(im_dir, image),
                              group="g1")
            buttons.append(button)
            button.bind(on_press=self.select)
            grid.add_widget(button)

        print(buttons[0].height)
        return grid

    def select(self, btn):
        for b in range(0, len(buttons)):
            buttons[b].__setattr__('height', 150)
            if buttons[b] == btn and btn.state == 'down':
                btn.__setattr__('height', 200)

        return btn

    def progress(self):
        current = self.ids.my_p_b.value
        current += .25
        self.ids.my_p_b.value = current
        self.ids.p_b_label.text = f'{int(current*100)}% Progress'


class MyApp(App):
    def build(self):
        return MyLayout()


if __name__ == '__main__':
    MyApp().run()