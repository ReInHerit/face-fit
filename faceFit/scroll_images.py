from kivy.app import App
from kivy.clock import Clock
from kivy.lang import Builder
from kivy.uix.button import Button
# Program to explain how to add carousel in kivy

import os

project_path = 'C:/Users/arkfil/Desktop/FITFace/faceFit'
path = project_path + '/images/'
lst = []
buttons = []
for file in os.listdir(path):
    if file.lower().endswith(('.png', '.jpg', '.gif')):
        lst.append(file)


kv = '''
<MyButt>:
    size_hint: None, None
    size: 250, 250
    text: ""
    Image:
        source: 'C:/Users/arkfil/Desktop/FITFace/faceFit/images/face_01.png'
        y: self.parent.y + self.parent.height//2 - 125
        x: self.parent.x + self.parent.width//2 - 125 
        size: 250, 250
        allow_stretch: False
BoxLayout:
    pos_hint: {'top': 1}
    orientation: 'vertical'
    size_hint_y: None
    height: self.minimum_height
    Label:
        text: 'One At A Time'
        size_hint: 1, None
        font_size: 50
        height: 75
        text_size: self.size
        halign: 'center'
    ScrollView:
        size_hint: 1, None
        height: 250
        do_scroll_y: False
        BoxLayout:
            id: box
            size_hint_x: None
            width: self.minimum_width
            orientation: 'horizontal'
            padding: 10
            spacing: 10
    
'''


class MyButt(Button):

    pass


class TestApp(App):
    def build(self):
        Clock.schedule_once(self.add_buttons)
        return Builder.load_string(kv)

    def add_buttons(self, dt):
        box = self.root.ids.box
        for i in range(25):
            # Add some widgets to the ScrollView
            box.add_widget(MyButt(text='Button ' + str(i)))


class ButtonApp(App):

    def build(self):
        # create a fully styled functional button
        # Adding images normal.png and down.png
        for f in range(0, len(lst)):
            btn = Button(text="Push Me !",
                         background_normal=path + 'face_01.png',
                         background_down='down.png',
                         size_hint=(.3, .3),
                         pos_hint={"x": 0.35, "y": 0.3}
                         )

        # bind() use to bind the button to function callback
            btn.bind(on_press=self.callback)
        buttons.append(btn)
        # return btn

        # callback function tells when button pressed

    def callback(self, event):
        print("button pressed")
        print('Yoooo !!!!!!!!!!!')

# root = ButtonApp()
# root.run()
TestApp().run()
