from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.lang import Builder

import os

project_path = '/faceFit'
path = project_path + '/images/'
lst = []
buttons = []
for file in os.listdir(path):
    if file.lower().endswith(('.png', '.jpg', '.gif')):
        lst.append(file)

Builder.load_string("""
<ButtonsApp>:
    orientation: "vertical"
    Button:
        text: ""
        Image:
            source: 'C:/Users/arkfil/Desktop/FITFace/faceFit/images/face_01.png'
            y: self.parent.y + self.parent.height//2 - 125
            x: self.parent.x + self.parent.width//2 - 125 
            size: 250, 250
            allow_stretch: False
    
""")

class ButtonsApp(App, BoxLayout):
    def build(self):
        return self

if __name__ == "__main__":
    ButtonsApp().run()