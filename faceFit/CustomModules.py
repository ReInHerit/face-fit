from kivy.app import App
from kivy.graphics import *
class CustomGraphics(App):
    def SetBG(layout, **options):
        with layout.canvas.before:
                if 'bg_color' in options:
                    bg_rgba = options['bg_color']
                    if len(bg_rgba) == 4:
                        Color(bg_rgba[0], bg_rgba[1], bg_rgba[2], bg_rgba[3])
                    elif len(bg_rgba) == 3:
                        Color(bg_rgba[0], bg_rgba[1], bg_rgba[2])
                    else:
                        Color(0,0,0,1)
                layout.bg_rect = Rectangle(pos=layout.pos, size=layout.size)
                def update_rect(instance, value):
                    instance.bg_rect.pos = instance.pos
                    instance.bg_rect.size = instance.size
                # listen to size and position changes
                layout.bind(pos=update_rect, size=update_rect)