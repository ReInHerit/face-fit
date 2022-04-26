from kivy.app import App
from kivy.graphics import Line, Color, Rectangle, Ellipse
from kivy.metrics import dp
from kivy.properties import StringProperty, BooleanProperty, Clock
from kivy.uix.anchorlayout import AnchorLayout
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.gridlayout import GridLayout
from kivy.uix.scrollview import ScrollView
from kivy.uix.stacklayout import StackLayout
from kivy.uix.widget import Widget

class WidgetExample(GridLayout):
    count = 1
    count_enabled = BooleanProperty(False)
    my_text = StringProperty(str(count))
    text_input_str= StringProperty('foo')
    # slider_value_txt = StringProperty('50')
    def on_button_click(self):
        print('button clicked')
        if self.count_enabled:
            self.count += 1
            self.my_text = str(self.count)

    def on_toggle_button_state(self, widget):
        if widget.state == 'normal':
            widget.text = 'OFF'
            self.count_enabled = False
            # OFF
        else:
            self.count_enabled = True
            widget.text = 'ON'
            # ON

    def on_switch_active(self, widget):
        print('Switch ' + str(widget.active))

    # def on_slider_value(self, widget):
    #     self.slider_value_txt = str(int(widget.value))
    #     print('Slider '+ str(int(widget.value)))
    def on_text_validate(self, widget):
        self.text_input_str = widget.text

class ScrollViewExample(ScrollView):
    pass


class StackLayoutExample(StackLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # self.orientation= 'lr-bt'
        for i in range(0, 100):
            # size = dp(100)+i*10
            size = dp(100)
            b = Button(text=str(i + 1), size_hint=(None,None), size=(size, size))
            self.add_widget(b)


# class GridLayoutExample(GridLayout):
#     pass


class AnchorLayoutExample(AnchorLayout):
    pass


class BoxLayoutExample(BoxLayout):
    pass
'''    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.orientation='vertical'
        b1 = Button(text='A')
        b2 = Button(text='B')
        b3 = Button(text='C')
        self.add_widget(b1)
        self.add_widget(b2)
        self.add_widget(b3)
'''
class MainWidget(Widget):
    pass

class TheFileApp(App):
    pass

class CanvasExample1(Widget):
    pass
class CanvasExample2(Widget):
    pass
class CanvasExample3(Widget):
    pass
class CanvasExample4(Widget):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        with self.canvas:
            Line(points=(100,100, 300,400), width=2)
            Color(0, 1, 0)
            Line(circle=(200,300,50))
            Line(rectangle=(50,150,25,150))
            self.rect = Rectangle(pos=(200, 150), size=(76,123))
    def on_button_a_click(self):
        # print('foo')
        x, y = self.rect.pos
        w, h = self.rect.size
        incr = dp(10)
        diff = self.width - (x + w)
        if diff < incr:
            x += diff
        else:
            x += incr
        self.rect.pos = (x, y)

class CanvasExample5(Widget):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.ball_size = dp(50)
        self.vx= dp(3)
        self.vy = dp(4)
        with self.canvas:
            self.ball = Ellipse(pos=(100, 100), size=(self.ball_size, self.ball_size))
        Clock.schedule_interval(self.update, 1/60)
    def on_size(self, *args):
        print('on size : ' + str(self.width) + ', ' + str(self.height))
        self.ball.pos = (self.center_x-self.ball_size/2, self.center_y - self.ball_size/2)
    def update(self, dt):
        x, y = self.ball.pos

        x += self.vx
        y += self.vy
        if y +self.ball_size > self.height:
            y = self.height- self.ball_size
            self.vy = -self.vy
        if x + self.ball_size  > self.width:
            x= self.width - self.ball_size
            self.vx = - self.vx
        if y < 0:
            y = 0
            self.vy = -self.vy
        if x < 0:
            x= 0
            self.vx = - self.vx



        self.ball.pos= (x, y)
class CanvasExample6(Widget):
    pass
class CanvasExample7(BoxLayout):
    pass

TheFileApp().run()