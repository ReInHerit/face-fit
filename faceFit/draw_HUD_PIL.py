import cv2
from PIL import Image, ImageDraw
import numpy as np


ref_folder = 'C:/Users/arkfil/Desktop/FITFace/faceFit/images/'
ref = 'C:/Users/arkfil/Desktop/FITFace/faceFit/images/13.jpg'

from PIL import Image, ImageDraw, ImageFont

picture = cv2.imread(ref)
cv2.cvtColor(picture, cv2.COLOR_BGR2RGB)
x = picture.shape[1]
y = picture.shape[0]
center = [x/2,y/2]
size = [150,250]
fill = (255, 255, 255, 255)
outline = (255, 255, 255)
def draw_hud(img, center_point, b_box):
# get an image
    with Image.open(img).convert("RGBA") as base:

        # make a blank image for the text, initialized to transparent text color
        txt = Image.new("RGBA", base.size, (255, 255, 255, 0))

        w_arrow_start = (center_point[0]+b_box[0], center_point[1])
        w_arrow_end =  (center_point[0]+b_box[0]+50, center_point[1])
        h_arrow_start = (center_point[0] , center_point[1]+b_box[1])
        h_arrow_end = (center_point[0] , center_point[1]+b_box[1] + 50)

        # get a font
        fnt = ImageFont.truetype("C:/Windows/Fonts/Arial.ttf", 40)
        # get a drawing context
        d = ImageDraw.Draw(txt)
        dr = ImageDraw.Draw(base)
        # draw text, half opacity
        # d.text((10, 10), "Hello", font=fnt, fill=(255, 255, 255, 128))
        # # draw text, full opacity
        # d.text((10, 60), "World", font=fnt, fill=(255, 255, 255, 255))
        d.line((w_arrow_start, w_arrow_end), fill=fill, width=10)
        d.regular_polygon((w_arrow_end,15),3,30,fill=fill,outline=outline)
        d.line((h_arrow_start, h_arrow_end), fill=fill, width=10)
        d.regular_polygon((h_arrow_end, 15), 3, 180, fill=fill, outline=outline)
        # d.line((0, base.size[1], base.size[0], 0), fill=128)
        d.arc([(0,0) , (50,100)], 15,65,None, 3)
        d.ellipse((center_point[0]-b_box[0], center_point[1]-b_box[1], center_point[0]+b_box[0], center_point[1]+b_box[1]), fill=(0, 0, 0, 0), outline=(255, 255, 255))




        out = Image.alpha_composite(base, txt)
        cv_out= cv2.cvtColor(np.array(out), cv2.COLOR_RGB2BGR)
        cv2.imshow('im',cv_out)
        cv2.waitKey(0)
        out.show()

draw_hud(ref, center, size)