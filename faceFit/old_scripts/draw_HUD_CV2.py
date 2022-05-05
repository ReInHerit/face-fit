import cv2
from PIL import Image, ImageDraw
import numpy as np


ref_folder = 'C:/Users/arkfil/Desktop/FITFace/faceFit/images/'
ref = 'C:/Users/arkfil/Desktop/FITFace/faceFit/images/13.jpg'



picture = cv2.imread(ref)
cv2.cvtColor(picture, cv2.COLOR_BGR2RGB)
x = picture.shape[1]
y = picture.shape[0]
center = [int(x/2),int(y/2)]
size = [150,250]
fill = (255, 255, 255, 255)
outline = (255, 255, 255)
def draw_hud(img, center_point, b_box, up_down, r_l):
    hud = np.zeros_like(img, np.uint8)
    arrow_l = 50
    ext = 15
    if r_l== 'right':
        w_arrow_start = (center_point[0]+b_box[0], center_point[1])
        w_arrow_end =  (center_point[0]+b_box[0] + arrow_l, center_point[1])
        k = ext
    elif r_l ==  'left':
        w_arrow_start = (center_point[0] - b_box[0], center_point[1])
        w_arrow_end = (center_point[0] - b_box[0] - arrow_l, center_point[1])
        k = - ext
    else:
        w_arrow_start = 0
        w_arrow_end = 0
        k = 0
    if up_down == 'up':
        h_arrow_start = (center_point[0], center_point[1] - b_box[1])
        h_arrow_end = (center_point[0], center_point[1] - b_box[1] - arrow_l)
        j = - ext
    elif up_down == 'down':
        h_arrow_start = (center_point[0], center_point[1] + b_box[1])
        h_arrow_end = (center_point[0], center_point[1] + b_box[1] + arrow_l)
        j = ext
    else:
        h_arrow_start = 0
        h_arrow_end = 0
        j = 0
    # Polygon corner points coordinates
    cv2.ellipse(hud, center_point, b_box, 0, 0, 360, (255, 0, 0), 5)
    pts1 = np.array([[w_arrow_end[0],w_arrow_end[1]], [w_arrow_end[0], w_arrow_end[1]+10], [w_arrow_end[0]+k, w_arrow_end[1]],
                     [w_arrow_end[0], w_arrow_end[1]-10] ], np.int32)

    pts1 = pts1.reshape((-1, 1, 2))
    cv2.line(hud, w_arrow_start, w_arrow_end, (255, 0, 0), 5)
    cv2.polylines(hud, [pts1], True, (255, 0, 0), 5 )

    pts2 = np.array(
        [[h_arrow_end[0], h_arrow_end[1]], [h_arrow_end[0] + 10, h_arrow_end[1]], [h_arrow_end[0], h_arrow_end[1]+ j],
         [h_arrow_end[0] - 10, h_arrow_end[1] ]], np.int32)

    pts2 = pts2.reshape((-1, 1, 2))
    cv2.line(hud, h_arrow_start, h_arrow_end, (255, 0, 0), 5)
    cv2.polylines(img, [pts2], True, (255, 0, 0), 5)



draw_hud(picture, center, size)