import cv2
import numpy as np

angle = 15 #Degrees
x = 100
y = 100
img = np.zeros((512, 512, 3), np.uint8)
triangle = np.array([(210, 70), (110, 220), (300, 220)], np.float32)
arrow = np.array([(x, y), (x*2/3,y),(x*2/3, y/2),(x/3, y/2),(x, y/8),(x+x*2/3, y/2),(x*4/3, y/2), (x*4/3, y)], np.float32)
cv2.polylines(img, [np.int32(arrow)], True, (0, 255, 0), 2)

def rotate(origin, point, _angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
    ox, oy = origin
    px, py = point

    qx = ox + np.cos(_angle) * (px - ox) - np.sin(_angle) * (py - oy)
    qy = oy + np.sin(_angle) * (px - ox) + np.cos(_angle) * (py - oy)
    return int(qx), int(qy)

# convert image to grayscale image
gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# convert the grayscale image to binary image
ret,thresh = cv2.threshold(gray_image,127,255,0)

# calculate moments of binary image
M = cv2.moments(thresh)

# calculate x,y coordinate of center
cX = int(M["m10"] / M["m00"])
cY = int(M["m01"] / M["m00"])
cX = 100
cY = 100
# put text and highlight the center
# cv2.circle(img, (cX, cY), 5, (255, 255, 255), -1)
# cv2.putText(img, "centroid", (cX - 25, cY - 25),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

arrow_rotated = arrow.copy()

for i, p in enumerate(arrow):
    arrow_rotated[i] = rotate((cX, cY), p, np.deg2rad(angle))

cv2.polylines(img, [np.int32(arrow_rotated)], True, (255, 0, 0), 3)

cv2.imshow('triangle rotation', img)
cv2.waitKey(0)