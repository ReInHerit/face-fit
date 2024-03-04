import base64
import io
import math
from math import floor, ceil
import os
import re
import numpy as np
import cv2
from PIL import Image

from . import Face_Maker as F_obj


def create_face_dict(images_folder):
    face_dict_list = []

    for idx, filename in enumerate(os.listdir(images_folder)):
        file_path = os.path.join(images_folder, filename)
        if os.path.isfile(file_path):
            ref_img = cv2.imread(file_path)
            p_face = F_obj.Face('ref')
            p_face.get_landmarks(ref_img)
            face_dict = {
                'which': p_face.which,
                'id': idx,
                'src': file_path,
                'points': p_face.points,
                'expression': [p_face.status['l_e'], p_face.status['r_e'], p_face.status['lips']],
                'pix_points': p_face.pix_points,
                'angles': [round_num(p_face.alpha) + 90, round_num(p_face.beta) + 90, round_num(p_face.gamma)],
                'bb': {'xMin': p_face.bb_p1[0], 'xMax': p_face.bb_p2[0], 'yMin': p_face.bb_p1[1],
                       'yMax': p_face.bb_p2[1], 'width': p_face.delta_x, 'height': p_face.delta_y,
                       'center': [p_face.bb_p1[0] + round_num(p_face.delta_x / 2),
                                  p_face.bb_p2[0] + round_num(p_face.delta_y / 2)]},
            }
            face_dict_list.append(face_dict)

    return face_dict_list


def distance(point1, point2):
    return math.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2 + (point2[2] - point1[2]) ** 2)


def extract_index(path):
    file_name = path.split('/').pop()
    print('fileName', file_name)
    replaced = re.sub(r'\D', '', file_name)
    print('replaced', replaced)
    num = None
    if replaced != '':
        num = int(replaced) - 1
    return num


def dot_product(v1, v2):
    # Compute the dot product of two vectors
    return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]


def readb64(base64_string):
    idx = base64_string.find('base64,')
    base64_string = base64_string[idx + 7:]

    sbuf = io.BytesIO()

    sbuf.write(base64.b64decode(base64_string, ' /'))
    pimg = Image.open(sbuf)
    return cv2.cvtColor(np.array(pimg), cv2.COLOR_RGB2BGR)


def round_num(value):
    x = floor(value)
    if (value - x) < .50:
        return x
    else:
        return ceil(value)


def import_image(path):
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise Exception(f'Image not found: {path}')
    if image.ndim == 2:
        image = np.expand_dims(image, axis=-1)
    return image


def view_image(image):
    cv2.imshow('image', image)
    cv2.waitKey(0)
    # cv2.destroyAllWindows()


def view_images_together(images):
    for i, image in enumerate(images):
        cv2.imshow(f'image{i}', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def hls_channels(image):
    hls_image = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    h, l, s = cv2.split(hls_image)
    return h, l, s


def LAB_channels(image):
    l, a, b = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2LAB))
    return l, a, b


def hls_to_bgr(h_channel, s_channel, l_channel):
    return cv2.cvtColor(cv2.merge((h_channel, s_channel, l_channel)), cv2.COLOR_HLS2BGR)


def preprocess_image(image):
    # Check if the image dtype is not uint8
    if image.dtype != np.uint8:
        # Scale the image to the range [0, 255]
        scaled_img = cv2.convertScaleAbs(image, alpha=(255.0 / image.max()))

        # Convert the scaled image to uint8
        image = np.uint8(scaled_img)
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        if image.shape[2] == 1 :
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        # Check the number of channels in the image
        if image.shape[2] == 3:
            # Image is either BGR or RGB
            # Check if the first channel is Red (indicating RGB)
            if image[:, :, 2].mean() > image[:, :, 0].mean():
                print("Image is in RGB color format")
            else:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                print("Image is in BGR color format")
        else:
            print("Image does not have 3 channels, cannot determine color format")

    return image