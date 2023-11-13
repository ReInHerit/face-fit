import cv2
import requests
import json
import os
import Face_Maker as F_obj
from math import floor, ceil
# from Face_Maker import get_landmarks, get_face_angles, check_expression
directoryPath = "../images"  # Replace with the actual directory path
ref_images = []
ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))
right_eye = [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7, 33];
left_eye = [362, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 382, 362];
mouth = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 78];
nose1 = [240, 97, 2, 326, 327];
nose2 = [2, 94, 19, 1, 4, 5, 195, 197, 6, 168, 8, 107, 66, 105, 63, 70];
nose3 = [8, 336, 296, 334, 293, 300];
# Create a set with a set comprehension
full_indices_set = set(
    index
    for indices in [right_eye, left_eye, mouth, nose1, nose2, nose3]
    for index in indices
)

# Convert the result to a list if needed
full_indices_list = list(full_indices_set)
def round_num(value):
    x = floor(value)
    if (value - x) < .50:
        return x
    else:
        return ceil(value)

try:
    for filename in os.listdir(directoryPath):
        filePath = os.path.join(directoryPath, filename)
        if os.path.isfile(filePath):
            ref_images.append('images/' + filename)
except OSError as err:
    print('Unable to scan directory:', err)
# Your start_message JSON data
  # Replace with your actual JSON data
ref_dict = []
w = 0
h = 0
for idx, file in enumerate(ref_images):
    ref_img = cv2.imread(os.path.join(ROOT_DIR, file))
    w, h = ref_img.shape[0], ref_img.shape[1]
    p_face = F_obj.Face('ref')
    p_face.get_landmarks(ref_img)
    face_dict = {
         'which': p_face.which,
         'id': idx,
         'src': file,
         'points': p_face.points,
         'expression': [p_face.status['l_e'], p_face.status['r_e'], p_face.status['lips']],
         'angles': [round_num(p_face.alpha) + 90, round_num(p_face.beta) + 90, round_num(p_face.gamma)],
         'bb': {'xMin': p_face.bb_p1[0], 'xMax': p_face.bb_p2[0], 'yMin': p_face.bb_p1[1],
                'yMax': p_face.bb_p2[1], 'width': p_face.delta_x, 'height': p_face.delta_y,
                'center': [p_face.bb_p1[0] + round_num(p_face.delta_x / 2),
                           p_face.bb_p2[0] + round_num(p_face.delta_y / 2)]},

         }
    ref_dict.append(face_dict)

for ref in ref_dict:
    points_2d = {}
    for index, land in enumerate(ref["points"]):
        if index in full_indices_set:
            points_2d[f'lmrk{index}'] = [round_num(land[0] * w),
                                         round_num(land[1] * h)]

        if index in {10, 133, 152, 226, 362, 446}:
            points_2d[f'lmrk{index}'] = [round_num(land[0] * w),
                                         round_num(land[1] * h),
                                         round_num(land[2])]
    ref['points'] = points_2d

with open('../json/ref_images.json', 'w') as json_file:
    json.dump(ref_dict, json_file)
print("ref_images exported to ref_images.json")
