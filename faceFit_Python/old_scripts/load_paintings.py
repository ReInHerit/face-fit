import os
import glob
import cv2
import mediapipe as mp
import Face as F_obj

full_files = []
ref_objs = []
try:
    root = os.path.dirname(os.path.abspath(__file__))
except NameError:
    import sys
    root = os.path.dirname(os.path.abspath(sys.argv[0]))

project_path = root
ref_path = project_path + '/images/'
full_painting_path = ref_path + 'full_images/'


for filename in glob.iglob(f'{full_painting_path}*.jpg'):
    full_files.append(filename)
for idx, file in enumerate(full_files):
    temp_img = cv2.imread(file)
    ref_objs.append(F_obj.Face('ref'))
    ref_objs[idx].get_landmarks(temp_img)
    ref_objs[idx].draw('contours')
    cv2.imshow('', ref_objs[idx].image)
    cv2.waitKey(0)



print(full_files)
