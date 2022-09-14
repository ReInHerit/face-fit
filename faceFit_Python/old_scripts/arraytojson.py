import cv2
import numpy as np
import glob
import json
import os
# with open('tris.json', 'r') as f:
#     media_pipes_tris = json.load(f)

input_file = open ('tris.json')
json_array = json.load(input_file)
store_list = []
print(len(json_array))
for i in range(0, int(len(json_array) / 3)):
    points = [json_array[i * 3], json_array[i * 3 + 1], json_array[i * 3 + 2]]
    store_list.append(points)
with open("../triangles2.json", 'w') as file: json.dump(store_list, file)
