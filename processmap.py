import json
import os
from os import path
from stitching import Stitcher
import cv2
import os
from os import path
import time
cv2.ocl.setUseOpenCL(False)

original_settings = {
    "final_megapix": -1,
    "detector": "akaze", # orb sift brisk akaze
    # "matcher_type": "affine",
    "matcher_type": "homography",
    "confidence_threshold": 0.995,
    "match_conf": 0.65, # 0.65
    "range_width": "10",
    "finder": "dp_colorgrad",
    "nfeatures": 2048,
    "adjuster": "ray",
    "wave_correct_kind": "no",
    "compensator":"no",
    "try_use_gpu": True,
    "matches_graph_dot_file": "matches_graph.txt",
    "warper_type": "plane",
    "crop": False,
    "timelapse": "no",
}

sift_settings = {
    "final_megapix": -1,
    "detector": "sift", # orb sift brisk akaze
    # "matcher_type": "affine",
    "matcher_type": "knnmatch",
    "confidence_threshold": 0.45,
    "match_conf": 0.65, # 0.65
    "range_width": "10",
    "finder": "dp_colorgrad",
    "nfeatures": 2048,
    "adjuster": "ray",
    "wave_correct_kind": "no",
    "compensator":"no",
    "try_use_gpu": True,
    "matches_graph_dot_file": "matches_graph.txt",
    "warper_type": "plane",
    "crop": False,}

stitcher = Stitcher(**sift_settings)


base= '/home/zhaoruopeng/pics/thumbnail/'
thumb_pair = {}
with open('map.txt') as f:
    lines = f.readlines()
    
    for line in lines:
        value, key = line.strip().split(' ')
        value = base + value
        if key in thumb_pair:
            thumb_pair[key].append(value)
        else:
            thumb_pair[key] = [value]

print("process map.txt success!---------------")

errors = open("errors.txt", "w")

current = time.time()


save_path = '/home/zhaoruopeng/pics/connect'
for key,value in thumb_pair.items():
    file_name = key
    file_path = path.join(save_path, file_name)
    if path.exists(file_path):
        continue
    print(f"start to stitch file : {file_name} , file number is: {len(value)}" )
    try:
        panorama = stitcher.stitch_crack(value)
        cv2.imwrite(file_path, panorama)
    except:
        errors.writelines(f"stitch {file_name} error, files is {value}")
    finally:
        continue
    # 如果有cameras.json数据就可以不执行下面这句了，只执行下下句代码
    now = time.time()
    cost = now - current
    current = now
    print(f"stitch {file_name} success, cost time: {cost}")

print("**************************end*************************")

