from stitching import Stitcher
import cv2
import os
from os import path
import time
cv2.ocl.setUseOpenCL(False)

original_settings = {
    "final_megapix": -1,
    "medium_megapix": 0.2,
    "low_megapix":0.1,
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
    "medium_megapix": 0.6,
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



base_dir  = path.abspath('./wandao')

test = True
test_number = 14 # 最大测试图片数量
files = os.listdir(base_dir)
images = []
number = 0
for file in files:
    if file.endswith(".JPG"):
        images.append(path.join(base_dir, file))
        number+=1
    if test and number>=test_number:
        break
start = time.time()
panorama = stitcher.stitch_crack(images)
# 如果有cameras.json数据就可以不执行下面这句了，只执行下下句代码
cv2.imwrite("stitch_panora.jpg", panorama)


# stitcher.stitch_verbose(images, verbose_dir="verbose")
end = time.time()
print("程序运行时间：", end - start)