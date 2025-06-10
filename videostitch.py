from stitching import Stitcher
import cv2
import os
from os import path
cv2.ocl.setUseOpenCL(False)

# settings = {
#     "final_megapix": -1,
#     "detector": "akaze", # orb sift brisk akaze
#     "matcher_type": "affine",
#     "confidence_threshold": 0.995,
#     # "match_conf": 0.8, # 0.65
#     "range_width": "homography",
#     "finder": "dp_colorgrad",
#     "nfeatures": 2048,
#     "adjuster": "reproj",
#     "wave_correct_kind": "no",
#     "compensator":"no",
#     "try_use_gpu": True,
#     "matches_graph_dot_file": "matches_graph.txt",
#     "warper_type": "plane",
#     "crop": False,
# }
# settings = {"wave_correct_kind": "no","compensator":"no", "crop":False, "timelapse": "as_is"}
settings = {"wave_correct_kind": "no","compensator":"no", "crop":False}
stitcher = Stitcher(**settings)



dir  = path.abspath('./output_frames')

files = os.listdir(dir)
images = []
count = 0
for file in files:
    images.append(path.join(dir, file))
    count+=1
    if count == 15:
        break
file = stitcher.stitch(images)
cv2.imwrite("videoStitch.jpg", file)