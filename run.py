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
    "medium_megapix": 1.0,
    # "low_megapix":0.1,
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


test = True
test_number = 14 # 最大测试图片数量
# 计算程序运行时间， 按照秒计算

start = time.time()


base_dir  = path.abspath('./wandao')
# base_dir  = path.abspath('E:\\test_wandao11')

files = os.listdir(base_dir)
images = []
number = 0
for file in files:
    if file.endswith(".JPG"):
        images.append(path.join(base_dir, file))
        number+=1
    if test and number>=test_number:
        break

dir = path.join(base_dir, 'masks')

masks = []
if os.path.exists(dir):
    files = os.listdir(dir)
    number = 0
    for file in files:
        if file.endswith(".JPG") or file.endswith(".png"):
            masks.append(path.join(dir, file))
            number+=1
        if test and number>=test_number:
            break

dir = path.join(base_dir, 'cracks')
cracks = []
if os.path.exists(dir):
    files = os.listdir(dir)
    number = 0
    for file in files:
        if file.endswith(".png"):
            cracks.append(path.join(dir, file))
            number+=1
        if test and number>=test_number:
            break


# panorama, panorama_with_seam_lines, with_seam_polygons = stitcher.stitch(images, road_masks_img=masks)
# 如果有cameras.json数据就可以不执行下面这句了，只执行下下句代码
# stitcher.produce_stitch_cameras_data(images, road_masks_img=masks, crack_imgs = cracks)
panorama, panorama_with_seam_lines = stitcher.stitch_with_camera_data(useCorrect=True)
cv2.imwrite("stitch10.jpg", panorama)
cv2.imwrite("stitch10_with_seam_lines.jpg", panorama_with_seam_lines)
# cv2.imwrite("wd11_error.jpg", panorama)
# cv2.imwrite("wd11_error_lines.jpg", panorama_with_seam_lines)
# panorama, panorama_with_seam_lines = stitcher.stitch_with_camera_data(useCorrect=True)
# cv2.imwrite("wd11_perfect.jpg", panorama)
# cv2.imwrite("wd11_perfect_lines.jpg", panorama_with_seam_lines)
# stitcher.stitch_verbose(images, verbose_dir="verbose")
end = time.time()
print("程序运行时间：", end - start)