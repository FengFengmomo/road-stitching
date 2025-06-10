import cv2
import numpy as np

path = "../masks/yellow/DJI_20240511161256_0057.JPG"
img = cv2.imread(path)

# 定义结构元素（大小根据凹槽宽度调整）
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))  # 椭圆形适应曲线

# 执行闭合操作（先膨胀后腐蚀）
img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
img = cv2.dilate(img, kernel, iterations=2)  # iterations控制膨胀次数


cv2.namedWindow("image", cv2.WINDOW_NORMAL)
cv2.imshow("image", img)
cv2.waitKey(0)