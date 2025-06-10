import cv2
import numpy as np

road = '../wandao/DJI_20240511161256_0057.JPG'
img = cv2.imread(road)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 对图片中白色部分增亮
img[img >= 150] = 255
img[img < 150] = 0

# 去除图中的不连贯部分

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)


# edges = cv2.Canny(img, 1, 1)


for i in range(4):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
blurred = cv2.GaussianBlur(img, (5, 5), 0)
# edges = cv2.Canny(blurred, 50, 150)
img = blurred

for i in range(4):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

blurred = cv2.GaussianBlur(img, (5, 5), 0)

img = blurred

cv2.namedWindow('road', cv2.WINDOW_NORMAL)
cv2.imshow('road', img)
cv2.waitKey(0)