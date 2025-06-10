import cv2
import numpy as np


white = [[0, 0, 170], [180, 30, 255], "masks\\white", 5000]
yellow = [[0, 70, 170], [60, 255, 255],"masks\\yellow",5000]
black = [[0, 0, 80], [200, 20, 130], "masks\\black",500000]
########hsv 色相，饱和度，色明度

def showPic(img):
    cv2.namedWindow('road', cv2.WINDOW_NORMAL)
    cv2.imshow('road', img)
    cv2.waitKey(0)

road = './6155c21bdd6dd3931418ebe5ee9b6c5.jpg'
img = cv2.imread(road)
lower = [0, 70, 170]
height = [60, 255, 255]
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
# lower_white = np.array(lower)    # H范围0~180, S和V范围0~255
# upper_white = np.array(height)
# mask_color = cv2.inRange(hsv, lower_white, upper_white)
white = cv2.inRange(hsv, np.array(white[0]), np.array(white[1]))
yellow = cv2.inRange(hsv, np.array(yellow[0]), np.array(yellow[1]))
black = cv2.inRange(hsv, np.array(black[0]), np.array(black[1]))
showPic(white)
showPic(yellow)
showPic(black)
white[white>0] = 1
yellow[yellow>0] = 1
black[black>0] = 1
mask_color = white+ yellow + black
mask_color[mask_color > 0] = 255

showPic(mask_color)


blurred = cv2.GaussianBlur(mask_color, (5, 5), 0)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (6,6))
img = blurred
img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=2)
img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=2)
img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=2)
# #
img[img < 60] = 0
showPic(img)
# blurred = cv2.GaussianBlur(mask_color, (5, 5), 0)
# img = blurred
# cv2.namedWindow('road', cv2.WINDOW_NORMAL)
# cv2.imshow('road', mask_color)
# cv2.waitKey(0)
# exit(0)


blurred = cv2.GaussianBlur(mask_color, (3, 3), 0)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
img = blurred
img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=2)
img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=2)
img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=2)

# img[img>0] = 255
# 轮廓面积过滤
contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
filtered_mask = np.zeros_like(img)
min_area = 5000*200  # 根据实际场景调整

for cnt in contours:
    area = cv2.contourArea(cnt)
    if area > min_area:
        # 可选：长宽比过滤
        # x, y, w, h = cv2.boundingRect(cnt)
        # aspect_ratio = w / h if h != 0 else 0
        # if aspect_ratio > 5 or aspect_ratio < 0.2:
        cv2.drawContours(filtered_mask, [cnt], -1, 255, -1)

img = filtered_mask

img[img>100] = 255
# 腐蚀膨胀一下
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))  # 椭圆形适应曲线

# 执行闭合操作（先膨胀后腐蚀）
img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
img = cv2.dilate(img, kernel, iterations=2)  # iterations控制膨胀次数
cv2.namedWindow('road', cv2.WINDOW_NORMAL)
cv2.imshow('road', img)
cv2.waitKey(0)

