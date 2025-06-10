import cv2
import numpy as np
from os import path
import os

'''
场景	黄色 HSV 范围	黑色 HSV 范围
晴天白天	[15, 100, 150]~[30, 255, 255]	[0, 0, 0]~[180, 50, 40]
阴天/低光照	[20, 50, 80]~[30, 255, 200]	[0, 0, 0]~[180, 50, 60]
夜间（有路灯）	[20, 100, 50]~[30, 255, 150]	[0, 0, 0]~[180, 50, 30]
'''
lower = [0, 0, 0]
height = [180, 50, 40]
white = [[0, 0, 170], [180, 30, 255], "masks", 5000]
yellow = [[0, 70, 170], [60, 255, 255],"masks",5000]
black = [[0, 0, 40], [200, 30, 120], "masks",500000]
def process(road, lower, height, threshold = 5000):
    # road = '../wandao/DJI_20240511161259_0060.JPG'
    img = cv2.imread(road)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    #公路面
    # lower = [0, 0, 40]
    # height = [200, 30, 120]
    #黄线
    # lower = [0, 70, 170]
    # height = [60, 255, 255]

    # 公路白线
    # lower_white = np.array([0, 0, 170])    # H范围0~180, S和V范围0~255
    # upper_white = np.array([180, 30, 255])
    lower_white = np.array(lower)
    upper_white = np.array(height)
    mask_color = cv2.inRange(hsv, lower_white, upper_white)
    blurred = cv2.GaussianBlur(mask_color, (3, 3), 0)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (6,6))
    img = blurred
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=2)
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=2)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=2)
    #
    img[img < 60] = 0

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
    min_area = threshold  # 根据实际场景调整,过滤掉小于该面积得区域

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
    return img
    cv2.namedWindow('road', cv2.WINDOW_NORMAL)
    cv2.imshow('road', img)
    cv2.waitKey(0)

dir  = path.abspath('../101')
file_ext = '.JPG'

files = os.listdir(dir)
images = []
for file in files:
    if file.endswith(file_ext):
        images.append(path.join(dir, file))


for road in images:
    param = yellow
    img = process(road, param[0], param[1], param[3])
    parent = path.dirname(road)
    filename = path.basename(road)
    parent = path.join(parent, param[2])
    if not path.exists(parent):
        os.mkdir(parent)
    savefile = path.join(parent, filename)
    # road = road.replace('wandao', ''+param[2])
    print(savefile)
    cv2.imwrite(savefile, img)

# files = './6155c21bdd6dd3931418ebe5ee9b6c5.jpg'
# param = black
# img = process(files, param[0], param[1], param[3])
# cv2.imwrite("first_100.jpg", img)
