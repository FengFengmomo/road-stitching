import numpy as np
import cv2

# 回调函数，未使用
def nothing(x):
    pass

# 将BGR图像转化为HSV图像
road = '../s101_wandao/DJI_20240915170208_0529.JPG'
win_img = "new"
win_img_old = "old"
pic = cv2.imread(road) # 自己想要分析的照片
pic = cv2.cvtColor(pic, cv2.COLOR_BGR2HSV)
h,s,v = cv2.split(pic)

histSize = 256
histRange = (0, histSize)
b_hist = cv2.calcHist([h], [0], None, [histSize], histRange)
g_hist = cv2.calcHist([s], [0], None, [histSize], histRange)
r_hist = cv2.calcHist([v], [0], None, [histSize], histRange)
# 显示原图像做对比
cv2.namedWindow(win_img_old, cv2.WINDOW_NORMAL)
cv2.imshow(win_img_old, b_hist)

# 新图像窗口
cv2.namedWindow(win_img, cv2.WINDOW_NORMAL)
#初始化滚动条
cv2.createTrackbar("H_min",win_img, 10, 255, nothing)
cv2.createTrackbar("S_min",win_img, 10, 255, nothing)
cv2.createTrackbar("V_min",win_img, 10, 255, nothing)
cv2.createTrackbar("H_max",win_img, 100, 255, nothing)
cv2.createTrackbar("S_max",win_img, 100, 255, nothing)
cv2.createTrackbar("V_max",win_img, 100, 255, nothing)

while True:
	# ESC按下退出
    if cv2.waitKey(10) == 27:
        print("finish adjust picture and quit")
        break

	# 读取滚动条现在的滚动条的HSV信息
    h_min = int(cv2.getTrackbarPos("H_min",win_img))
    s_min = int(cv2.getTrackbarPos("S_min",win_img))
    v_min = int(cv2.getTrackbarPos("V_min",win_img))
    h_max = int(cv2.getTrackbarPos("H_max",win_img))
    s_max = int(cv2.getTrackbarPos("S_max",win_img))
    v_max = int(cv2.getTrackbarPos("V_max",win_img))
	# 拆分、读入新数据后，重新合成调整后的图片
    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])
    print(lower, upper)
    mask = cv2.inRange(pic, lower, upper)
    cv2.imshow(win_img, mask)

cv2.destroyAllWindows()
