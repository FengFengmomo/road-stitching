import cv2

file = './wandao/DJI_20240511161258_0059.JPG'

img = cv2.imread(file)
img_r = cv2.resize(img, (img.shape[1]//2, img.shape[0]//2))
cv2.imwrite('./wandao/DJI_20240511161258_0059_r.JPG', img_r)