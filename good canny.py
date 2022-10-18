import matplotlib.pyplot as py
import cv2
import numpy as np

img = cv2.imread('D:\\Images\\car_lane.jpg')
img = cv2.resize(img, (700, 500))

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_blur_gauss = cv2.GaussianBlur(img_gray, (15,15), 0)

canny_gauss = cv2.Canny(img_blur_gauss, 125, 125/3)

cnt = sorted(cv2.findContours(canny_gauss, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2], key=cv2.contourArea)[-1]
mask = np.zeros((500, 700), np.uint8)
masked = cv2.drawContours(mask, [cnt],-1, 255, -1)

dst = cv2.bitwise_and(img, img, mask= mask)

#cv2.imshow('Gray Input', img_gray)
cv2.imshow('Canny Gauss', canny_gauss)
cv2.imshow('Contours', masked)
cv2.imshow('Segment', dst)
cv2.waitKey(0)
cv2.destroyAllWindows()