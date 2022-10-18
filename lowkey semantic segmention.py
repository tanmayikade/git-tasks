import cv2

img = cv2.imread('D:\AGV Images\semantic\Input Semantic\\erfurt.png')
img = cv2.resize(img, (700, 500))
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_blur_gauss = cv2.GaussianBlur(img, (5,5), 0)

img_rgb = cv2.cvtColor(img_blur_gauss, cv2.COLOR_BGR2RGB)
img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)

#light_blue = (90, 0, 0)
light_blue = (57, 0, 0)  # 100 much better than above for car_lane.jpg
dark_blue = (128, 128, 128)

mask = cv2.inRange(img_hsv, light_blue, dark_blue)  # light is lower boundary for threshold region ; dark is upper boundary
result = cv2.bitwise_and(img, img, mask=mask)

ret, thresh = cv2.threshold(result, 0, 265, cv2.THRESH_BINARY)
ulta = cv2.bitwise_not(thresh)

if not ret:
    print('Threshold Successful!')

cv2.imshow('Thresh', thresh)
cv2.imwrite('D:\AGV Images\semantic\Output Semantic\\erfurt_output.jpg', thresh)

#cv2.imshow('Ulta', ulta)
cv2.imshow('Input', img)
cv2.imshow('Masked', result)
cv2.waitKey(0)
cv2.destroyAllWindows()
