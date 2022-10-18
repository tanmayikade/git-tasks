import cv2

img = cv2.imread('D:\\Images\\car_lane.jpg')
img = cv2.resize(img, (700, 500))

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)

light_blue = (100, 0, 0)
dark_blue = (128, 128, 128)

mask = cv2.inRange(img_hsv, light_blue, dark_blue)  # light is lower boundary for threshold region ; dark is upper boundary
result = cv2.bitwise_and(img, img, mask=mask)

ret, thresh = cv2.threshold(result, 0, 265, cv2.THRESH_BINARY)

if not ret:
    print('Threshold Successful!')

cv2.imshow('Thresh', thresh)
cv2.imshow('Input', img)
cv2.imshow('Masked', result)
cv2.waitKey(0)
cv2.destroyAllWindows()