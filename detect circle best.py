import numpy as np
import cv2

img = cv2.imread('D:\line_ransac.png')
cv2.imshow("input", img)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.medianBlur(gray, 9)
edges = cv2.Canny(gray, 255, 125)
cv2.imshow("canny", edges)

minDist = 8
param1 = 35
param2 = 35/4
minRadius = 3
maxRadius = 8

# docstring of HoughCircles: HoughCircles(image, method, dp, minDist[, circles[, param1[, param2[, minRadius[, maxRadius]]]]]) -> circles
circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, 1.5, minDist, param1=param1, param2=param2, minRadius=minRadius, maxRadius=maxRadius)

count = 0
if circles is not None:
    circles = np.int32(np.around(circles))
    for i in circles[0,:]:
        cv2.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 2)
        cv2.circle(img, (i[0], i[1]), 2, (0, 0, 255), 0)
        count += 1
else:
    print("NoneType Error!")

# Show result for testing:
print("count = ", count)
cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# this model detected 65/68 circles 95% accurate