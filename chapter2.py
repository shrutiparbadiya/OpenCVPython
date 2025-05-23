import cv2
import numpy as np

img = cv2.imread("Resources/lena.png")
kernal = np.ones((5,5),np.uint8)

imgGray =  cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
imgBlur = cv2.GaussianBlur(imgGray, (7, 7), 0)
imgCanny = cv2.Canny(imgBlur, 100, 100)
imgDialation = cv2.dilate(imgCanny, kernal, iterations=1)
imgEroded = cv2.erode(imgDialation, kernal, iterations=1)

# cv2.imshow("gray", imgGray)
# cv2.imshow("Blurred_Image", imgBlur)
# cv2.imshow("Canny", imgCanny)
# cv2.imshow("Dialation Image", imgDialation)
# cv2.imshow("Erosion Image", imgEroded)
cv2.waitKey(0)

cv2.destroyAllWindows()
