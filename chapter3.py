import cv2
import numpy as np

img = cv2.imread("Resources/lambo.PNG")
print(img.shape)

imgResize = cv2.resize(img,(1000,500))
print(imgResize.shape)

cv2.imshow("Image", img)
cv2.imshow("Resize", imgResize)

imgCropped = img[0:200,200:500]
cv2.imshow("Cropped", imgCropped)

cv2.waitKey(0)

cv2.destroyAllWindows()
