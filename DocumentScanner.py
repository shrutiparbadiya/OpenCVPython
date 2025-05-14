import cv2
import numpy as np

from chapter2 import imgDialation
from chapter3 import imgCropped

###################
widthImg = 640
heightImg = 480
###################

def preProcessing(img):
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (5,5), 1)
    imgCanny = cv2.Canny(imgBlur, 50, 50)
    kernal = np.ones((5, 5), np.uint8)
    imgDialation = cv2.dilate(imgCanny, kernal, iterations=2)
    imgThreshold = cv2.erode(imgDialation, kernal, iterations=1)

    return imgThreshold

def getContours(img):
    biggest = np.array([])
    maxArea = 0
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        #print(area)

        if area > 5000:
            # cv2.drawContours(imgContour, cnt, -1, (255, 0, 0), 3)
            peri = cv2.arcLength(cnt, True)
            #print(peri)
            approx = cv2.approxPolyDP( cnt, 0.02 * peri, True)
            if area > maxArea and len(approx) == 4:
                biggest = approx
                maxArea = area

    cv2.drawContours(img, biggest, -1, (255, 0, 0), 20)
    return biggest

def reorder (myPoints):
    myPoints = myPoints.reshape((4, 2))
    myPointsNew = np.zeros((4, 1, 2), dtype="int32")
    add = myPoints.sum(1)
    # print("add", add)

    myPointsNew[0] = myPoints[np.argmin(add)]
    myPointsNew[3] = myPoints[np.argmax(add)]



    diff = np.diff(myPoints, axis=1)
    myPointsNew[1] = myPoints[np.argmin(diff)]
    myPointsNew[2] = myPoints[np.argmax(diff)]

    # print("myPointsNew", myPointsNew)
    return myPointsNew
framewidth = 480
frameheight = 640
image = cv2.imread("Resources/paper.jpg")
# cap.set(3, framewidth)
# cap.set(4, frameheight)
# cap.set(10, 150)

def getWarp(img, biggest):
    biggest = reorder(biggest)
    pts1 = np.float32(biggest)
    pts2 = np.float32([[0, 0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    imgOutput = cv2.warpPerspective(img, matrix, (widthImg, heightImg))
    imgCropped = imgOutput[20:imgOutput.shape[0]-20, 20:imgOutput.shape[1]-20]
    imgcropped = cv2.resize(imgCropped, (widthImg, heightImg))
    return imgCropped

# while True:
    # success, img = cap.read()

imageResize = cv2.resize(image, (widthImg, heightImg))
imgContour = imageResize.copy()
imgThreshold = preProcessing(imageResize)
biggest = getContours(imgThreshold)
print(biggest)
WarpedImg = getWarp(imageResize, biggest)
cv2.imshow('frame', WarpedImg)

cv2.waitKey(0)
