import cv2

img = cv2.imread('Resources/lena.png')

cv2.imshow("Original Image", img)
cv2.waitKey(0)

cam = cv2.VideoCapture(0)

while True:
    success, img = cam.read()
    cv2.imshow("Webcam", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap = cv2.VideoCapture("Resources/test_video.mp4")

while True:
    success, img = cap.read()
    cv2.imshow("Video", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
