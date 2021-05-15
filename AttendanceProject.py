import os
import cv2
import face_recognition
import numpy as np











cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()





    delay = 1
    cv2.imshow("WebCam", img)
    cv2.waitKey(delay)