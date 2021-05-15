import cv2
import matplotlib.pyplot as plt
from deepface import DeepFace
from Utilities import *


capture = cv2.VideoCapture(0)

while True:
    isTrue, frame = capture.read()
    # frame = cv2.imread("ImagesAttendace/Elon Musk.jfif")
    expressions = DeepFace.analyze(frame, actions=['emotion'])
    status = expressions["dominant_emotion"]
    frame, info = findFace(frame, status)
    delay = 1
    cv2.imshow("ELon Musk", frame)
    cv2.waitKey(delay)







