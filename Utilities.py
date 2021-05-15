######################
# IMPORT MODULES
######################
import cv2
import numpy as np
import djitellopy
import face_recognition



######################
# DEFINE FUNCTIONS
######################=
# INITIALIZE TELLO
def initializeTello():
    myDrone = djitellopy.Tello()
    myDrone.connect()
    myDrone.for_back_velocity = 0
    myDrone.left_right_velocity = 0
    myDrone.up_down_velocity = 0
    myDrone.yaw_velocity = 0
    print(myDrone.get_battery())
    myDrone.streamoff()
    myDrone.streamon()
    return myDrone


# GET TELLO FRAME
def getTelloFrame(myDrone, w, h):
    myFrame = myDrone.get_frame_read()
    myFrame = myFrame.frame
    img = cv2.resize(myFrame, (w, h))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

# FIND ENCODINGS
def findEncodings(images):
    encodeList = []
    for img in images:
        print(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[1]
        encodeList.append(encode)
    return encodeList


# FIND FACES
def findFace(img):
    faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(imgGray, 1.1, 6)

    myFaceListArea = []
    myFaceListC = []
    for (x, y, w, h) in faces:
        area = w*h
        cx = x + w//2
        cy = y + h//2
        myFaceListArea.append(area)
        myFaceListC.append([cx, cy])

    if len(myFaceListArea) != 0:
        i = myFaceListArea.index(max(myFaceListArea))
        return img, [myFaceListC[i], myFaceListArea[i]]
    else:
        return img, [[0, 0], 0]


# FACE RECOGNITION
def faceRecognition(img, status, encodeListKnown, classNames):
    imgS = img.copy()
    imgS = cv2.resize(imgS, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    faceLocFrames = face_recognition.face_locations(imgS)
    encodeFaceFrames = face_recognition.face_encodings(imgS, faceLocFrames)

    for encodeFace, faceLoc in zip(encodeFaceFrames, faceLocFrames):

        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        # print("Face Distance:", faceDis)


        matchIndex = np.argmin(faceDis)
        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
            if status == "happy":
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
                cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            elif status == "neutral":
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 0, 255), cv2.FILLED)
                cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            elif status == "angry":
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (255, 0, 0), cv2.FILLED)
                cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
        else:
            name = ("Desconocido").upper()
            # print("Name:", name)
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (255, 0, 0), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img



# TRACK FACES
def trackFaces(myDrone, info, w, h, pid, pYaw_error, pUp_down_error):
    print("INFO:", info)
    yaw_error = info[0][0] - w//2
    yaw_speed = pid[0]*yaw_error + pid[1]*(yaw_error-pYaw_error)
    yaw_speed = int(np.clip(yaw_speed, -100, 100))

    up_down_error = info[0][1] - h//2
    up_down_speed = pid[0]*up_down_error + pid[1]*(up_down_error-pUp_down_error)
    up_down_speed = int(np.clip(up_down_speed, -100, 100))
    up_down_speed = up_down_speed*-1
    up_down_speed = int(up_down_speed*0.65)

    if info[0][0] != 0:
        if yaw_speed > 0:
            print("GO RIGHT:", yaw_speed)
        if yaw_speed < 0:
            print("GO LEFT:", yaw_speed)
        if up_down_speed > 0:
            print("GO UP:", up_down_speed)
        if up_down_speed < 0:
            print("GO DOWN:", up_down_speed)
        # myDrone.yaw_velocity = yaw_speed
        # myDrone.up_down_velocity = up_down_speed


    else:
        myDrone.for_back_velocity = 0
        myDrone.left_right_velocity = 0
        myDrone.up_down_velocity = 0
        myDrone.yaw_velocity = 0
        error = 0



    if myDrone.send_rc_control:
        myDrone.send_rc_control(myDrone.for_back_velocity,
                                myDrone.left_right_velocity,
                                myDrone.up_down_velocity,
                                myDrone.yaw_velocity)

    return yaw_error, up_down_error





















