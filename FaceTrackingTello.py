######################
# IMPORT MODULES
######################
from Utilities import *
import os
from deepface import DeepFace


######################
# DEFINE VARIABLES
######################
startContour = 1   # 0 For Flight  -  1 For Testing
w, h = 840, 440
myDrone = initializeTello()
pid = [0.14, 0.14, 0]
pYaw_error = 0
pUp_down_error = 0
pFace = False


######################
# IMAGES PATH
######################
path = 'ImagesAttendace' # YOUR PATH FOR YOUR FACES IMAGES
images = []
classNames = []
myList = os.listdir(path)
print(myList)
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)
encodeListKnown = findEncodings(images)




#######################
# MAIN LOOP
#######################
while True:
    # STEP 1: TAKE OFF AND GET THE TELLO IMG FRAME
    if startContour == 0:
        myDrone.takeoff()
        startContour = 1
    img = getTelloFrame(myDrone, w, h)

    # STEP 2: FIND ALL FACES
    img, info = findFace(img)

    # STEP3: ANALYZE EMOTIONS
    emotions = DeepFace.analyze(img, actions=["emotion"])
    status = emotions["dominant_emotion"]

    # STEP 3: RECOGNIZE FACES
    img = faceRecognition(img, status, encodeListKnown, classNames)

    # STEP 4:
    pYaw_error, pUp_down_error = trackFaces(myDrone, info, w, h, pid, pYaw_error, pUp_down_error)

    # pError = trackFaces(myDrone, info, w, pid, pError)
    cv2.imshow("Tello Video", img)

    if cv2.waitKey(1) and 0xFF == ord("q"):
        myDrone.land()
        break