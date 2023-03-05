import math
import time
from ctypes import cast, POINTER

import HandTracking.HandTrackingModule as htm
import cv2
import numpy as np
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

################################################################
webcamp_height = 640
webcamp_width = 480
################################################################

################################################################
# pycaw python library--> which are used for volume control
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
# volume.GetMute()
# volume.GetMasterVolumeLevel()
volume_Range = volume.GetVolumeRange()
min_volume = volume_Range[0]
max_volume = volume_Range[1]

###############################################################
cap = cv2.VideoCapture(0)
cap.set(3, webcamp_width)
cap.set(4, webcamp_height)

htm_moduel = htm.handDetector(detectionCon=0.7)
pTime = 0
volBar = 400
volPer = 0

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty frame")
        # if loading video, then used break instead of continue
        continue
    image = htm_moduel.findHands((image))
    lmList = htm_moduel.findPosition(image, draw=False)
    if len(lmList) != 0:
        # print(lmList[4], lmList[8])
        x1, y1 = lmList[4][1], lmList[4][2]
        x2, y2 = lmList[8][1], lmList[8][2]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.circle(image, (x1, y1), 10, (255, 0, 255), cv2.FILLED)
        cv2.circle(image, (x2, y2), 10, (255, 0, 255), cv2.FILLED)
        cv2.line(image, (x1, y1), (x2, y2), (255, 0, 255), 3)
        cv2.circle(image, (cx, cy), 10, (0, 0, 255), cv2.FILLED)

        # calculate length of line
        lenght = math.hypot(x2 - x1, y2 - y1)
        # print(lenght)
        # Hand Range(30 to 220)
        # Volume range (-65.23 0.0)
        vol = np.interp(lenght, [30, 220], [min_volume, max_volume])
        volBar = np.interp(lenght, [30, 280], [400, 150])
        volPer = np.interp(lenght, [30, 280], [0, 100])
        volume.SetMasterVolumeLevel(vol, None)
        print(int(lenght), vol)
        if lenght < 30:
            cv2.circle(image, (cx, cy), 10, (255, 255, 255), cv2.FILLED)

    cv2.rectangle(image, (50, 150), (85, 400), (255, 0, 0), cv2.FILLED)
    cv2.rectangle(image, (50, int(volBar)), (85, 400), (0, 0, 255), cv2.FILLED)
    cv2.putText(image, f"{int(volPer)}%", (40, 450), cv2.FONT_HERSHEY_PLAIN, 2,
                (255, 0, 0), 2)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(image, f"FPS: {int(fps)}", (20, 70), cv2.FONT_HERSHEY_PLAIN, 2,
                (255, 0, 0), 2)
    cv2.imshow("VolumeControlByHands: ", image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
