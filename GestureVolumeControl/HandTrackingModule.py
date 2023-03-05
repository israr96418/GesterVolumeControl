# HandTracking module--> Now we can use it multiple project
# Thats way i create it as a module

import time

import cv2
import mediapipe as mp

class handDetector():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5, mode_complixity=0):

        self.mode = mode
        self.maxHands = 2
        self.detectionCon = int(detectionCon)
        self.trackCon = int(trackCon)
        self.mode_complixity = mode_complixity
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands,
                                        self.detectionCon, self.trackCon)

        self.mp_drawing_utils = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

    def findHands(self, img, draw=True):
        img.flags.writeable = False  # to improve model performance , optionally mark that image is not writeable
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        img.flags.writeable = True
        if self.results.multi_hand_landmarks:
            for handlms in self.results.multi_hand_landmarks:
                if draw:
                    self.mp_drawing_utils.draw_landmarks(img,
                                                         handlms, self.mpHands.HAND_CONNECTIONS,
                                                         self.mp_drawing_utils.DrawingSpec(color=(0, 0, 255),
                                                                                           thickness=3,
                                                                                           circle_radius=3),
                                                         self.mp_drawing_utils.DrawingSpec(color=(255, 0, 255),
                                                                                           thickness=2,
                                                                                           circle_radius=2))
                    # self.mp_drawing_styles.get_default_hand_connections_style()
                    # self.mp_drawing_styles.get_default_hand_landmarks_style()

        return img

    def findPosition(self, img, draw=True):
        """we have total 21 landmarks, each of have x,y z and having value in pixel
        we need to convert it into decemial by multiplying with its Height, Weights"""
        self.lmlist =[]

        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[0]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                #                 print(id, cx, cy)
                self.lmlist.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 15,
                               (255, 0, 255), cv2.FILLED)

                # it will draw circle on specific landmard and we known that every landmark having specific index
                # total index = 21
                # if id == 0:
                #     cv2.circle(img, (cx, cy), 15,
                #                (255, 0, 255), cv2.FILLED)
                # if len(lmlist != 0):

        return self.lmlist

    def FingerUp(self):
        tipIds = [4, 8, 12, 16, 20]
        fingers = []
        # Thumb
        if self.lmlist[tipIds[0]][1] < self.lmlist[tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)
        # 4 Finger
        for id in range(1, 5):
            if self.lmlist[tipIds[id]][2] < self.lmlist[tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        # print(fingers)
        totalFinger = fingers.count(1)
        # print(totalFinger)
        return fingers


def main():
    pTime = 0
    cTime = 0

    cap = cv2.VideoCapture(0)
    detector = handDetector()

    while cap.isOpened():
        success, img = cap.read()
        if not success:
            print("Ignoring for empty frame")
            continue
        img = detector.findHands(img)
        lmlist = detector.findPosition(img)
        if len(lmlist) != 0:
            print(lmlist[4])

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 70),
                    cv2.FONT_HERSHEY_COMPLEX, 3, (255, 0, 255), 3)
        cv2.imshow("Image", img)
        # cv2.waitKey(1)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Stop the camera
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main();
