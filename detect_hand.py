import cv2 as cv
import mediapipe as mp


class handDetector():
    def __init__(self,mode=False,maxHands=2,detectionCon=0.5,trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHand = mp.solutions.hands
        self.hands = self.mpHand.Hands(self.mode,self.maxHands,self.detectionCon,self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findHand(self,img,draw=True):
        imgRGB = cv.cvtColor(img,cv.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handsLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img,handsLms,self.mpHand.HAND_CONNECTIONS)
        return img
    def findPosition(self,img,handNo=0,draw = True):
        lmlist = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id,lm in enumerate(myHand.landmark):
                h,w,c = img.shape
                cx,cy = int(lm.x*w),int(lm.y*h)

                lmlist.append([id,cx,cy])
                if draw:
                    cv.circle(img,(cx,cy),5,(255,0,255),2,cv.FILLED)
        return lmlist

    
    #Find Middle Finger MCP
    def where(self,img):
        #Change to RGB
        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)

        self.results = self.hands.process(imgRGB)
        # print(results.multi_hand_landmarks)
        if self.results.multi_hand_landmarks:
            for handsLms in self.results.multi_hand_landmarks:
                return handsLms.landmark[9].x,handsLms.landmark[9].y

