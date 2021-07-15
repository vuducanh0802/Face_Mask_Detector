# How to use
# python detect_mask_video.py --video examples/face_mask.mp4

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2
import os
from detect_hand import handDetector

def rescaleimage(frame, mark = 1000):
    if int(frame.shape[1]) > mark:
        width = mark
        height = int(frame.shape[0] * mark) // int(frame.shape[1])
        dimension = (width, height)
        return cv2.resize(frame, dimension, interpolation=cv2.INTER_AREA)
    else:
        return frame

def detect_and_predict_mask(frame, faceNet, maskNet):
    # grab the dimensions of the frame and change blod
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
                                 (104.0, 177.0, 123.0))

    #detect face
    faceNet.setInput(blob)
    detections = faceNet.forward()

    # initialize list faces, locations and predicts from model
    faces = []
    locs = []
    preds = []

    # detections
    for i in range(0, detections.shape[2]):
        # extract the confidence
        confidence = detections[0, 0, i, 2]

        # choose confidence that > threshold
        if confidence > args["confidence"]:
            # calculate bbox
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # assure bbox in frame
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            # ROI, BGR to RGB, resize to 224x224 and preprocess
            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)

            # add face and bounding box into list
            faces.append(face)
            locs.append((startX, startY, endX, endY))

    # prediction only if detect at least 1 face
    if len(faces) > 0:
        # predict all face for efficiency
        faces = np.array(faces, dtype="float32")
        preds = maskNet.predict(faces, batch_size=32)

    # (location, predict)
    return (locs, preds)


# initialization
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", required=True,
            help="path to input video")
ap.add_argument("-f", "--face", type=str,
                default="face_detector",
                help="path to face detector model directory")
ap.add_argument("-m", "--model", type=str,
                default="mask_detector.model",
                help="path to trained face mask detector model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
                help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# load face detector model from face_detector folder
print("[INFO] loading face detector model...")
prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
weightsPath = os.path.sep.join([args["face"],
                                "res10_300x300_ssd_iter_140000.caffemodel"])
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# load trained model
print("[INFO] loading face mask detector model...")
maskNet = load_model(args["model"])

# load input video and process
print("[INFO] starting video ...")
vs = VideoStream(src=args["video"]).start()
time.sleep(2.0)


while True:
    # resize

    frame = vs.read()
    frame = imutils.resize(frame, width=800)

    # detect faces in the frame and predict
    (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)
    try:
        # for each detected face
        for (box, pred) in zip(locs, preds):
            # unpack the bounding box and predictions
            (startX, startY, endX, endY) = box
            (mask, withoutMask) = pred

            # class label and color
            # Frontend Stuff
            label = "Mask" if mask > withoutMask else "No Mask"
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

            detector = handDetector()
            frame = detector.findHand(frame)
            if str(type(detector.where(frame))) == "<class 'tuple'>":
                if endX > float(detector.where(frame)[0]) * frame.shape[1] > startX and endY > float(
                        detector.where(frame)[1]) * frame.shape[0] > startY:
                    label = "Had hand covered face, cannot detect"
                    color = (0, 0, 255)
                    prob = 0

            label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

            cv2.putText(frame, label, (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

    except TypeError:
        pass
    cv2.imshow("FACE_MASK_DETECTOR_TL", rescaleimage(frame))
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

# cleanup
cv2.destroyAllWindows()
vs.stop()
