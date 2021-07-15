from flask import Flask,jsonify,request,render_template, Response
from utils import read_image, prepare_image
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


app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


# Initialization
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--face", type=str,
                default="face_detector",
                help="path to face detector model directory")
ap.add_argument("-m", "--model", type=str,
                default="mask_detector.model",
                help="path to trained face mask detector model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
                help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# load face detector model from folder
print("[INFO] loading face detector model...")
prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
weightsPath = os.path.sep.join([args["face"],
                                "res10_300x300_ssd_iter_140000.caffemodel"])
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# load trained model
print("[INFO] loading face mask detector model...")
maskNet = load_model(args["model"])


def gen():
    # video stream and webcam
    print("[INFO] starting video stream...")
    vs = VideoStream(src=0).start()
    time.sleep(2.0)


    while True:
        frame = vs.read()
        frame = imutils.resize(frame, width=800)

        # detect faces and mask or no mask
        (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

        for (box, pred) in zip(locs, preds):
            # unpack the bounding box and predictions
            (startX, startY, endX, endY) = box
            (mask, withoutMask) = pred

            # class label and color for bounding box and text
            label = "Mask" if mask > withoutMask else "No Mask"
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

            # prob
            label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

            cv2.putText(frame, label, (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

        # show output frame
        cv2.imshow("FACE_MASK_DETECTOR_TL", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
    # cleanup
    cv2.destroyAllWindows()
    vs.stop()

@app.route('/webcam')
def webcam():
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/upload', methods=['POST'])
def upload():
    ap = argparse.ArgumentParser()
    ap.add_argument("-f", "--face", type=str,
                    default="face_detector",
                    help="path to face detector model directory")
    ap.add_argument("-m", "--model", type=str,
                    default="mask_detector.model",
                    help="path to trained face mask detector model")
    ap.add_argument("-c", "--confidence", type=float, default=0.5,
                    help="minimum probability to filter weak detections")
    args = vars(ap.parse_args())

    print("[INFO] loading face detector model...")
    prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
    weightsPath = os.path.sep.join([args["face"],
                                    "res10_300x300_ssd_iter_140000.caffemodel"])
    net = cv2.dnn.readNet(prototxtPath, weightsPath)


    print("[INFO] loading face mask detector model...")
    model = load_model(args["model"])

    file = request.files['image']

    image = read_image(file)

    orig = image.copy()
    (h, w) = image.shape[:2]

    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300),
                                 (104.0, 177.0, 123.0))

    # face detections
    print("[INFO] computing face detections...")
    net.setInput(blob)
    detections = net.forward()


    for i in range(0, detections.shape[2]):

        confidence = detections[0, 0, i, 2]

        if confidence > args["confidence"]:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            face = image[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)
            face = np.expand_dims(face, axis=0)


            (mask, withoutMask) = model.predict(face)[0]

            label = "Mask" if mask > withoutMask else "No Mask"
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

            label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

            cv2.putText(image, label, (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)

    # Prepare image for html
    to_send = prepare_image(image)

    return render_template('templates/index.html', image_to_show=to_send,
                           init=True)
if __name__ == '__main__':
    app.run(debug=True)