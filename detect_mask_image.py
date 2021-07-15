# How to use
# python detect_mask_image.py --image examples/example_01.jpg

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import argparse
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

def mask_image():
	# initialization
	ap = argparse.ArgumentParser()
	ap.add_argument("-i", "--image", required=True,
		help="path to input image")
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
	net = cv2.dnn.readNet(prototxtPath, weightsPath)

	# load trained_model
	print("[INFO] loading face mask detector model...")
	model = load_model(args["model"])

	# load input image and preprocess
	image = cv2.imread(args["image"])
	orig = image.copy()
	(h, w) = image.shape[:2]

	# image to blob
	blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300),
		(104.0, 177.0, 123.0))

	# face detections
	print("[INFO] computing face detections...")
	net.setInput(blob)
	detections = net.forward()

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
			face = image[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)
			face = np.expand_dims(face, axis=0)

			# prediction mask/no mask
			(mask, withoutMask) = model.predict(face)[0]

			# Frontend Stuff
			label = "Mask" if mask > withoutMask else "No Mask"
			color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

			# detector = handDetector()
			# image = detector.findHand(image)
			# if str(type(detector.where(image))) == "<class 'tuple'>":
			# 	if endX > float(detector.where(image)[0]) * image.shape[1] > startX and endY > float(
			# 			detector.where(image)[1]) * image.shape[0] > startY:
			# 		label = "Had hand covered face, cannot detect"
			# 		color = (0, 0, 255)
			# 		prob = 0

			#Result & probability
			label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
			cv2.putText(image, label, (startX, startY - 10),
				cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
			cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)

	# show output image
	cv2.imshow("Output", rescaleimage(image))
	cv2.waitKey(0)

if __name__ == "__main__":
	mask_image()
