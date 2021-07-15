# Face_Mask_Detector

## INTRO ##
**Face mask detector using OpenCV, Mediapipe, Tensorflow and Flask**
- This is my project in Year 1 during my time at university, while COVID-19 is spreading. This detector is used to
  - Detect face masks in images
  - Detect face masks in video
  - Detect face masks in real-time video streams
- Solution:
  - Detecting faces in images/video
  - Extracting each individual face
  - Applying our face mask classifier
  
 ## TECHSTACK/FRAMEWORK USED ##
- OpenCV
- Mediapipe
- Tensorflow
- Keras
- Flask
- Caffe-based face detector
- MobileNetV2

## INSTALLATION AND RUNNING ##
1. Clone the repo
2. Use the package manager pip to install package

    `pip install -r requirement.txt`
 
3. Open source code and read how to run

## RESULT ##
<img src="Readme_images/Evaluating Network.png">

<img src="Readme_images/plot.png">

## FEATURES ##
- Using OpenCV is detect blobs
- Using Mediapipe to detect hand, a problem of covering face by hand, making a lot of noise to the prediction is addressed
- Build web demo using Flask

## WORKING ON ##
- Dealing with many people in a frame

## REFERENCE ##
- [Pyimagesearch](https://www.pyimagesearch.com/2020/05/04/covid-19-face-mask-detector-with-opencv-keras-tensorflow-and-deep-learning/)
- [Mobilenet model](https://phamdinhkhanh.github.io/2020/09/19/MobileNet.html#6-t%C3%A0i-li%E1%BB%87u)
- [MobileNetV2: Inverted Residuals and Linear Bottlenecks - Mark Sandler, Andrew Howard](https://arxiv.org/abs/1801.04381)
- [Github](https://github.com/chandrikadeb7/Face-Mask-Detection)
- [Hand Tracking 30 FPS using CPU | OpenCV Python (2021) | Computer Vision](https://www.youtube.com/watch?v=NZde8Xt78Iw)
