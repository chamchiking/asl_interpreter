# Asl(American Sign Language video Interpreter)
This repository contains the following contents

- Sample program
- Hand sign recognition model(TFLite)
- Learning data for hand sign recognition and notebook for learning

# Requirements
- mediapipe 0.8.1
- OpenCV 3.4.2 or Later
- Tensorflow 2.3.0 or Later
 if-nightly 2.5.0 or Later (Only when createing a TFLite for an LSTM model)
- scikit-learn 0.23.2 or Later (Only if you want to display the confusion matrix)
- matplotlib3.3.2 or Later (Only if you want to display the confusion matrix)

# Demo
Here's how to run the demo using your webcam.

`python app.py`

The following options can be specified when running the demo.
- --device
  Specifying the camera device number (Defualt : 0)
- --width
  Width at the time of camera capture (Default : 960)
- --height
  Height at the time of camera capture (Default : 540)
- -- use_static_image_mode
  Whether to use static_image_mode option for MediaPipe inference (Default : Unspecified)
- --min_detection_confidence
  Detection confidence threshold (Default : 0.5)
- --min_tracking_confidence
  Tracking confidence threshold (Default : 0.5)

# Directory

## app.py
This is a sample program for infernce.

## extract_hand_csv.ipynb
for extracting csv files from "asl_images_in" folder
writing into /model/keypoint_classifier/keypoint.csv

everytime you run the extract_hand_csv keypoint.csv, it will append landmarks of hand images in the asl_imaes_in folder

## python app.py
python app.py --device 0
will open webcam with number 0 and start interpreting