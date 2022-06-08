# Asl(American Sign Language video Interpreter)

## extract_hand_csv.ipynb
for extracting csv files from "asl_images_in" folder
writing into /model/keypoint_classifier/keypoint.csv

everytime you run the extract_hand_csv keypoint.csv, it will append landmarks of hand images in the asl_imaes_in folder

## python app.py
python app.py --device 0
will open webcam with number 0 and start interpreting