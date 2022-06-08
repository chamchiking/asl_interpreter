#!/usr/bin/env python
# -*- coding: utf-8 -*-
import csv
import copy
import argparse
import itertools
import glob
from os import getcwd

import cv2 as cv
import mediapipe as mp

from model import KeyPointClassifier

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--in_file", type=str, default='')
    parser.add_argument('--use_static_image_mode', action='store_true')
    parser.add_argument("--min_detection_confidence",
                        help='min_detection_confidence',
                        type=float,
                        default=0.7)
    parser.add_argument("--min_tracking_confidence",
                        help='min_tracking_confidence',
                        type=int,
                        default=0.5)

    args = parser.parse_args()

    return args



def main():
    # Argument parsing #################################################################
    args = get_args()

    in_filename = ["asl_learn.mp4"]
    out_name="sample"
    out_dirname="frames"

    current_dir = getcwd() + '/'
    cnt = 1000
    for vid in in_filename:
        print("Reading" + vid)
        cap = cv.VideoCapture(vid)
        if(cnt*1000 ==0):
            print("Reading...%\n", cnt)
            
        while(True):
            ret, frame = cap.read()
            print(ret)
            if ret:
                cv.imwrite(out_dirname +"/" + out_name + "_" + str(cnt) +".jpg", frame)
                cnt +=1
            else:
                print("File grab failed")
                break

        cap.release()


    out_filename = "asl_result.txt"

    use_static_image_mode = args.use_static_image_mode
    min_detection_confidence = args.min_detection_confidence
    min_tracking_confidence = args.min_tracking_confidence


    # Model load #############################################################
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=use_static_image_mode,
        max_num_hands=1,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    keypoint_classifier = KeyPointClassifier()

    with open('model/keypoint_classifier/keypoint_classifier_label.csv',
              encoding='utf-8-sig') as f:
        keypoint_classifier_labels = csv.reader(f)
        keypoint_classifier_labels = [
            row[0] for row in keypoint_classifier_labels
        ]
    #  ############################################################################
    # 파일을 쭉 읽으면서 인식하는 부분
    f = open(out_filename, "a")
    # cap = cv.VideoCapture(in_filename)
    IMAGE_FILES = []
    for filename in glob.glob('frames/*.jpg'):
        IMAGE_FILES.append(filename)
    detected_list = ""
    for idx, file in enumerate(IMAGE_FILES):
        image = cv.flip(cv.imread(file), 1)
        results = hands.process(cv.cvtColor(image, cv.COLOR_BGR2RGB))
        
        # #######################################################################
        if not results.multi_hand_landmarks:
            continue
    
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                            results.multi_handedness):
            # Landmark calculation
            landmark_list = calc_landmark_list(image, hand_landmarks)
            # Conversion to relative coordinates / normalized coordinates
            pre_processed_landmark_list = pre_process_landmark(
                landmark_list)
            # Hand sign classification
            hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
            detected_asl = keypoint_classifier_labels[hand_sign_id]
            # print(detected_asl)
            detected_list += detected_asl
            # f.write(detected_asl)
    f.write(detected_list)                
    f.close()



def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    # Keypoint
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        # landmark_z = landmark.z

        landmark_point.append([landmark_x, landmark_y])

    return landmark_point

def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    # Convert to a one-dimensional list
    temp_landmark_list = list(
        itertools.chain.from_iterable(temp_landmark_list))

    # Normalization
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list


if __name__ == '__main__':
    main()