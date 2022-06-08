import os
import random
import glob
from os import getcwd
import cv2
import numpy as np
     
current_dir = getcwd()+'/'

#FIXME
in_filename = ['classes/learn_asl.mp4']
out_filename = "sample"
out_dirname = "images"

cnt = 10000
for vid in in_filename:
	print("Reading "+vid)
	cap = cv2.VideoCapture(vid)
	if(cnt*1000==0):
		print("Reading... %d\n",cnt)

	while(True):
		ret, frame = cap.read()
		print(ret)
		if ret:
			cv2.imwrite(out_dirname+"/"+out_filename+"_"+str(cnt)+".jpg",frame)
			cnt += 1
		else:
			print("File grab failed")
			break

	cap.release()
