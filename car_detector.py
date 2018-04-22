import cv2
import time
import numpy as np 

car_classifier = cv2.CascadeClassifier('cars.xml')

# Initialize video capture for video file
#cap = cv2.VideoCapture('videoplayback.mp4')
cap = cv2.VideoCapture('cars.mp4')

# Loop once video is successfully loaded
while True:
	ret, frame = cap.read()
	
	# This is necessary so that we don't get errors when the video is completed
	if ret == False:
		break

	#gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
	# Reducing the resolution of the video by half
	frame = cv2.resize(frame,None,fx=0.5,fy=0.5,interpolation=cv2.INTER_LINEAR)

	gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
	
	# Pass frame to our body classifier
	cars = car_classifier.detectMultiScale(gray,1.1,1)

	# Extract bounding boxes for any body identified
	for (x,y,w,h) in cars:
		cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
		cv2.imshow('Cars',frame)
	
	if cv2.waitKey(1) == 13:
		break

cap.release()
cv2.destroyAllWindows()