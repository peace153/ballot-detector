import cv2
import numpy as np

import os
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp"
#vcap = cv2.VideoCapture("rtsp://192.168.1.2:5554/camera", cv2.CAP_FFMPEG)
vcap = cv2.VideoCapture("rtsp://192.168.1.124:5554/camera", cv2.CAP_FFMPEG)
while(1):
    ret, frame = vcap.read()
    if ret == False:
        print("Frame is empty")
        break
    else:
        cv2.imshow('VIDEO', frame)
        cv2.waitKey(1)

#REF : https://medium.com/beesightsoft/opencv-python-connect-to-android-camera-via-rstp-9eb78e2903d5