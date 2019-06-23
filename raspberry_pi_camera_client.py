import sys
sys.path.insert(0, 'imagezmq/imagezmq')  # imagezmq.py is in ../imagezmq

import socket
import time
import cv2
from imutils.video import VideoStream
import imagezmq

sender = imagezmq.ImageSender(connect_to='tcp://192.168.137.1:5555')

rpi_name = socket.gethostname()
picam = VideoStream(usePiCamera=True, resolution=(1280, 720)).start()
time.sleep(2.0)  # allow camera sensor to warm up
while True:
    image = picam.read()
    print(image.shape)
    sender.send_image(rpi_name, image)
