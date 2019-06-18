from tkinter import *
import numpy as np
import cv2
from PIL import ImageTk, Image
from pandas import DataFrame
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import time
import threading

partys = { 1: {'name': 'XX1', 'score': 0 },
           2: {'name': 'XX2', 'score': 0 },
           3: {'name': 'XX3', 'score': 0 },
           4: {'name': 'XX4', 'score': 0 },
           5: {'name': 'XX5', 'score': 0 },
           6: {'name': 'XX6', 'score': 0 },
           7: {'name': 'XX7', 'score': 0 },
           8: {'name': 'XX8', 'score': 0 },
           9: {'name': 'XX9', 'score': 0 },
          10: {'name': 'XX10', 'score': 0 },
          77: {'name': 'Total', 'score': 0 },#Total
          88: {'name': 'Novote', 'score': 0 },#No vote
          99: {'name': 'Reject', 'score': 0 }}#Reject

####---- GUI

GUI = Tk()
GUI.title('CountScore')
GUI.geometry('1280x720+50+50')


SetLabel = Label(GUI,text='ผลการนับคะแนนบัตรเลือกตั้ง',font=('Angsana New',20))
SetLabel.place(x=830,y=15)

#Set up Image
# -----
imageFrame = Frame(GUI, width=640, height=480)
imageFrame.place(x=10,y=10)

##Set Traffic Icon
# -----
iconFrame = Frame(GUI, width=100, height=100)
iconFrame.place(x=680,y=50)
licon_TL = Label(iconFrame)
licon_TL.grid(row=0,column=0)


icon01 = Image.open("TL.jpg")
icon02 = Image.open("GL.jpg")
icon03 = Image.open("YL.jpg")
icon04 = Image.open("RL.jpg")

##--Set Frame Party name
frName = Frame(GUI)
frName.place(x=830,y=50)

##--Set Frame Party Score
frScore = Frame(GUI)
frScore.place(x=900,y=50)

##--Set Frame Graph
frChart = Frame(GUI)
frChart.place(x=10,y=500)

##--Set live Camera
dim = (640, 480)
#Capture video frames
lmain = Label(imageFrame)
lmain.grid(row=0, column=0)

# cap = cv2.VideoCapture(0)

import os
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp"
# #vcap = cv2.VideoCapture("rtsp://192.168.1.2:5554/camera", cv2.CAP_FFMPEG)
cap = cv2.VideoCapture("rtsp://10.184.111.218:5554/camera", cv2.CAP_FFMPEG)

def setIcon(icon_no = 1):
    if icon_no == 2:#Green
        iconName = icon02
    elif icon_no == 3:#Yellow
        iconName = icon03
    elif icon_no == 4:#Red
        iconName = icon04
    else:#Gray
        iconName = icon01
    
    TL_load = ImageTk.PhotoImage(image=iconName)
    licon_TL.TL_load = TL_load
    licon_TL.configure(image=TL_load)


def countScore(party_no):
    partys[77]['score'] = partys[77]['score'] + 1
    partys[party_no]['score'] = partys[party_no]['score'] + 1
    c = Label(frScore,text=partys[party_no]['score'], font=('Angsana New',16),foreground='red')
    rowid = party_no
    if party_no == 88:
        party_no = 12
    elif party_no == 99:
        party_no = 13
    c.grid(row=party_no,column=0)
    c = Label(frScore,text=partys[77]['score'], font=('Angsana New',16),foreground='red')
    c.grid(row=11,column=0)

tmp = 0
def detect_card():
    global tmp
    if tmp > 100:
        tmp = 0
    tmp = tmp+1
    return (tmp%15) == 0

def detect_x():
    global tmp
    res = tmp%12
    if res == 0:
        res = 88
    elif res == 11:
        res == 99
    return res

state = 0

def show_frame():
    global state
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)
    frame =cv2.resize(frame,dim,interpolation = cv2.INTER_AREA)
    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    img = Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image=img)
    lmain.imgtk = imgtk
    lmain.configure(image=imgtk)
    lmain.after(40, show_frame) 

    # Add Traffic Icon
    # Set Default icon_num :1
    icon_num = 2
    setIcon(icon_num)

    # # Display Score
    if state == 0:
        if detect_card():
            party = detect_x()
            countScore(party)
            state = 1
            print(partys)
            print("off")
        else:
            print("on")
    elif state == 1:
        if not detect_card():
            state = 0
            print("on")
        else:
            print("off")

##-- Set Barchart 
    # barChart()
rows = 0
for party_id,score in partys.items():
    rows = rows + 1
    b = Label(frName,text=score['name'], font=('Angsana New',16),foreground='blue')
    b.grid(row=rows,column=0)
    c = Label(frScore,text=score['score'], font=('Angsana New',16),foreground='red')
    c.grid(row=rows,column=0)

##-- Display
show_frame()
print("show_frame")
##-- Set Barchart 
GUI.mainloop()