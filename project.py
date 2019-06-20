from tkinter import *
import numpy as np
import cv2
from PIL import ImageTk, Image
from pandas import DataFrame
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import time
import threading

partys = {  1: {'name': 'X01', 'score': 0 },
            2: {'name': 'X02', 'score': 0 },
            3: {'name': 'X03', 'score': 0 },
            4: {'name': 'X04', 'score': 0 },
            5: {'name': 'X05', 'score': 0 },
            6: {'name': 'X06', 'score': 0 },
            7: {'name': 'X07', 'score': 0 },
            8: {'name': 'X08', 'score': 0 },     
            9: {'name': 'X09', 'score': 0 },
            10: {'name': 'X10', 'score': 0 },
            11: {'name': 'X11', 'score': 0 },
            12: {'name': 'X12', 'score': 0 },
            13: {'name': 'X13', 'score': 0 },
            14: {'name': 'X14', 'score': 0 },
            15: {'name': 'X15', 'score': 0 },
            16: {'name': 'X16', 'score': 0 },
            17: {'name': 'X17', 'score': 1 },
            18: {'name': 'X18', 'score': 1 },
            19: {'name': 'X19', 'score': 1 },
            20: {'name': 'X30', 'score': 1 },
            21: {'name': 'X21', 'score': 1 },
            22: {'name': 'X22', 'score': 1 },
            23: {'name': 'X23', 'score': 1 },
            24: {'name': 'X24', 'score': 1 },
            25: {'name': 'X25', 'score': 1 },
            26: {'name': 'X26', 'score': 1 },
            27: {'name': 'X27', 'score': 1 },
            28: {'name': 'X28', 'score': 1 },
            29: {'name': 'X29', 'score': 1 },
            30: {'name': 'X30', 'score': 1 },
            77: {'name': 'Total', 'score': 0 },
            88: {'name': 'Novote', 'score': 0 },
            99: {'name': 'Reject', 'score': 0 }}

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

resFrame = Frame(GUI, width=400, height=200)
resFrame.place(x=680,y=120)
lres = Label(resFrame)
lres.grid(row=0, column=0)

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
frName = Frame(GUI,width=70)
frName.place(x=830,y=50)

frName2 = Frame(GUI,width=70)
frName2.place(x=1055,y=50)

##--Set Frame Party Score
frScore = Frame(GUI,width=100)
frScore.place(x=900,y=50)

frScore2 = Frame(GUI,width=100)
frScore2.place(x=1125,y=50)

##--Set live Camera
dim = (640, 480)
#Capture video frames
lmain = Label(imageFrame)
lmain.grid(row=0, column=0)

cap = cv2.VideoCapture(0)

# import os
# os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp"
# # #vcap = cv2.VideoCapture("rtsp://192.168.1.2:5554/camera", cv2.CAP_FFMPEG)
# cap = cv2.VideoCapture("rtsp://10.184.111.218:5554/camera", cv2.CAP_FFMPEG)

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

def showRes():
    global resFrame
    dim = (200,400)
    # pic = Image.open("croped_img.png")

    frame = cv2.imread("croped_img.png")

    frame =cv2.resize(frame,dim,interpolation = cv2.INTER_AREA)
    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    img = Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image=img)

    resFrame = ImageTk.PhotoImage(image=img)
    lres.resFrame = resFrame
    lres.configure(image=resFrame)

def countScore(party_no):
    partys[77]['score'] = partys[77]['score'] + 1
    partys[party_no]['score'] = partys[party_no]['score'] + 1

    frame = frScore
    rowid = party_no

    if rowid > midpoint:
        frame = frScore2
        rowid = party_no-midpoint

    c = Label(frame,text=partys[party_no]['score'], font=('Angsana New',16),foreground='red')
    if party_no == 88:
        party_no = 32
    elif party_no == 99:
        party_no = 33
    c.grid(row=rowid,column=0)
    c = Label(frScore2,text=partys[77]['score'], font=('Angsana New',16),foreground='red')
    c.grid(row=31-midpoint,column=0)

tmp = 0
def order_points(pts):
	rect = np.zeros((4, 2), dtype = "float32")
	s = pts.sum(axis = 1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]
	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]
	return rect


def four_point_transform(image, pts):
	rect = order_points(pts)
	(tl, tr, br, bl) = rect
	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	maxWidth = max(int(widthA), int(widthB))
	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxHeight = max(int(heightA), int(heightB))
	dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32")
	M = cv2.getPerspectiveTransform(rect, dst)
	warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
	return warped

template_bgr = cv2.imread('card_border_black.jpg')
template_gray = cv2.cvtColor( template_bgr, cv2.COLOR_BGR2GRAY )
h,w = template_gray.shape[:2]

# detector = cv2.xfeatures2d.SIFT_create()
detector = cv2.ORB_create()
template_kps,template_descs = detector.detectAndCompute( template_gray, None )
FLANN_INDEX_KDTREE = 0
FLANN_INDEX_LSH = 6
# index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
index_params = dict(algorithm = FLANN_INDEX_LSH,
                        table_number = 6,       # 12
                        key_size = 12,          # 20
                        multi_probe_level = 1)  # 2
search_params = dict(checks=50)
# Initiate the matcher
flann = cv2.FlannBasedMatcher(index_params,search_params)

def detect_card(cam_bgr):
    cam_gray = cv2.cvtColor( cam_bgr, cv2.COLOR_BGR2GRAY )
    # Find sets of keypoints and descriptors
    (cam_kps,cam_descs) = detector.detectAndCompute( cam_gray, None )
    if cam_descs is None:
        return None,False
    print('Input image: keypoints = ' + str( len(cam_kps) ) + ' , descriptors = ' + str(cam_descs.shape) )

    # Find the matches for each descriptor in 'template_descs'
    matches = flann.knnMatch( template_descs,   # query set of descriptors
                              cam_descs ,       # train set of descriptors
                              k=2 )             # only find two best matches for each query descriptor

    goodMatches = []
    for i, mn in enumerate(matches):
        if ( len(mn) == 2 ):                # prevent the case when only one match is found
            m = mn[0]  ;   n = mn[1]        # 'm' is the best match, 'n' is the second-best match
            if m.distance < 0.6 * n.distance: 
            # if m.distance < 0.6 * n.distance:   
                goodMatches.append( m )

    MIN_MATCH_COUNT = 10
    homoResult = cam_bgr.copy()
    print(len(goodMatches))
    if len ( goodMatches ) < MIN_MATCH_COUNT:
        print ("Not enough matches found for computing the homography.")
        return None,False
    else:        
        template_pts = np.float32([ template_kps[m.queryIdx].pt for m in goodMatches ]).reshape(-1,1,2)
        cam_pts = np.float32([ cam_kps[m.trainIdx].pt for m in goodMatches ]).reshape(-1,1,2)

        # Compute the 3x3 homography matrix for converting points in model image to points in camera image
        H_m2c, mask = cv2.findHomography(template_pts, cam_pts, cv2.RANSAC,5.0)
        matchesMask = mask.ravel().tolist()

        print(H_m2c)
        if H_m2c is None:
            print ("Not enough matches found for computing the homography.")
            return None,False
        else:
            # Find four corners of the template image in camera coordinate
            h,w = template_gray.shape[:2]
            src_pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
            dst_pts = cv2.perspectiveTransform(src_pts, H_m2c)

            hmcp = homoResult.copy()
            homoResult = cv2.polylines( homoResult,             # image  
                                        [np.int32(dst_pts)],    # array of polygonal curves
                                        True,                   # draw polylines as closed polygons
                                        (0,255,0),             # line color
                                        2, cv2.LINE_AA )         # line thickness , line's type
            rect = cv2.boundingRect(dst_pts)
            x, y, w, h = rect
            croped = homoResult[y:y+h, x:x+w].copy()
            area = w * h
            print(x, y, w, h)
            print('area:', area)
            if area < 3000 or x < 0 or y < 0:
                print("image out of bound")
                return None,False
            else:
                pts = np.array(dst_pts.reshape(4,2), dtype = "float32")
                warped = four_point_transform(hmcp, pts)
                # cv2.imshow('warped', warped)
                cv2.imwrite('croped_img.png', warped)
    return warped,True


def detect_x():
    global tmp
    res = tmp%12
    if res == 0:
        res = 88
    elif res == 11:
        res == 99
    return res

state = 0

must_detect = True
count = 0
def show_frame():
    global state,must_detect,count
    _, frame = cap.read()
    ok = True
    if must_detect:
        res,ok = detect_card(frame)
    # frame = cv2.flip(frame, 1)
    frame =cv2.resize(frame,dim,interpolation = cv2.INTER_AREA)
    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    img = Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image=img)
    lmain.imgtk = imgtk
    lmain.configure(image=imgtk)
    lmain.after(40, show_frame) 

    # Add Traffic Icon
    # Set Default icon_num :1
    icon_num = 1
    setIcon(icon_num)

    # # Display Score
    if state == 0:
        
        print(ok)
       
        if ok:
            print(res)
            print("##############################################################")
            must_detect = False
            showRes()
            party = detect_x()
            countScore(party)
            state = 1
            print(partys)
            print("off")
        else:
            print("on")
    elif state == 1:
        if not ok:
            state = 0
            must_detect = True
            count=0
            print("on")
        else:
            count = count+1
            if count>50:
                must_detect = True
            print("off")

##-- Set Barchart 
    # barChart()
rows = 0
midpoint = int(len(partys)/2)

for party_id,score in partys.items():
    rows = rows + 1
    nFrame = frName
    sFrame = frScore
    target = rows
    if rows > midpoint:
        nFrame = frName2
        sFrame = frScore2
        target = rows - midpoint
    b = Label(nFrame,text=score['name'], font=('Angsana New',16),foreground='blue')
    b.grid(row=target,column=0)
    c = Label(sFrame,text=score['score'], font=('Angsana New',16),foreground='red')
    c.grid(row=target,column=0)

##-- Display
show_frame()
print("show_frame")
GUI.mainloop()