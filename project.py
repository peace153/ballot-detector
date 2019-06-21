from tkinter import *
import numpy as np
import cv2
from PIL import ImageTk, Image
from pandas import DataFrame
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import time
import threading

import sys
sys.path.insert(0, 'imagezmq/imagezmq')  # imagezmq.py is in ../imagezmq
import imagezmq

import math

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
            17: {'name': 'X17', 'score': 0 },
            18: {'name': 'X18', 'score': 0 },
            19: {'name': 'X19', 'score': 0 },
            20: {'name': 'X30', 'score': 0 },
            21: {'name': 'X21', 'score': 0 },
            22: {'name': 'X22', 'score': 0 },
            23: {'name': 'X23', 'score': 0 },
            24: {'name': 'X24', 'score': 0 },
            25: {'name': 'X25', 'score': 0 },
            26: {'name': 'X26', 'score': 0 },
            27: {'name': 'X27', 'score': 0 },
            28: {'name': 'X28', 'score': 0 },
            29: {'name': 'X29', 'score': 0 },
            30: {'name': 'X30', 'score': 0 },
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
resFrame.place(x=660,y=170)
lres = Label(resFrame)
lres.grid(row=0, column=0)

##Set Traffic Icon
# -----
iconFrame = Frame(GUI, width=100, height=100)
iconFrame.place(x=685,y=50)
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

image_hub = imagezmq.ImageHub()

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
    dim = (150,250)

    frame = cv2.imread("croped_img.png")

    frame =cv2.resize(frame,dim,interpolation = cv2.INTER_AREA)
    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    img = Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image=img)

    resFrame = ImageTk.PhotoImage(image=img)
    lres.resFrame = resFrame
    lres.configure(image=resFrame)

def countScore(party_no=99):
    partys[77]['score'] = partys[77]['score'] + 1
    partys[party_no]['score'] = partys[party_no]['score'] + 1

    frame = frScore
    rowid = party_no

    if rowid > midpoint:
        frame = frScore2
        temp = party_no
        if party_no == 88:
            temp = 32
        elif party_no == 99:
            setIcon(4)
            temp = 33
        rowid = temp-midpoint
    # update score of party_no
    c = Label(frame,text=partys[party_no]['score'], font=('Angsana New',16),foreground='red') 
    c.grid(row=rowid,column=0)

    # update score of total
    c = Label(frScore2,text=partys[77]['score'], font=('Angsana New',16),foreground='red')
    c.grid(row=31-midpoint,column=0)

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

template_bgr = cv2.imread('card_border_black_2.png')
template_gray = cv2.cvtColor( template_bgr, cv2.COLOR_BGR2GRAY )
h,w = template_gray.shape[:2]

detector = cv2.xfeatures2d.SIFT_create()
# detector = cv2.ORB_create()
template_kps,template_descs = detector.detectAndCompute( template_gray, None )
FLANN_INDEX_KDTREE = 0
FLANN_INDEX_LSH = 6
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
# index_params = dict(algorithm = FLANN_INDEX_LSH,
#                         table_number = 6,       # 12
#                         key_size = 12,          # 20
#                         multi_probe_level = 1)  # 2
search_params = dict(checks=50)
# Initiate the matcher
flann = cv2.FlannBasedMatcher(index_params,search_params)

def detect_card(cam_bgr):
    output = None
    found = False
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
            area = w * h
            print(x, y, w, h)
            print('area:', area)
            if area < 30000 or x < 0 or y < 0:
                print("image out of bound")
            else:
                found = True
                pts = np.array(dst_pts.reshape(4,2), dtype = "float32")
                warped = four_point_transform(hmcp, pts)
                output = cv2.resize(warped, (480, 640))
                cv2.imwrite('croped_img.png', output)
    return output, found

def dist(coor):
    return math.sqrt((coor[0][0] - coor[0][2])**2 + (coor[0][1] - coor[0][3])**2)

def angle(l):
    x1, y1, x2, y2 = l
    return math.degrees(math.atan((x1-x2)/(y1-y2)))

def detect_x(frame):
    global card_pass, card_number
    points = []
    try:
        hough_lineP = frame.copy()
        output_img = frame.copy()

        # sharpen image
        k = 7
        kernel = np.zeros( (k,k), np.float32 ) 
        kernel.itemset( (k//2,k//2) , 2 ) 
        kernel -= np.ones( (k,k) , np.float32 ) / (k*k) 

        sharpen_img = cv2.filter2D(frame, -1, kernel) 
        sharpen_img_gray = cv2.cvtColor(sharpen_img, cv2.COLOR_RGB2GRAY)

        # Canny edge detection
        thresh1 = 100
        thresh2 = 255
        canny = cv2.Canny (sharpen_img_gray, thresh1, thresh2)

        # Eilminate table lines
        k = 3
        hk = int(k/2)
        lineh = np.zeros((k,k), dtype = np.uint8)
        lineh[hk,...] = 1
        horizontal = cv2.morphologyEx(canny, cv2.MORPH_OPEN, lineh, iterations=2)
        mor_open = canny-horizontal

        linev = np.zeros((k,k),dtype = np.uint8)
        linev[...,hk] = 1 
        vertical = cv2.morphologyEx(mor_open, cv2.MORPH_OPEN, linev, iterations=2)
        mor_open_ver = mor_open-vertical

        # left column
        top_margin_l  = 145
        bottom_margin_l  = 600
        front_margin_l  = 192
        back_margin_l  = 232

        # right Column
        top_margin_r  = 145
        bottom_margin_r = 600
        front_margin_r = 412
        back_margin_r = 452

        list_lines = []
        column_coordinates = [(top_margin_l, bottom_margin_l, front_margin_l, back_margin_l),
                                (top_margin_r, bottom_margin_r, front_margin_r, back_margin_r)]
        y_found, yh_found, x_found, xw_found = (0, 0, 0, 0)
        for y, yh, x, xw in column_coordinates:
            crop_img = mor_open_ver[y:yh, x:xw].copy()

            # Preprocess Mark edge
            kernel = np.ones((3,3), np.uint8)
            img_dilation = cv2.dilate(crop_img, kernel, iterations=2) # max white
            img_erosion = cv2.erode(img_dilation, kernel, iterations=3) # black

            # Find stright line from edge detected
            linesP = cv2.HoughLinesP( img_erosion,                 # 1-channel 8-bit binary input image
                                    rho=1,                          # distance resolution of the accumulator in pixels
                                    theta=np.pi/180,                # angle resolution of the accumulator in radians
                                    threshold=10,                   # accumulator threshold, only lines with enough votes (>threshold) will be returned 
                                    lines=None,                     # output vector of lines #lines=np.array([])
                                    minLineLength = 10,             # minimum line length, line segments shorter than this will be rejected
                                    maxLineGap = 5)                 # maximum gap allowed between points on the same line to link them
            
            if linesP is not None:
                y_found, yh_found, x_found, xw_found = (y, yh, x, xw)
                for i in range(len(linesP)):
                    list_lines.append(linesP[i])
                
        coordinates = []
        if list_lines is not None:
            list_lines = sorted(list_lines, key = dist, reverse = True)
            # Find lines with angle difference more than 65 degree and collect line coordinate
            diff_ang = 50
            for i in range(len(list_lines)-1):
                l = list_lines[i+1][0]
                angleL = angle(l)
                sameInCoor = False
                for j in range(len(coordinates)):
                    c = coordinates[j]
                    angleC = angle(c)
                    # check = (0 < angleC <2 ) or (minAngle < angleC < maxAngle) or (-maxAngle < angleC < -minAngle)
                    if (abs(angleL - angleC) < diff_ang):
                        sameInCoor = True
                        break
                if not sameInCoor:
                    coordinates.append(l)     
            print(coordinates)

        coor_x = []
        number_x = 0
        x_cal1 = 0
        x_cal2 = 0
        if coordinates is not None:
            for i in range(len(coordinates)-1):
                c1 = coordinates[i] 
                x1, y1, x2, y2 = c1
                m = (y1 - y2) / (x1 - x2)
                b = y2 - m*x2

                for j in range(i,len(coordinates)):
                    c2 = coordinates[j]
                    x3, y3, x4, y4 = c2
                    
                    y_cal1 = m*x3 + b
                    y_cal2 = m*x4 + b

                    mj = (y3 - y4) / (x3 - x4)
                    bj = y3 - mj*x3
                    check = (m != 0) and (mj != 0) 
                    
                    if (m != 0) and (mj != 0):
                        x_cal1 = (y1 - bj) / mj
                        x_cal2 = (y2 - bj) / mj
                        
                    checkx1 = (y3 > y_cal1) & (y4 < y_cal2) #& ((x3 > x_cal1) or (x4 < x_cal2))
                    checkx2 = (y3 < y_cal1) & (y4 > y_cal2) #& ((x3 < x_cal1) or (x4 > x_cal2))
                        
                    if checkx1:
                        number_x += 1
                        coor_x.append(c1)
                        coor_x.append(c2)
                    elif checkx2:
                        number_x += 1
                        coor_x.append(c1)
                        coor_x.append(c2)

        if number_x == 1:
            for coor in coor_x:
                x1, y1, x2, y2 = tuple(coor)
                points.append((x1 + x_found, y1 + y_found))
                points.append((x2 + x_found, y2 + y_found))
                cv2.line(output_img[y_found:yh_found, x_found:xw_found], (x1, y1), (x2, y2), (0,255,0), 1, cv2.LINE_8)

            print("Card number: ", card_number)
            print("Found X :", number_x, "ea")
            print("Result : Card is good")
            print("Coordinate X mark:", points)
            card_number += 1
            card_pass += 1               
            print(sharpen_img_gray.shape)
            print("--" * 20)
        else:
            print("Card number: ", card_number)
            print("Found X :", number_x, "ea")
            print("Result : Card is not good")
            card_number += 1               
            print("--" * 20)
    except:
        print("Card number: ", card_number)
        print("Result***: ERROR FOUND")   
    print("Print OUT")
    # score = card_pass/card_number*100
    # print("% Passed :", score)
    # print("% Fail :", 100 - score)
    return points

def find_party_number(input_x):
    try:
        # If image size has changed, adjust only these 2 parameters.
        w = 480 # Card width
        h = 640 # Card height
        step_x = 228
        step_y = 26
        
        # Add variant 0.5%
        firstBox1_x = 188 * 0.95
        firstBox2_x = 234 * 1.05
        firstBox3_x = 188 * 0.95
        firstBox4_x = 234 + 1.05
        
        firstBox1_y = 136 - 0.95
        firstBox2_y = 136 - 0.95
        firstBox3_y = 162 + 1.05
        firstBox4_y = 162 + 1.05

        novote_box1_x = 416 * 0.95
        novote_box2_x = 460 * 1.05
        novote_box3_x = 416 * 0.95
        novote_box4_x = 460 * 1.05
        
        novote_box1_y = 550 * 0.95
        novote_box2_y = 550 * 0.95
        novote_box3_y = 582 * 1.05
        novote_box4_y = 582 * 1.05

        x = []
        y = []
        
        for i in range(len(input_x)):
            x.append(input_x[i][0])
            y.append(input_x[i][1])
        
        if(x[0] >= firstBox1_x and x[1] < firstBox2_x and x[2] >= firstBox3_x and x[3] < firstBox4_x and 
           y[0] >= firstBox1_y and y[1] >= firstBox2_y and y[2] < firstBox3_y and y[3] < firstBox4_y):
            return 1
        elif(x[0] >= firstBox1_x and x[1] < firstBox2_x and x[2] >= firstBox3_x and x[3] < firstBox4_x and 
             y[0] >= (firstBox1_y+(1*step_y)) and y[1] >= (firstBox2_y+(1*step_y)) and 
             y[2] < (firstBox3_y+(1*step_y)) and y[3] < (firstBox4_y+(1*step_y))):
            return 2
        elif(x[0] >= firstBox1_x and x[1] < firstBox2_x and x[2] >= firstBox3_x and x[3] < firstBox4_x and 
             y[0] >= (firstBox1_y+(2*step_y)) and y[1] >= (firstBox2_y+(2*step_y)) and 
             y[2] < (firstBox3_y+(2*step_y)) and y[3] < (firstBox4_y+(2*step_y))):
            return 3
        elif(x[0] >= firstBox1_x and x[1] < firstBox2_x and x[2] >= firstBox3_x and x[3] < firstBox4_x and 
             y[0] >= (firstBox1_y+(3*step_y)) and y[1] >= (firstBox2_y+(3*step_y)) and 
             y[2] < (firstBox3_y+(3*step_y)) and y[3] < (firstBox4_y+(3*step_y))):
            return 4
        elif(x[0] >= firstBox1_x and x[1] < firstBox2_x and x[2] >= firstBox3_x and x[3] < firstBox4_x and 
             y[0] >= (firstBox1_y+(4*step_y)) and y[1] >= (firstBox2_y+(4*step_y)) and 
             y[2] < (firstBox3_y+(4*step_y)) and y[3] < (firstBox4_y+(4*step_y))):
            return 5
        elif(x[0] >= firstBox1_x and x[1] < firstBox2_x and x[2] >= firstBox3_x and x[3] < firstBox4_x and 
             y[0] >= (firstBox1_y+(5*step_y)) and y[1] >= (firstBox2_y+(5*step_y)) and 
             y[2] < (firstBox3_y+(5*step_y)) and y[3] < (firstBox4_y+(5*step_y))):
            return 6
        elif(x[0] >= firstBox1_x and x[1] < firstBox2_x and x[2] >= firstBox3_x and x[3] < firstBox4_x and 
             y[0] >= (firstBox1_y+(6*step_y)) and y[1] >= (firstBox2_y+(6*step_y)) and 
             y[2] < (firstBox3_y+(6*step_y)) and y[3] < (firstBox4_y+(6*step_y))):
            return 7
        elif(x[0] >= firstBox1_x and x[1] < firstBox2_x and x[2] >= firstBox3_x and x[3] < firstBox4_x and 
             y[0] >= (firstBox1_y+(7*step_y)) and y[1] >= (firstBox2_y+(7*step_y)) and 
             y[2] < (firstBox3_y+(7*step_y)) and y[3] < (firstBox4_y+(7*step_y))):
            return 8
        elif(x[0] >= firstBox1_x and x[1] < firstBox2_x and x[2] >= firstBox3_x and x[3] < firstBox4_x and 
             y[0] >= (firstBox1_y+(8*step_y)) and y[1] >= (firstBox2_y+(8*step_y)) and 
             y[2] < (firstBox3_y+(8*step_y)) and y[3] < (firstBox4_y+(8*step_y))):
            return 9
        elif(x[0] >= firstBox1_x and x[1] < firstBox2_x and x[2] >= firstBox3_x and x[3] < firstBox4_x and 
             y[0] >= (firstBox1_y+(9*step_y)) and y[1] >= (firstBox2_y+(9*step_y)) and 
             y[2] < (firstBox3_y+(9*step_y)) and y[3] < (firstBox4_y+(9*step_y))):
            return 10
        elif(x[0] >= firstBox1_x and x[1] < firstBox2_x and x[2] >= firstBox3_x and x[3] < firstBox4_x and 
             y[0] >= (firstBox1_y+(10*step_y)) and y[1] >= (firstBox2_y+(10*step_y)) and 
             y[2] < (firstBox3_y+(10*step_y)) and y[3] < (firstBox4_y+(10*step_y))):
            return 11
        elif(x[0] >= firstBox1_x and x[1] < firstBox2_x and x[2] >= firstBox3_x and x[3] < firstBox4_x and 
             y[0] >= (firstBox1_y+(11*step_y)) and y[1] >= (firstBox2_y+(11*step_y)) and 
             y[2] < (firstBox3_y+(11*step_y)) and y[3] < (firstBox4_y+(11*step_y))):
            return 12
        elif(x[0] >= firstBox1_x and x[1] < firstBox2_x and x[2] >= firstBox3_x and x[3] < firstBox4_x and 
             y[0] >= (firstBox1_y+(12*step_y)) and y[1] >= (firstBox2_y+(12*step_y)) and 
             y[2] < (firstBox3_y+(12*step_y)) and y[3] < (firstBox4_y+(12*step_y))):
            return 13
        elif(x[0] >= firstBox1_x and x[1] < firstBox2_x and x[2] >= firstBox3_x and x[3] < firstBox4_x and 
             y[0] >= (firstBox1_y+(13*step_y)) and y[1] >= (firstBox2_y+(13*step_y)) and 
             y[2] < (firstBox3_y+(13*step_y)) and y[3] < (firstBox4_y+(13*step_y))):
            return 14
        elif(x[0] >= firstBox1_x and x[1] < firstBox2_x and x[2] >= firstBox3_x and x[3] < firstBox4_x and 
             y[0] >= (firstBox1_y+(14*step_y)) and y[1] >= (firstBox2_y+(14*step_y)) and 
             y[2] < (firstBox3_y+(14*step_y)) and y[3] < (firstBox4_y+(14*step_y))):
            return 15
        
        elif(x[0] >= (firstBox1_x+step_x) and x[1] < (firstBox2_x+step_x) and 
             x[2] >= (firstBox3_x+step_x) and x[3] < (firstBox4_x+step_x) and 
             y[0] >= firstBox1_y and y[1] >= firstBox2_y and y[2] < firstBox3_y and y[3] < firstBox4_y):
            return 16
        elif(x[0] >= (firstBox1_x+step_x) and x[1] < (firstBox2_x+step_x) and 
             x[2] >= (firstBox3_x+step_x) and x[3] < (firstBox4_x+step_x) and 
             y[0] >= (firstBox1_y+(1*step_y)) and y[1] >= (firstBox2_y+(1*step_y)) and 
             y[2] < (firstBox3_y+(1*step_y)) and y[3] < (firstBox4_y+(1*step_y))):
            return 17
        elif(x[0] >= (firstBox1_x+step_x) and x[1] < (firstBox2_x+step_x) and 
             x[2] >= (firstBox3_x+step_x) and x[3] < (firstBox4_x+step_x) and 
             y[0] >= (firstBox1_y+(2*step_y)) and y[1] >= (firstBox2_y+(2*step_y)) and 
             y[2] < (firstBox3_y+(2*step_y)) and y[3] < (firstBox4_y+(2*step_y))):
            return 18
        elif(x[0] >= (firstBox1_x+step_x) and x[1] < (firstBox2_x+step_x) and 
             x[2] >= (firstBox3_x+step_x) and x[3] < (firstBox4_x+step_x) and 
             y[0] >= (firstBox1_y+(3*step_y)) and y[1] >= (firstBox2_y+(3*step_y)) and 
             y[2] < (firstBox3_y+(3*step_y)) and y[3] < (firstBox4_y+(3*step_y))):
            return 19
        elif(x[0] >= (firstBox1_x+step_x) and x[1] < (firstBox2_x+step_x) and 
             x[2] >= (firstBox3_x+step_x) and x[3] < (firstBox4_x+step_x) and 
             y[0] >= (firstBox1_y+(4*step_y)) and y[1] >= (firstBox2_y+(4*step_y)) and 
             y[2] < (firstBox3_y+(4*step_y)) and y[3] < (firstBox4_y+(4*step_y))):
            return 20
        elif(x[0] >= (firstBox1_x+step_x) and x[1] < (firstBox2_x+step_x) and 
             x[2] >= (firstBox3_x+step_x) and x[3] < (firstBox4_x+step_x) and 
             y[0] >= (firstBox1_y+(5*step_y)) and y[1] >= (firstBox2_y+(5*step_y)) and 
             y[2] < (firstBox3_y+(5*step_y)) and y[3] < (firstBox4_y+(5*step_y))):
            return 21
        elif(x[0] >= (firstBox1_x+step_x) and x[1] < (firstBox2_x+step_x) and 
             x[2] >= (firstBox3_x+step_x) and x[3] < (firstBox4_x+step_x) and 
             y[0] >= (firstBox1_y+(6*step_y)) and y[1] >= (firstBox2_y+(6*step_y)) and 
             y[2] < (firstBox3_y+(6*step_y)) and y[3] < (firstBox4_y+(6*step_y))):
            return 22
        elif(x[0] >= (firstBox1_x+step_x) and x[1] < (firstBox2_x+step_x) and 
             x[2] >= (firstBox3_x+step_x) and x[3] < (firstBox4_x+step_x) and 
             y[0] >= (firstBox1_y+(7*step_y)) and y[1] >= (firstBox2_y+(7*step_y)) and 
             y[2] < (firstBox3_y+(7*step_y)) and y[3] < (firstBox4_y+(7*step_y))):
            return 23
        elif(x[0] >= (firstBox1_x+step_x) and x[1] < (firstBox2_x+step_x) and 
             x[2] >= (firstBox3_x+step_x) and x[3] < (firstBox4_x+step_x) and 
             y[0] >= (firstBox1_y+(8*step_y)) and y[1] >= (firstBox2_y+(8*step_y)) and 
             y[2] < (firstBox3_y+(8*step_y)) and y[3] < (firstBox4_y+(8*step_y))):
            return 24
        elif(x[0] >= (firstBox1_x+step_x) and x[1] < (firstBox2_x+step_x) and 
             x[2] >= (firstBox3_x+step_x) and x[3] < (firstBox4_x+step_x) and 
             y[0] >= (firstBox1_y+(9*step_y)) and y[1] >= (firstBox2_y+(9*step_y)) and 
             y[2] < (firstBox3_y+(9*step_y)) and y[3] < (firstBox4_y+(9*step_y))):
            return 25
        elif(x[0] >= (firstBox1_x+step_x) and x[1] < (firstBox2_x+step_x) and 
             x[2] >= (firstBox3_x+step_x) and x[3] < (firstBox4_x+step_x) and 
             y[0] >= (firstBox1_y+(10*step_y)) and y[1] >= (firstBox2_y+(10*step_y)) and 
             y[2] < (firstBox3_y+(10*step_y)) and y[3] < (firstBox4_y+(10*step_y))):
            return 26
        elif(x[0] >= (firstBox1_x+step_x) and x[1] < (firstBox2_x+step_x) and 
             x[2] >= (firstBox3_x+step_x) and x[3] < (firstBox4_x+step_x) and 
             y[0] >= (firstBox1_y+(11*step_y)) and y[1] >= (firstBox2_y+(11*step_y)) and 
             y[2] < (firstBox3_y+(11*step_y)) and y[3] < (firstBox4_y+(11*step_y))):
            return 27
        elif(x[0] >= (firstBox1_x+step_x) and x[1] < (firstBox2_x+step_x) and 
             x[2] >= (firstBox3_x+step_x) and x[3] < (firstBox4_x+step_x) and 
             y[0] >= (firstBox1_y+(12*step_y)) and y[1] >= (firstBox2_y+(12*step_y)) and 
             y[2] < (firstBox3_y+(12*step_y)) and y[3] < (firstBox4_y+(12*step_y))):
            return 28
        elif(x[0] >= (firstBox1_x+step_x) and x[1] < (firstBox2_x+step_x) and 
             x[2] >= (firstBox3_x+step_x) and x[3] < (firstBox4_x+step_x) and 
             y[0] >= (firstBox1_y+(13*step_y)) and y[1] >= (firstBox2_y+(13*step_y)) and 
             y[2] < (firstBox3_y+(13*step_y)) and y[3] < (firstBox4_y+(13*step_y))):
            return 29
        elif(x[0] >= (firstBox1_x+step_x) and x[1] < (firstBox2_x+step_x) and 
             x[2] >= (firstBox3_x+step_x) and x[3] < (firstBox4_x+step_x) and 
             y[0] >= (firstBox1_y+(14*step_y)) and y[1] >= (firstBox2_y+(14*step_y)) and 
             y[2] < (firstBox3_y+(14*step_y)) and y[3] < (firstBox4_y+(14*step_y))):
            return 30
        elif(x[0] >= novote_box1_x and x[1] < novote_box2_x and 
             x[2] >= novote_box3_x and x[3] < novote_box4_x and 
             y[0] >= novote_box1_y and y[1] >= novote_box2_y and 
             y[2] < novote_box3_y and y[3] < novote_box4_y):
             return 88
        else:
            return 99 # Bad card
    except:
         print("Error in findPartyNumber()")
    return 99 # Bad card

state = 0

must_detect = True
count = 0
card_pass = 0
card_number = 0
wait_crop = 0
def show_frame():
    global state, must_detect, count, card_pass, card_number, wait_crop
    _, frame = image_hub.recv_image()
    image_hub.send_reply(b'OK')

    res = None
    ok = True
    if must_detect:
        res, ok = detect_card(frame)
    # frame = cv2.flip(frame, 1)
    resized_frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
    cv2image = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGBA)
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
            wait_crop += 1
            setIcon(2)
            print(res)
            print("##############################################################")
            if wait_crop == 5:
                must_detect = False
                showRes()
                points = detect_x(res)
                if points:
                    party_number = find_party_number(points)
                    print("##############################################################", points)
                    print("##############################################################", party_number)
                    countScore(party_number)
                else:
                    countScore()
                state = 1
                print(partys)
                print("off")
        else:
            wait_crop = 0
            setIcon(1)
            print("on")
    elif state == 1:
        wait_crop = 0
        if not ok:
            # setIcon(2)
            state = 0
            must_detect = True
            count=0
            print("on")
        else:
            setIcon(2)
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