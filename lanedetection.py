import cv2
import numpy as np
import matplotlib.pyplot as plt
import math


######################################################
#            Homography Estimation 
######################################################
def estimateHomography():
    img = cv2.imread('./Data/data_1/data/0000000222.png')    

    x,y,_ = img.shape
    cropImg = img[math.floor(x/2):x,0:y]
    # p1 = np.array([[590,289],[312,479],[726,295],[815,435]],np.float32)
    # p2 = np.array([[0,0],[0,200],[0,200],[200,200]],np.float32)
    
    # p1 = np.array([[265,257],[460,249],[128,323],[543,273]],np.float32)
    # p2 = np.array([[0,0],[500,0],[0,500],[500,500]],np.float32)

    cv2.circle(cropImg,(539,67),5,(0,0,255),-1)
    cv2.circle(cropImg,(360,187),5,(0,255,0),-1)
    cv2.circle(cropImg,(745,70),5,(255,0,0),-1)
    cv2.circle(cropImg,(818,190),5,(0,255,255),-1)

    # p1 = np.array([[529,329],[340,454],[741,326],[818,453]],np.float32)
    # p2 = np.array([[0,0],[0,300],[300,0],[300,300]],np.float32)

    p1 = np.array([[539,67],[360,187],[745,70],[818,190]],np.float32)
    p2 = np.array([[75,75],[75,500],[300,75],[300,500]],np.float32)
    H = cv2.getPerspectiveTransform(p1,p2)
    # warpedImage =cv2.warpPerspective(cropImg,H,(400,500))

    # cv2.imshow('image',cropImg)
    # cv2.imshow('image2',warpedImage)

    return H 




def preProcess(frame):
    #ROI
    x,y,_ = frame.shape
    cropFrame = frame[math.floor(x/2):x,0:y]

    
    #Undistort image
    #--Camera Matrix
    K = np.array([[  1.15422732e+03 , 0.00000000e+00   , 6.71627794e+02 ],
                  [  0.00000000e+00 ,  1.14818221e+03  , 3.86046312e+02 ],
                  [  0.00000000e+00 ,  0.00000000e+00  , 1.00000000e+00 ]])

    #--Distortion Coefficients
    dist = np.array([ -2.42565104e-01 , -4.77893070e-02 , -1.31388084e-03 , -8.79107779e-05, 2.20573263e-02])
    undistortFrame = cv2.undistort(cropFrame, K, dist, None, K)

    #Detect lanes
    detectedLanes = detectLaneCandidates(undistortFrame)
    detectedLanesBGR = cv2.cvtColor(detectedLanes, cv2.COLOR_HLS2BGR)

    #Smoothing and Edge detection
    gray = cv2.cvtColor(detectedLanesBGR,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(1,1),0) 
    edges = cv2.Canny(blur,100,200)
    # edges = cv2.Sobel(blur,cv2.CV_64F,0,1,ksize=3) 
    
    #Dilate image
    kernel = np.ones((2,2),np.uint8)
    dilateFrame = cv2.dilate(edges,kernel,iterations=2)
    # erodeFrame = cv2.erode(dilateFrame,kernel,iterations=2)

    opening =cv2.morphologyEx(dilateFrame, cv2.MORPH_OPEN, kernel)

    # cv2.imshow('res',opening)
    return opening


def detectLaneCandidates(pf):
    hsv = cv2.cvtColor(pf,cv2.COLOR_BGR2HSV)

    # Range for yellow 
    lower_yellow = np.array([20,100,100])
    upper_yellow = np.array([30,255,255])
    mask1 = cv2.inRange(hsv, lower_yellow, upper_yellow)


    # Range for white 
    lower_white= np.array([0,0,255])
    upper_white= np.array([80,100,255])
    mask2 = cv2.inRange(hsv,lower_white,upper_white)

    mask = mask2 + mask1
    res = cv2.bitwise_and(pf,pf,mask = mask)

    # H = estimateHomography()
    # warpedImage = cv2.warpPerspective(res,H)
    # kernel = np.ones((2,2),np.uint8)
    # maskprop= cv2.dilate(mask,kernel,iterations=2)
    return res


def detectLanes(pf):
    H = estimateHomography()
    warpedFrame =cv2.warpPerspective(pf,H,(400,500))

    
    hist = np.sum(warpedFrame,axis = 0) 

    _,x = warpedFrame.shape
    stepsize = x/hist.shape[0]
    
    # show the plotting graph of an image 

    # plt.imshow(warpedFrame) 
    # cv2.imshow('process')

    titles = ['warpedFrame']
    images = [warpedFrame]
    for i in range(2):
        plt.subplot(1,2,i+1)
        if(i == 1):
            plt.plot(np.arange(0,x,stepsize),hist) 
        else:
            plt.imshow(images[i],'gray')
            plt.title(titles[i])
    plt.show()


######################################################
#               Process Frame
######################################################
def processFrame(frame):
    pf = preProcess(frame)
    detectLanes(pf)
    # detectLaneCandidates(preprocessedFrame,frame)
    # cv2.imshow('sobel',frame)
    # estimateHomography()



######################################################
#               Reading Video 
#####################################################
cap = cv2.VideoCapture('./video.avi')
plt.ion()

while(cap.isOpened()):
    ret, frame = cap.read()
    # frame = cv2.resize(frame, None,fx=0.9, fy=0.9, interpolation = cv2.INTER_CUBIC)

    processFrame(frame)
    cv2.imshow('frame',frame)
    plt.pause(0.001)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()



    
