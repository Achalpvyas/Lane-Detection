import cv2
import numpy as np
import math


######################################################
#            Homography Estimation 
######################################################
def estimateHomography():
    # img = cv2.imread('./Data/data_1/data/0000000302.png')
  
    
    # p1 = np.array([[590,289],[312,479],[726,295],[815,435]],np.float32)
    # p2 = np.array([[0,0],[0,200],[0,200],[200,200]],np.float32)
    
    # cv2.circle(img,(445,291),5,(0,255,255),-1)
    # cv2.circle(img,(733,278),5,(0,0,255),-1)
    # cv2.circle(img,(56,407),5,(0,255,255),-1)
    # cv2.circle(img,(856,447),5,(0,0,255),-1)

    p1 = np.array([[445,291],[56,407],[733,278],[856,447]],np.float32)
    p2 = np.array([[0,0],[0,500],[500,0],[500,500]],np.float32)

    H = cv2.getPerspectiveTransform(p1,p2)
    # warpedImage =cv2.warpPerspective(img,H,(500,500))

    # cv2.imshow('image',img)
    # cv2.imshow('image',warpedImage)
    # cv2.waitKey(0)
    return H 




def preProcess(image):

    #Camera Matrix
    K = np.array([[  1.15422732e+03 , 0.00000000e+00   , 6.71627794e+02 ],
                  [  0.00000000e+00 ,  1.14818221e+03  , 3.86046312e+02 ],
                  [  0.00000000e+00 ,  0.00000000e+00  , 1.00000000e+00 ]])

    #Distortion Coefficients
    dist = np.array([ -2.42565104e-01 , -4.77893070e-02 , -1.31388084e-03 , -8.79107779e-05, 2.20573263e-02])

    destination = cv2.undistort(image, K, dist, None, K)

    gray = cv2.cvtColor(destination,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),0) 
    edges = cv2.Canny(blur,100,200)

    x,y = edges.shape

    # crop = edges[math.floor(x/2):x,0:y]
    edges[0:math.floor(x/2),0:y] = 0
    # print(x,y)
    # cv2.imshow('edges',edges)
    return edges


def detectLaneCandidates(pf,frame):
    # H = estimateHomography()
    # warpedImage = cv2.warpPerspective(image,H,(500,500))
    
    # cv2.imshow('warpedImage',warpedImage)
    # so = cv2.Sobel(pf,cv2.CV_64F,1,0,ksize=5) 
    # cv2.imshow('sobel',so)
    

   



######################################################
#               Process Frame
######################################################
def processFrame(frame):
    preprocessedFrame = preProcess(frame)
    detectLaneCandidates(preprocessedFrame,frame)
    cv2.imshow('sobel',frame)



######################################################
#               Reading Video 
#####################################################
cap = cv2.VideoCapture('./Data/data_2/challenge_video.mp4')

while(cap.isOpened()):
    ret, frame = cap.read()
    frame = cv2.resize(frame, None,fx=0.5, fy=0.5, interpolation = cv2.INTER_CUBIC)

    processFrame(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


if __name__ == '__main__':
    estimateHomography()
    
