import numpy as np
import cv2
import matplotlib.pyplot as plt


_,thresh = cv2.threshold(warpedtag,220,255,cv2.THRESH_BINARY)

def preprocessing(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),0) 
    edges = cv2.Canny(blur,100,200)
    return edges 


######################################################
#              Reading Video 
#####################################################
cap = cv2.VideoCapture('./Night_Drive.mp4')

while(cap.isOpened()):
    ret, frame = cap.read()
    frame = cv2.resize(frame, None,fx=0.5, fy=0.5, interpolation = cv2.INTER_CUBIC)

    processFrame(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

