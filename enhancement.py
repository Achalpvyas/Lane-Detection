import numpy as np
import cv2
import matplotlib.pyplot as plt


# _,thresh = cv2.threshold(warpedtag,220,255,cv2.THRESH_BINARY)

def preprocessing(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),0) 
    edges = cv2.Canny(blur,100,200)
    return edges 

def histogram(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    pixelNumber = gray.shape[0] * gray.shape[1]
    # cv2.imshow("gray image",gray)
    pixelDensity = np.zeros(256)

    # pixelDensity[gray] += 1
    for i in range(0, gray.shape[0]):
        for j in range(0, gray.shape[1]):
            pixelDensity[gray[i,j]] += 1
    # print(pixelDensity)
    # print('--------------')
    # print(pixelNumber)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #      cv2.destroyAllWindows()
    return pixelDensity, pixelNumber

def histogramEqualization(pixelDensity, pixelNumber):
    CDF = np.zeros(256)
    for i in range(0,len(pixelDensity)):
        for j in range(0, i+1):
            CDF[i] += pixelDensity[j] / pixelNumber
    # print('---------')
    # print(CDF)
    return CDF

def newImage(gray, CDF):
    for i in range(0, gray.shape[0]):
        for j in range(0, gray.shape[1]):
            gray[i,j] = np.round(CDF[gray[i,j]]*255)
    

# img = cv2.imread('Lena.png')
# gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# p,q = histogram(img)
# # print(p)
# plt.plot(np.arange(0,256,1),p)
# plt.show()
# cdf = histogramEqualization(p,q)
# plt.plot(np.arange(0,256,1),cdf)
# plt.show()
# newImage(gray,cdf)


######################################################
#              Reading Video 
#####################################################
cap = cv2.VideoCapture('NightDrive.mp4')

while(cap.isOpened()):
    ret, frame = cap.read()
    frame = cv2.resize(frame, None,fx=0.5, fy=0.5, interpolation = cv2.INTER_CUBIC)
    # p,q = histogram(frame)
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    # print(p)
    # plt.plot(np.arange(0,256,1),p)
    # plt.show()
    # cdf = histogramEqualization(p,q)
    # # plt.plot(np.arange(0,256,1),cdf)
    # # plt.show()
    # newImage(gray,cdf)
    # cv2.imshow("gray image",gray)
    equ = cv2.equalizeHist(gray)
    cv2.imshow("output",equ)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

