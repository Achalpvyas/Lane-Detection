import numpy as np
import cv2
import matplotlib.pyplot as plt

# Applying gamma correction
def gammaCorrection(frame, gamma = 1.0):
    newPixel = np.zeros(256, np.uint8)
    for i in range(256):
        newPixel[i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
    frame = newPixel[frame]
    return frame

# Function to view Histogram plots
def viewPlot(inputFrame, enhancedFrame, histequ, gamma):
    # Figures for plotting histograms
    fig = plt.figure()
    plt1 = fig.add_subplot(221) 
    plt2 = fig.add_subplot(222)
    plt3 = fig.add_subplot(223)
    plt4 = fig.add_subplot(224)

    # Plotting original image
    plt1.hist(inputFrame.ravel(),256,[0,256])
    plt1.set_title('Input Frame Histogram')
    plt1.set_xlabel('pixel')
    plt1.set_ylabel('pixel density')

    # Plotting histogram of enhanced image
    plt2.hist(enhancedFrame.ravel(),256,[0,256])
    plt2.set_title('Enhanced Frame Histogram')
    plt2.set_xlabel('pixel')
    plt2.set_ylabel('pixel density')

    # Plotting histogram output of histogram equalization 
    plt3.hist(histequ.ravel(),256,[0,256])
    plt3.set_title('Histogram Equalization')  
    plt3.set_xlabel('pixel')
    plt3.set_ylabel('pixel density')

    plt4.hist(gamma.ravel(),256,[0,256])
    plt4.set_title('Gamma Correction')  
    plt4.set_xlabel('pixel')
    plt4.set_ylabel('pixel density')
    plt.show()


def videoEnhancement(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # equ = cv2.equalizeHist(gray)
    # # ret, thresh1 = cv2.threshold(equ, 225, 150, cv2.THRESH_BINARY) 
    # # bilateral = cv2.bilateralFilter(equ, 9, 75, 75)
    # # medBlur = cv2.medianBlur(equ,5) 

    # Histogram Equalization on gray image
    equ = cv2.equalizeHist(gray)

    # Applying different techniques to improve image quality
    # # plt.hist(equ.ravel(),256,[0,256])
    # # print(p)
    # # plt.plot(np.arange(0,256,1),p)
    # # plt.show()
    # # kernel = np.ones((5, 5), np.uint8) 
    
    # # Using cv2.erode() method  
    # # image = cv2.erode(equ, kernel)  
    # # img_dilation = cv2.dilate(image, kernel)
    # # kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    # # im = cv2.filter2D(equ, -1, kernel)
    # # equ = cv2.adaptiveThreshold(equ, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)
    # equ = cv2.threshold(equ, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    frame32=np.asarray(frame,dtype="int32")
    # Multiplying factor
    alpha = 3.3
    # Adding factor
    beta = 50

    # Making darker pixels bright
    enhanceImage = frame32*alpha + beta
    enhanceImage = np.clip(enhanceImage,0,255)
    enhanceImage=np.asarray(enhanceImage,dtype="uint8")
    gamma = gammaCorrection(frame, 0.4)
    # To view Histogram plots uncomment the below line
    # viewPlot(frame, enhanceImage , equ, gamma)

    # Enhanced Image output
    cv2.imshow('Enhanced Video',enhanceImage)

    # Input frame
    cv2.imshow('Input Video',frame)

    cv2.imshow("Histogram Equalization", equ)

    cv2.imshow("Gamma", gamma)

    return enhanceImage


######################################################
#     Reading Input Videoand Creating output video 
#####################################################

# For reading input video
cap = cv2.VideoCapture('NightDrive.mp4')
fourcc = cv2.VideoWriter_fourcc(*'XVID')
output = cv2.VideoWriter('NightDriveOutput.avi',fourcc, 20.0,(960,540))

while(cap.isOpened()):
    ret, frame = cap.read()

    frame = cv2.resize(frame, None,fx=0.5, fy=0.5, interpolation = cv2.INTER_CUBIC)
    # print(frame.shape)
    enhancedFrame = videoEnhancement(frame)

    output.write(enhancedFrame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
output.release()
cv2.destroyAllWindows()

