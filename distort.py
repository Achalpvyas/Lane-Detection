import cv2
import numpy as np


def undistortImage(image):

    #Camera Matrix
    K = np.array([[  1.15422732e+03 , 0.00000000e+00  , 6.71627794e+02],
     [  0.00000000e+00 ,  1.14818221e+03  , 3.86046312e+02],
     [  0.00000000e+00 ,  0.00000000e+00  , 1.00000000e+00]])

    #Distortion Coefficients
    dist = np.array([ -2.42565104e-01 , -4.77893070e-02 , -1.31388084e-03 , -8.79107779e-05,
        2.20573263e-02])

    destination = cv2.undistort(image, K, dist, None, K)

    return destination

# Sample testing
img = cv2.imread('0000000001.png')
image = undistortImage(img)
cv2.imshow('undistorted image',image)
if cv2.waitKey(0) & 0xFF == ord('q'):
    cv2.destroyAllWindows()
