#Generating Video from Images using OpenCV-Python
import cv2
import numpy as np
import glob
 
img_array = []
names = []
for filename in glob.glob('./Data/data_1/data/*.png'):
    names.append(filename)

names.sort()

for filename in names:
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width,height)
    img_array.append(img)

out = cv2.VideoWriter('video.avi',cv2.VideoWriter_fourcc(*'DIVX'), 10, size)
 
for i in range(len(img_array)):
    out.write(img_array[i])
out.release()
















