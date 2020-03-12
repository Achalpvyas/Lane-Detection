import numpy as np
import cv2


def turnPrediction(frame, leftLane, rightLane):
    imageCenter = frame.shape[1]//2
    output = ""
    laneCenter = (leftLane + RightLane)//2
    if imageCenter > leftLane + laneCenter:
        output = output + "Vehicle is turning Left"
    elif imageCenter < leftLane + laneCenter:
        output = output + "Vehicle is turning right"
    else:
        output = output + "Vehicle is moving straight"
    
    frame = cv2.putText(frame, output, (100,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    return frame