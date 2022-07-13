import cv2
import numpy as np
import pyautogui
import pygetwindow as gw
import sys
import time
import os 
from imageai.Detection import ObjectDetection
from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()


obj_detect = ObjectDetection()
obj_detect.setModelTypeAsYOLOv3()
obj_detect.setModelPath(r"yolo.h5")
obj_detect.loadModel()

# define the codec
fourcc = cv2.VideoWriter_fourcc(*'h264')
# frames per second
fps = 60.0
# the time you want to record in seconds

# search for the window, getting the first matched window with the title
#w = gw.getWindowsWithTitle(window_name)[0]
# activate the window
#w.activate()


for i in range(int(fps)):
    # make a screenshot
    img = pyautogui.screenshot()
    # convert these pixels to a proper numpy array to work with OpenCV
    frame = np.array(img)
    # convert colors from BGR to RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    annotated_image, preds = obj_detect.detectObjectsFromImage(input_image=frame,
                    input_type="array",
                      output_type="array",
                      display_percentage_probability=True,
                      display_object_name=True)
    resize = cv2.resize(annotated_image, (960, 540))
    time.sleep(1/1000)
    cv2.imshow("Eglo Development beta", resize)
    # if the user clicks q, it exits
    if (cv2.waitKey(1) & 0xFF == ord("q")) or (cv2.waitKey(1)==27):
        break

# make sure everything is closed when exited
cv2.destroyAllWindows()