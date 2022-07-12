import cv2
import numpy as np
import pyautogui
import pygetwindow as gw
import sys
import random
import string
import os 
from imageai.Detection import ObjectDetection
from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()


obj_detect = ObjectDetection()
obj_detect.setModelTypeAsYOLOv3()
obj_detect.setModelPath(r"yolo.h5")
obj_detect.loadModel()
os.system('''/usr/bin/osascript -e 'tell app "Finder" to set frontmost of process "python" to true' ''')

# the window name, e.g "notepad", "Chrome", etc.
window_name = 'opera'

# define the codec
fourcc = cv2.VideoWriter_fourcc(*"XVID")
# frames per second
fps = 15.0
# the time you want to record in seconds
record_seconds = 10000000000000

# search for the window, getting the first matched window with the title
w = gw.getWindowsWithTitle(window_name)[0]
# activate the window
w.activate()


for i in range(int(record_seconds * fps)):
    # make a screenshot
    img = pyautogui.screenshot(region=(w.left, w.top, w.width, w.height))
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
    cv2.imshow("Eglo Development beta", resize)
    # if the user clicks q, it exits
    if cv2.waitKey(1) == ord("q"):
        break

# make sure everything is closed when exited
cv2.destroyAllWindows()