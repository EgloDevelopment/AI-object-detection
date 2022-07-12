from imageai.Detection import ObjectDetection

from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()

obj_detect = ObjectDetection()
obj_detect.setModelTypeAsYOLOv3()
obj_detect.setModelPath(r"yolo.h5")
obj_detect.loadModel()

import cv2

cam_feed = cv2.VideoCapture(0)
cam_feed.set(cv2.CAP_PROP_FRAME_WIDTH, 900)
cam_feed.set(cv2.CAP_PROP_FRAME_HEIGHT, 900)

while True:    
    ret, img = cam_feed.read()   
    annotated_image, preds = obj_detect.detectObjectsFromImage(input_image=img,
                    input_type="array",
                      output_type="array",
                      display_percentage_probability=True,
                      display_object_name=True)

    cv2.imshow("Detection software", annotated_image)     
    
    if (cv2.waitKey(1) & 0xFF == ord("q")) or (cv2.waitKey(1)==27):
        break

cam_feed.release()
cv2.destroyAllWindows()