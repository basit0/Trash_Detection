import numpy as np
import tensorflow as tf
import cv2
import time
import os
yolo_dir = os.path.join("coco")
conf=0.3
threshold = 0.3


labelspath = os.path.sep.join([yolo_dir, "coco.names"])
labels = open(labelspath).read().strip().split("\n")

np.random.seed(42)
colors = np.random.randint(0,255, size= (len(labels), 3), dtype= "uint8")


weights_path = os.path.sep.join([yolo_dir, "yolov3.weights"])
config_path = os.path.sep.join([yolo_dir, "yolov3.cfg"])


net = cv2.dnn.readNetFromDarknet(config_path, weights_path)

image = cv2.imread("4.jpeg")
(H,W) = image.shape[:2]
# determine only the *output* layer names that we need from YOLO
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# construct a blob from the input image and then perform a forward
# pass of the YOLO object detector, giving us our bounding boxes and
# associated probabilities
blob = cv2.dnn.blobFromImage(image, 1/255.0, (416,416), swapRB=True, crop=False)
net.setInput(blob)
start = time.time()
layeroutputs = net.forward(ln)
end = time.time()

# show timing information on YOLO
print("[INFO] YOLO took {:.6f} seconds".format(end - start))


# initialize our lists of detected bounding boxes, confidences, and class IDs, respectively
boxes = []
confidences =[]
classids = []
classes =[]

# loop over each of the layer outputs
for output in layeroutputs:
    # loop over each of the detections
    for detection in output:
        # extract the class ID and confidence (i.e., probability) of the current object detection
        scores = detection[5:]
        classid = np.argmax(scores)
        confidence = scores[classid]
        # filter out weak predictions by ensuring the detected probability is greater than the minimum probability
        if confidence > conf:
            # scale the bounding box coordinates back relative to the size of the image, keeping in mind that YOLO actually
            # returns the center (x, y)-coordinates of the bounding box followed by the boxes' width and height
            box = detection[0:4] * np.array([W, H, W, H])            
            (centerX, centerY, width, height)  = box.astype("int")
            # use the center (x, y)-coordinates to derive the top and and left corner of the bounding box
            X = int(centerX - (width/2))
            Y = int(centerY - (height/2))
            
            # update our list of bounding box coordinates, confidences, and class IDs
            boxes.append([X, Y, int(width), int(height)])
            confidences.append(float(confidence))
            classids.append(classid)

# apply non-maxima suppression to suppress weak, overlapping bounding boxes
idxs = cv2.dnn.NMSBoxes(boxes, confidences, conf, threshold)



# ensure at least one detection exists
if len(idxs) > 0:
    # loop over the indexes we are keeping
    for i in idxs.flatten():
        # extract the bounding box coordinates
        k=labels[classids[i]],classids[i]
        classes.append(k)
        (x, y) = (boxes[i][0], boxes[i][1])
        (w, h) = (boxes[i][2], boxes[i][3])
        # draw a bounding box rectangle and label on the image
        color = [int(c) for c in colors[classids[i]]]
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        text = "{}: {:.4f}".format(labels[classids[i]], confidences[i])
        cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
# show the output image
cv2.imshow("Image", image)
cv2.imwrite("filename.jpg", image)
print("[INFO] YOLO took  at the end {:.6f} seconds".format(end - start))
print("classids",classes)
#print("confidences",confidences)
#print("boxes",boxes)
cv2.waitKey(0)