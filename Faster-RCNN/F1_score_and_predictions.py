#
# Author: Ilyas Aroui
# Date: 29/03/19
# Description: 
# This script calculates the predictions and the scores: F1, precision and recall on all the test images.
# you can also change the decision threshold multiple times to get a precision-recall graph.

## The whole code is built upon this file
## https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10/blob/master/Object_detection_image.py

## I adapted it to cover all the test set and calculated the scores using IOU (intersection over union) script.

# Import packages
import os
import cv2
import numpy as np
import tensorflow as tf
import sys
import pandas as pd
# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

# Import utilites
from utils import label_map_util
from utils import visualization_utils as vis_util


def union(au, bu, area_intersection):
	area_a = (au[2] - au[0]) * (au[3] - au[1])
	area_b = (bu[2] - bu[0]) * (bu[3] - bu[1])
	area_union = area_a + area_b - area_intersection
	return area_union


def intersection(ai, bi):
	x = max(ai[0], bi[0])
	y = max(ai[1], bi[1])
	w = min(ai[2], bi[2]) - x
	h = min(ai[3], bi[3]) - y
	if w < 0 or h < 0:
		return 0
	return w*h


def iou(a, b):
	# a and b should be (x1,y1,x2,y2)

	if a[0] >= a[2] or a[1] >= a[3] or b[0] >= b[2] or b[1] >= b[3]:
		return 0.0

	area_i = intersection(a, b)
	area_u = union(a, b, area_i)

	return float(area_i) / float(area_u + 1e-6)


# Name of the directory containing the object detection module we're using
MODEL_NAME = 'inference_graph'
IMAGE_NAME = 'images/test/'

data = pd.read_csv("images/test_labels.csv")
data = data.values
img_names = np.unique(data[:, 0])
# Grab path to current working directory
CWD_PATH = os.getcwd()

# Path to frozen detection graph .pb file, which contains the model that is used
# for object detection.
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,'training','labelmap.pbtxt')



# Number of classes the object detector can identify
NUM_CLASSES = 1
TP = 0
FP = 0
FN = 0
# Load the label map.
# Label maps map indices to category names, so that when our convolution
# network predicts `5`, we know that this corresponds to `king`.
# Here we use internal utility functions, but anything that returns a
# dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=False)
category_index = label_map_util.create_category_index(categories)

# Load the Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.Session(graph=detection_graph)

# Define input and output tensors (i.e. data) for the object detection classifier

# Input tensor is the image
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

# Output tensors are the detection boxes, scores, and classes
# Each box represents a part of the image where a particular object was detected
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

# Each score represents level of confidence for each of the objects.
# The score is shown on the result image, together with the class label.
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

# Number of objects detected
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

# Load image using OpenCV and
# expand image dimensions to have shape: [1, None, None, 3]
# i.e. a single-column array, where each item in the column has the pixel RGB value
for i in range(img_names.shape[0]):
  # Path to image
  PATH_TO_IMAGE = os.path.join(CWD_PATH,IMAGE_NAME + img_names[i])
  image = cv2.imread(PATH_TO_IMAGE)
  image_expanded = np.expand_dims(image, axis=0)
  # Perform the actual detection by running the model with the image as input
  (boxes, scores, classes, num) = sess.run(
    [detection_boxes, detection_scores, detection_classes, num_detections],
    feed_dict={image_tensor: image_expanded})
    
  # Draw the results of the detection (aka 'visulaize the results')
  _, drawn_boxes = vis_util.visualize_boxes_and_labels_on_image_array(
      	image,
      	np.squeeze(boxes),
      	np.squeeze(classes).astype(np.int32),
      	np.squeeze(scores),
      	category_index,
      	use_normalized_coordinates=True,
      	line_thickness=1,
      	min_score_thresh=0.60)
  GT_boxes = data[np.where(data[:,0]==img_names[i])[0], 4:]
  tp = 0
  # print(GT_boxes)
  # print(drawn_boxes)
  for pred_box in drawn_boxes:
  	(pred_x1, pred_y1, pred_x2, pred_y2)= pred_box
  	for j in range(GT_boxes.shape[0]):
  		# iou_map = iou((pred_x1, pred_y1, pred_x2, pred_y2), (GT_boxes[j,0],GT_boxes[j,1],GT_boxes[j,2], GT_boxes[j,3]))
  		# print(iou_map)
  		distance = np.sqrt((pred_x1-GT_boxes[j,0])**2+(pred_y1 - GT_boxes[j,1])**2)
  		if distance <= 6:
  			tp += 1
  			break  
  # for j in range(GT_boxes.shape[0]):
  #   # cv2.rectangle(image, (GT_boxes[j,0],GT_boxes[j,1]), (GT_boxes[j,2], GT_boxes[j,3]), (0,0,255), 1)
  #   cv2.circle(image, (GT_boxes[j,0]+ 7,GT_boxes[j,1] + 7), 8, (0,0,255), thickness=1)

  # All the results have been drawn on image. Now display the image.
  print('\n tp = ', tp)
  TP += tp
  FP += len(drawn_boxes) - tp
  FN += GT_boxes.shape[0] - tp
  print(i+1)
#   scale_percent = 220 # percent of original size
#   width = int(image.shape[1] * scale_percent / 100)
#   height = int(image.shape[0] * scale_percent / 100)
#   dim = (width, height)
# # resize image
#   resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
#   cv2.imwrite('results/'+img_names[i],resized)
  # cv2.imshow('Object detector', resized)
  # # Press any key to close the image
  # cv2.waitKey(0)
  # # Clean up
  # cv2.destroyAllWindows()

precision = TP/(TP+FP)
recall = TP/(TP+FN)
F1 = 2*precision*recall/(precision+recall)

print("\n precision: ", precision )
print("\n recall: ", recall)
print("\n F1 score: ",F1)
