# Deep-Learning-for-Dendritic-Spines-Detection
Benchmarking Yolov2, Faster-RCNN and Shape-Priors-CNN on dendritic spines detection

[Please read the report for more details](https://github.com/ily-R/Deep-Learning-for-Dendritic-Spines-Detection/blob/master/report.pdf)
### 1-Faster-RCNN:

<p align="center">
  <img src="https://github.com/ily-R/Deep-Learning-for-Dendritic-Spines-Detection/blob/master/Faster-RCNN/results/img1180.jpg?raw=true" alt="capture reconstruction"/>
</p>

The model is already trained.
* The final  weights are in `Faster-RCNN/inference_graph`.
* The results are in `Faster-RCNN/results`

* To re-calculate the predictions on the test set run `F1_score_and_predictions.py`
* To recalculate on other images change the path to the folder containing the images.

### 2-SP-CNN: 

<p align="center">
  <img src="https://github.com/ily-R/Deep-Learning-for-Dendritic-Spines-Detection/blob/master/SP-CNN/gt.JPG?raw=true" alt="capture reconstruction"/>
</p>

<p align="center">
  <img src="https://github.com/ily-R/Deep-Learning-for-Dendritic-Spines-Detection/blob/master/SP-CNN/edges.JPG?raw=true" alt="capture reconstruction"/>
</p>

run the jupyter notebook. Note that the train images are not complete (due to size issues), but they will give you a sense of the overall pipeline.

### 3-YOLOv2:


<p float="center">
  <img src="https://github.com/ily-R/Deep-Learning-for-Dendritic-Spines-Detection/blob/master/YOLOV2/results/658.jpg" />
  <img src="https://github.com/ily-R/Deep-Learning-for-Dendritic-Spines-Detection/blob/master/YOLOV2/results/268.jpg" /> 
  <img src="https://github.com/ily-R/Deep-Learning-for-Dendritic-Spines-Detection/blob/master/YOLOV2/results/1198.jpg" />
</p>

#### Testing:

* create a text file and name it `test.txt`. this file contain the path to images you want to test on. Get inspired by an already existing test.txt and train.txt to get an idea.

* To calculate the map, f1-score... at a specific threshold (say 0.5, change it to other values for precision-recall graph) do the following:

From darknet folder run :`./darknet detector map cfg/obj.data cfg/yolo-obj.cfg backup/yolo-obj_last.weights -thresh 0.5`

* To test on a specific image (say the image's name is img.png) do the following:

From darknet folder run: `./darknet detector test cfg/obj.data cfg/yolo-obj.cfg backup/yolo-obj_last.weights img.png -thresh 0.55 ` 

### References:

* [You Only Look Once: Unified, Real-Time Object Detection](https://arxiv.org/abs/1506.02640)
* [YOLO9000: Better, Faster, Stronger](https://arxiv.org/abs/1612.08242)
* [Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](https://arxiv.org/abs/1506.01497)
* [Deep Networks with Shape Priors for Nucleus Detection](https://arxiv.org/abs/1807.03135)
