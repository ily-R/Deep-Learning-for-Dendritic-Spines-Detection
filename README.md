# Deep-Learning-for-Dendritic-Spines-Detection
Benchmarking Yolov2, Faster-RCNN and Shape-Priors-CNN on dendritic spines detection

1- **Faster-RCNN**:

The model is already trained and the final weights are in Faster-RCNN/inference_graph, also the results are in Faster-RCNN/results.
To re-calculate the predictions on the test set run F1_score_and_predictions.py. to recalculate on other images change the path to the folder 
containing the images.

2- **SP-CNN**: 

run the jupyter notebook. Note that the train images are not complete (due to size issues), but they will give you a sense of the overall pipeline.

3- **YOLOv2**:

**FIRST**: create a text file and name it test.txt. this file contain the path the images you want to test on. see already existing test.txt and train.txt to get an idea.

-To calculate the map, f1-score... at a specific threshold (say 0.5, change it to other values for precision-recall graph) do the following:
----from darknet folder run :"./darknet detector map cfg/obj.data cfg/yolo-obj.cfg backup/yolo-obj_last.weights -thresh 0.5"

-To test on a specific image (the image's name is img.png) do the following:
----From darknet folder run: "./darknet detector test cfg/obj.data cfg/yolo-obj.cfg backup/yolo-obj_last.weights img.png -thresh 0.55 "  
