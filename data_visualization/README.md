# Data visualization
This folder contains all the necessary python scripts to visualize results.


## classcount.py
Plots the breakdown of objects on size for both the training and testing set.

## combine_video.py
This script is a simple function that puts two of the same frame stitched together, where the top will be used for video swin detections and the bottom for swin detections.

## eval_cm.py
Calculates the evaluation scores per class on different video characteristics: all videos, moving camera videos and static camera videos. This script uses fiftyone's evaluate_detection function to report precision, recall and f1 scores. Additionally, a confusion matrix is plotted. However, fiftyone classifies all false positives as having no matching object at all. This means all false positives will have the 'none' class as the confused class, even if there is overlap with an actual other object. The provided file [fiftyone/coco.py](https://github.com/SartisticV/ThesisTransformers/blob/main/data_visualization/fiftyone/coco.py) can be used to replace the original [fiftyone](https://github.com/voxel51/fiftyone/blob/develop/fiftyone/utils/eval/coco.py) implementation in the local fiftyone library download.

## offset_labels.py
Creates a prediction and annotation json file that fits for the combine_video.py output. The prediction jsons of both the multi-frame model and single-frame model are combined into a single json and adjusted for position. The annotation json is duplicated so both the top and bottom part of the video have the ground truth labels available.

## plot_functions
Contains the functions to plot the precision-confidence, recall-confidence, precision-recall and f1-confidence curves.

## video_display
Code to display the differences between single-frame and multi-frame models in video, using the videos outputted by [combine_video.py](https://github.com/SartisticV/ThesisTransformers/blob/main/data_visualization/combine_video.py) and the labels from [offset_labels.py](https://github.com/SartisticV/ThesisTransformers/blob/main/data_visualization/offset_labels.py).