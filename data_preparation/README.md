# Data preparation
This folder contains all the necessary python scripts to prepare data.


## add_speed.py
This script takes the ground truth labels exported by [prepare_visdrone.py](https://github.com/SartisticV/ThesisTransformers/blob/main/data_preparation/prepare_visdrone.py) and adds to each annotation a speed. Speed is calculated by calculating the translation of the center of the bounding box in pixels between the current frame and the next. Because VisDrone-VID makes some errors in reusing instance_id's, there is a possibility for this speed calculation to be very high. We filter outlying speed values.

## prepare_visdrone.py
Creates the labels for the VisDrone-VID dataset in COCO format, as well as the videos in the correct folder structure and ignore regions filled in.

## prepare_uadetrac.py
Creates the labels for the UADETRAC dataset in COCO format, as well as the videos in the correct folder structure and ignore regions filled in.