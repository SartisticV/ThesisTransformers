# H-Deformable-DETR
H-Deformable-DETR ([Github](https://github.com/HDETR/H-Deformable-DETR), [Paper](https://arxiv.org/abs/2207.13080)) is altered to include a video swin backbone.


## Setup
This implementation of H-Deformable DETR comes with the necessary docker files needed to set up the environment. After first launching the docker, run the [setup.sh](https://github.com/SartisticV/ThesisTransformers/blob/main/H-Deformable-DETR/setup.sh) script to complete the setup of the project.

On a SLURM system, a conda environment can be created instead. To set up the same environment, the following steps need to be taken:
1. `conda install pytorch=1.9.0a0+df837d0 torchvision cudatoolkit=11.2.1 -c pytorch`
2. `conda install gxx_linux-64`
3. `FORCE_CUDA=1 pip install mmcv-full==1.3.17 mmdet==2.28.2 timm==1.0.7 torch==1.9.0 numpy==1.26.4`
4. `pip install -r requirements.txt`
5. Run [setup.sh](https://github.com/SartisticV/ThesisTransformers/blob/main/H-Deformable-DETR/setup.sh) 
6.  ````
    git clone https://github.com/NVIDIA/apex -b 22.04-dev
    cd apex
    python setup.py install

## Training & Weights

The current functionality of the project includes the standard single-frame H-Deformable-DETR model, together with 3 multi-frame variations:

* Single-frame: run with [start_sf.sh](https://github.com/SartisticV/ThesisTransformers/blob/main/H-Deformable-DETR/start_sf.sh) (or [start_slurm_sf.sh](https://github.com/SartisticV/ThesisTransformers/blob/main/H-Deformable-DETR/start_slrum_sf.sh) for SLURM). Standard implementation of H-Deformable-DETR.

* Multi-frame with sum fusion: run with [start_mf.sh](https://github.com/SartisticV/ThesisTransformers/blob/main/H-Deformable-DETR/start_slurm_mf.sh) (or [start_slurm_mf.sh](https://github.com/SartisticV/ThesisTransformers/blob/main/H-Deformable-DETR/start_slrum_mf.sh) for SLURM), and changing the --fusion argument to 'sum'. Simple summation over the temporal dimension.

* Multi-frame with Multi-head Attention fusion: run with [start_mf.sh](https://github.com/SartisticV/ThesisTransformers/blob/main/H-Deformable-DETR/start_slurm_mf.sh) (or [start_slurm_mf.sh](https://github.com/SartisticV/ThesisTransformers/blob/main/H-Deformable-DETR/start_slrum_mf.sh) for SLURM), and changing the --fusion argument to 'mha'. Uses Multi-head Attention to iteratively combine frames with an added skip connection.

* Multi-frame with adjusted Transformer encoder: run with [start_mf.sh](https://github.com/SartisticV/ThesisTransformers/blob/main/H-Deformable-DETR/start_slurm_mf.sh) (or [start_slurm_mf.sh](https://github.com/SartisticV/ThesisTransformers/blob/main/H-Deformable-DETR/start_slrum_mf.sh) for SLURM), and changing the --fusion argument to 'enc'. Keeps the backbone output the same, but changes the encoder input projection layer to accommodate the extra temporal dimension.

Currently, swin transformer weights are used to initialise the video swin backbone. In the [multi-frame shell script](https://github.com/SartisticV/ThesisTransformers/blob/main/H-Deformable-DETR/start_mf.sh), the argument --pretrained2d is added, which will inflate the backbone weights given to --pretrained_backbone_path using [this](https://github.com/SartisticV/ThesisTransformers/blob/8b3a229bca8facd96c73b93bd343d548d8ea19ae/H-Deformable-DETR/models/video_swin_transformer.py#L599) function. Transformer weights are loaded in seperately using --resume. Thus, it is possible to load in a combination of weights:

* Inflated [Swin Transformer weights](https://github.com/microsoft/Swin-Transformer) + resuming with [H-Deformable-DETR weights](https://github.com/HDETR/H-Deformable-DETR)

* All [H-Deformable-DETR weights](https://github.com/HDETR/H-Deformable-DETR), with backbone weights inflated

* [Video Swin Transformer weights](https://github.com/SwinTransformer/Video-Swin-Transformer) + resuming with [H-Deformable-DETR weights](https://github.com/HDETR/H-Deformable-DETR)

## Video Data
The video swin backbone requires a number of frames (given as the argument --num_ref_frames) and the stride between frames (given as the argument --stride), set in the [multi-frame shell script](https://gitlab.tsn.tno.nl/intelligent_imaging/students/veraarpr/-/blob/main/H-Deformable-DETR/start_mf.sh?ref_type=heads). Frames will always be selected around the current frame, with a preference to past frames. So if the num_ref_frames is 4 and the stride 2, with the current frame id being 32 as an example, the sampled frames will be [28,30,**32**,34].

Dataset specific parameters are set in the [config file](https://github.com/SartisticV/ThesisTransformers/blob/main/H-Deformable-DETR/configs/two_stage/deformable-detr-hybrid-branch/36eps/swin/swin_tiny_hybrid_branch_lambda1_group6_t1500_dp0_mqs_lft_deformable_detr_plus_iterative_bbox_refinement_plus_plus_two_stage.sh). The --interval argument in the config controls the interval between each frame from the dataset we train/evaluate on. So if --interval 10, every 10th frame of the dataset will be sampled. The video swin backbone will still have access to all frames of the dataset, so this interval choice does not affect the frames sampled by the backbone.

## Dataloader
The datasets for the swin backbone and the video swin backbone are loaded in with [coco.py](https://github.com/SartisticV/ThesisTransformers/blob/c64168861d9878201bf278bf254867167d8f6bff/H-Deformable-DETR/datasets/coco.py#L171) and [coco_vid.py](https://github.com/SartisticV/ThesisTransformers/blob/c64168861d9878201bf278bf254867167d8f6bff/H-Deformable-DETR/datasets/coco_vid.py#L210) respectively. Put here the path to your train and test set. The argument --coco_path must be given as the shared root directory of all your datasets.

## COCOVID format
### Annotations
The COCOVID format is an extension of the conventional COCO format. COCOVID format introduces a new 'videos' dictionary alongside the existing 'categories', 'images', and 'annotations' keys. The resulting json file will thus have the following structure:

`{categories: [...], images: [...], videos: [...], annotations: [...]}`

1. categories\
The 'categories' key defines the classes present in the dataset. Each category is represented by a dictionary with 'id' and 'name' attributes.

    `"categories": [
    {"id": 1, "name": "car"},
    {"id": 2, "name": "van"},
    ...
    ]`
2. images\
The 'images' key contains information about each image in the dataset. Each image is represented by a dictionary with the following attributes:

    * file_name: file name, including path if stored per video
    * height
    * width
    * id: unique image id
    * video_id: id of associated video
    * frame_id: position of frame in the associated video

    `"images": [{"file_name": "MVI_20011/img00001.jpg", "height": 540, "width": 960, "id": 1, "frame_id": 1, "video_id": 1}, {"file_name": "MVI_20011/img00002.jpg", "height": 540, "width": 960, "id": 2, "frame_id": 2, "video_id": 1}, ... ]`
3. videos\
The 'videos' key gives information about the videos in the dataset. Each video is represented by a dictionary with 'id' and 'name' attributes.

    `"videos": [{"id": 1, "name": "MVI_20011"}, {"id": 2, "name": "MVI_20012"}, ... ]`
4. annotations\
The 'annotations' key includes details about the labeled objects in the images, linking them to specific categories and images. Each image is represented by a dictionary with the following attributes:
    * id: unique annotation id
    * instance_id: unique identifier for instances across frames (-1 if no tracking information available)
    * image_id: id of associated image
    * video_id: id of associated video
    * frame_id: position of associated image in associated video
    * category_id: category id of annotated objects
    * bbox: bounding box values of annotated object in [x_min, y_min, width, height] format.
    * area: area of bounding box
    * iscrowd: boolean value indicating whether the annotated object is ignored during training/testing

    `"annotations": [{"id": 1, "video_id": 1, "image_id": 1, "category_id": 1, "instance_id": 1, "bbox": [592, 378, 160, 162], "area": 25920, "iscrowd": false}, {"id": 2, "video_id": 1, "image_id": 1, "category_id": 1, "instance_id": 2, "bbox": [557, 120, 47, 43], "area": 2021, "iscrowd": false}, ... ]`

Note: more attributes can be added to each dictionary if applicable. Some datasets provide additional information, for example whether an object is occluded or not. The attributes mentioned previously seem to be the minimum required for most models.

### Predictions
Predictions in COCOVID format work the same as in COCO format. Predictions are stored in a json file as a list of dictionaries, with each containing the following attributes:
* image_id: id of the image on which the prediction is made
* category_id: predicted class
* bbox: bounding box values of the prediction in [x_min, y_min, width, height] format
* score: confidence score of the prediction

    `[{"image_id": 1, "category_id": 1, "bbox": [743.0, 359.0, 151.0, 108.0], "score": 0.8037319183349609}, {"image_id": 1, "category_id": 2, "bbox": [743.0, 359.0, 151.0, 108.0], "score": 0.046543557196855545}, ... ]`

