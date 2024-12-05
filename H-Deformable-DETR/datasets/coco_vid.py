# Modified by Qianyu Zhou and Lu He
# ------------------------------------------------------------------------
# TransVOD++
# Copyright (c) 2022 Shanghai Jiao Tong University. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from Deformable DETR
# Copyright (c) SenseTime. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
COCO dataset which returns image_id for evaluation.

Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py
"""
from pathlib import Path

import torch
import torch.utils.data
from pycocotools import mask as coco_mask
from .parsers.coco_video_parser import CocoVID
from .torchvision_datasets import CocoDetection as TvCocoDetection
from util.misc import get_local_rank, get_local_size
import datasets.transforms_multi as T
from torch.utils.data.dataset import ConcatDataset
import random
import numpy as np


class CocoDetection(TvCocoDetection):
    def __init__(self, img_folder, ann_file, transforms, return_masks, interval1, interval2, num_ref_frames=3,
                 is_train=True, filter_key_img=True, cache_mode=False, local_rank=0, local_size=1, interval=1, stride=2):
        super(CocoDetection, self).__init__(img_folder, ann_file,
                                            cache_mode=cache_mode, local_rank=local_rank, local_size=local_size,
                                            interval=interval)
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(return_masks)
        self.ann_file = ann_file
        self.frame_range = [-2, 2]
        self.num_ref_frames = num_ref_frames
        self.cocovid = CocoVID(self.ann_file)
        self.is_train = is_train
        self.filter_key_img = filter_key_img
        self.interval1 = interval1
        self.interval2 = interval2
        self.stride = stride

    def __getitem__(self, idx):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, target). target is the object returned by ``coco.loadAnns``.
        """
        imgs = []

        coco = self.coco
        img_id = self.ids[idx]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = coco.loadAnns(ann_ids)
        img_info = coco.loadImgs(img_id)[0]
        path = img_info['file_name']
        video_id = img_info['video_id']
        img = self.get_image(path)

        target = {'image_id': img_id, 'annotations': target}
        img, target = self.prepare(img, target)
        # imgs.append(img)

        if video_id == -1:
            for i in range(self.num_ref_frames):
                imgs.append(img)
        else:
            img_ids = self.cocovid.get_img_ids_from_vid(video_id)

            # Temporal Augmentation during training - WIP
            #if self.is_train:
            #    self.stride = random.choice(range(5))+1


            left = max(img_ids[0], img_id - (self.stride * (self.num_ref_frames // 2)))
            right = min(img_ids[-1], img_id + (self.stride * ((self.num_ref_frames-1) // 2)))

            if left == img_ids[0]:
                right = left + self.stride * (self.num_ref_frames-1)
            if right == img_ids[-1]:
                left = right - self.stride * (self.num_ref_frames-1)
            ref_img_ids = list(range(left, right+self.stride, self.stride))
            #print(ref_img_ids)

            for ref_img_id in ref_img_ids:
                ref_ann_ids = coco.getAnnIds(imgIds=ref_img_id)
                ref_img_info = coco.loadImgs(ref_img_id)[0]
                ref_img_path = ref_img_info['file_name']
                ref_img = self.get_image(ref_img_path)
                imgs.append(ref_img)
        if self._transforms is not None:
            imgs, target = self._transforms(imgs, target)

        return torch.stack(imgs, dim=1), target


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False):
        self.return_masks = return_masks

    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = convert_coco_poly_to_mask(segmentations, h, w)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        if self.return_masks:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        return image, target


def make_coco_transforms(image_set, test):
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    if not test:
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomResize([600], max_size=1000),
            normalize,
        ])

    if test:
        return T.Compose([
            # Comment out line below if testing on original dimensions is required
            T.RandomResize([600], max_size=1000),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


def build(image_set, args, test):
    root = Path(args.coco_path)
    assert root.exists(), f'provided COCO path {root} does not exist'
    mode = 'instances'
    PATHS = {
        "train_det": [(root / "train" / "data", root / "train" / 'labels.json')],
        "train_vid": [(root / "Data" / "VID", root / "annotations" / 'imagenet_vid_train.json')],
        "train_joint": [(root / "Data", root / "annotations" / 'imagenet_vid_train_joint_30.json')],
        "valua": [(root / "uadetrac" / "val" / "data", root / "uadetrac" / "val" / 'labels.json')],
        "uadetrac_ir": [(root / "uadetrac_ir" / "train" / "data", root / "uadetrac" / "train" / 'labels.json')],
        "valua_ir": [(root / "uadetrac_ir" / "val" / "data", root / "uadetrac_ir" / "val" / 'labels.json')],
        "uadetrac_ir_rf": [(root / "uadetrac_ir_rf" / "train" / "data", root / "uadetrac" / "train" / 'labels.json')],
        "valua_ir_rf": [(root / "uadetrac_ir_rf" / "val" / "data", root / "uadetrac" / "val" / 'labels.json')],
        "uadetrac_ir_rf_small": [(root / "uadetrac_ir_rf" / "train" / "data", root / 'labels_train_smallv5.json')],
        "valua_ir_small": [(root / "uadetrac_ir" / "val" / "data", root / 'labels_val_smallv5.json')],
        "valtwit": [(root / "twitcam" / "val" / "data", root / "twitcam" / "val" / 'labels.json')],
        "uadetrac": [(root / "uadetrac" / "train" / "data", root / "uadetrac" / "train" / 'labels.json')],
        "twitcam": [(root / "twitcam" / "full" / "data", root / "twitcam" / "full" / 'labels_train_1class.json')],
        "virat": [(root / "virat" / "train" / "data", root / "virat" / "train" / 'labels_train_1class.json')],
        "valvir": [(root / "virat" / "val" / "data", root / "virat" / "val" / 'labels.json')],
        "pesmod": [(root / "pesmodcoco" / "train" / "data", root / "pesmodcoco" / "train" / 'labels.json')],
        "valpes": [(root / "pesmodcoco" / "val" / "data", root / "pesmodcoco" / "val" / 'labels.json')],
        "visdrone": [(root / "visdrone_new" / "train" / "data", root / "visdrone_new" / "train" / 'labels.json')],
        "vistest": [(root / "visdrone_new" / "val" / "data", root / "visdrone_new" / "val" / 'labels.json')],
        "uav": [(root / "UAV123" / "data_seq" / "UAV123", root / "UAV123" / "data_seq" / 'labels_train_1class.json')]

    }
    datasets = []
    for (img_folder, ann_file) in PATHS[image_set]:
        dataset = CocoDetection(img_folder, ann_file, transforms=make_coco_transforms(image_set, test),
                                is_train=(not args.eval), interval1=args.interval1,
                                interval2=args.interval2, num_ref_frames=args.num_ref_frames, return_masks=args.masks,
                                cache_mode=args.cache_mode,
                                local_rank=get_local_rank(), local_size=get_local_size(), interval=args.interval, stride=args.stride)
        datasets.append(dataset)
    if len(datasets) == 1:
        return datasets[0]
    return ConcatDataset(datasets)


