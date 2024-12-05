# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import xml.etree.ElementTree as ET
from collections import defaultdict
import cv2
from pycocotools import mask as maskUtils
import numpy as np

import json
from tqdm import tqdm

CLASSES = ('car', 'van', 'bus', 'others')


def parse_args():
    parser = argparse.ArgumentParser(
        description='ImageNet VID to COCO Video format')
    parser.add_argument(
        '-i',
        '--input',
        help='root directory of ImageNet VID annotations',
    )
    parser.add_argument(
        '-o',
        '--output',
        help='directory to save coco formatted label file',
    )
    return parser.parse_args()

def convert_vid(VID, mode='train'):
    """Convert ImageNet VID dataset in COCO style.

    Args:
        VID (dict): The converted COCO style annotations.
        ann_dir (str): The path of ImageNet VID dataset.
        save_dir (str): The path to save `VID`.
        mode (str): Convert train dataset or validation dataset. Options are
            'train', 'val'. Default: 'train'.
    """
    assert mode in ['train', 'val']

    #ToDo: add yolo anns

    curr_path = '/data/uadetrac_ir'
    curr_path2 = '/data/uadetrac_ir_rf'
    if not os.path.exists(curr_path):
        os.mkdir(curr_path)
        os.mkdir(curr_path2)

    records = dict(
        vid_id=1,
        img_id=1,
        ann_id=1,
        global_instance_id=1,
        num_vid_train_frames=0,
        num_no_objects=0,
        small=0,
        medium=0,
        large=0,)
    obj_num_classes = dict()

    if mode == 'train':
        xml_dir = '/data/DETRAC-Train-Annotations-XML'
        img_dir = '/data/uadetrac/train/data'
    else:
        xml_dir = '/data/DETRAC-Test-Annotations-XML'
        img_dir = '/data/uadetrac/val/data'

    curr_path, curr_path2 = mkdir(curr_path, curr_path2, mode)
    curr_path, curr_path2 = mkdir(curr_path, curr_path2, 'data')

    for xml in tqdm(sorted(os.listdir(xml_dir))):
        xmlpath = osp.join(xml_dir, xml)
        tree = ET.parse(xmlpath)
        root = tree.getroot()
        instance_id_maps = dict()
        name = xml[:-4]

        _,_ = mkdir(curr_path, curr_path2, name)

        video = dict(
            id=records['vid_id'],
            name=name)
        VID['videos'].append(video)
        frames = []
        for frame in root[2:]:
            frames.append(int(frame.attrib['num']))

        vid_dir = osp.join(img_dir, name)
        fn = os.listdir(vid_dir)
        tl = 2
        for frame_id in range(1,len(fn)+1):

            img_prefix = osp.join(name, 'img%05d' % frame_id)
            #print(img_prefix)
            # parse XML annotation file
            width = 960
            height = 540
            file_name = f'{img_prefix}.jpg'

            img1 = cv2.imread(osp.join(img_dir, file_name))
            img2 = cv2.imread(osp.join(img_dir, file_name))

            iglist = []

            for ig in root[1]:
                x, y, w, h = [
                    int(float(ig.attrib['left'])),
                    int(float(ig.attrib['top'])),
                    int(float(ig.attrib['width'])),
                    int(float(ig.attrib['height']))
                ]

                #img1[y:y + h, x:x + w] = (0, 0, 0)
                iglist.append([x, y, w, h])

            cv2.imwrite(osp.join(curr_path, file_name), img1)

            cv2.imwrite(osp.join(curr_path2, file_name), img1)
            image = dict(
                file_name=file_name,
                height=height,
                width=width,
                id=records['img_id'],
                frame_id=frame_id,
                video_id=records['vid_id'],
                is_vid_train_frame=1)
            VID['images'].append(image)

            if frame_id not in frames:
                print(file_name, 'has no objects.')
                records['num_no_objects'] += 1
                records['img_id'] += 1
                continue


            for obj in root[tl][0].findall('target'):
                class_name = obj[1].attrib['vehicle_type']
                if class_name not in CLASSES:
                    continue
                category_id = CLASSES.index(class_name)+1
                x, y, w, h = [
                    int(float(obj[0].attrib['left'])),
                    int(float(obj[0].attrib['top'])),
                    int(float(obj[0].attrib['width'])),
                    int(float(obj[0].attrib['height']))
                ]

                if w*h<=32**2:
                    records['small'] += 1
                elif w*h>96**2:
                    records['large'] += 1
                else:
                    records['medium'] += 1

                ious = iouoverlap(iglist, [[x, y, w, h]])

                img1[y:y + h, x:x + w] = img2[y:y + h, x:x + w]

                iscrowd = False

                for i in ious:
                    if i>=0.5 and mode=='val':
                        iscrowd=True

                track_id = obj.attrib['id']
                if track_id in instance_id_maps:
                    instance_id = instance_id_maps[track_id]
                else:
                    instance_id = records['global_instance_id']
                    records['global_instance_id'] += 1
                    instance_id_maps[track_id] = instance_id
                occluded = obj.find('occlusion')
                ann = dict(
                    id=records['ann_id'],
                    video_id=records['vid_id'],
                    image_id=records['img_id'],
                    category_id=category_id,
                    instance_id=instance_id,
                    bbox=[x, y, w, h],
                    area=w * h,
                    iscrowd=iscrowd,
                    occluded=1 if occluded is not None else 0,
                    generated=0)
                if category_id not in obj_num_classes:
                    obj_num_classes[category_id] = 1
                else:
                    obj_num_classes[category_id] += 1
                VID['annotations'].append(ann)
                records['ann_id'] += 1
            cv2.imwrite(osp.join(curr_path2, file_name), img1)
            tl += 1
            records['img_id'] += 1
        records['vid_id'] += 1
    #if not osp.isdir(save_dir):
    #    os.makedirs(save_dir)
    json.dump(VID, open(f'uadetrac_vid_{mode}.json', 'w'))
    print(f'-----UA DETRAC {mode}------')
    print(f'{records["vid_id"]- 1} videos')
    print(f'{records["img_id"]- 1} images')
    print(
        f'{records["num_vid_train_frames"]} train frames for video detection')
    print(f'{records["num_no_objects"]} images have no objects')
    print(f'{records["ann_id"] - 1} objects')
    print(f'{records["small"]} small objects')
    print(f'{records["medium"]} medium objects')
    print(f'{records["large"]} large objects')
    print('-----------------------')
    #for i in range(1, len(CLASSES) + 1):
        #print(f'Class {i} {CLASSES[i - 1]} has {obj_num_classes[i]} objects.')

def mkdir(path1, path2, name):
    path1 = osp.join(path1, name)
    path2 = osp.join(path2, name)
    if not os.path.exists(path1):
        os.mkdir(path1)
    if not os.path.exists(path2):
        os.mkdir(path2)

    return path1, path2

def iouoverlap(dts, gts):
    dts = np.asarray(dts)
    gts = np.asarray(gts)
    ious = np.zeros((len(dts), len(gts)))
    for j, gt in enumerate(gts):
        gx1 = gt[0]
        gy1 = gt[1]
        gx2 = gt[0] + gt[2]
        gy2 = gt[1] + gt[3]
        garea = gt[2] * gt[3]
        for i, dt in enumerate(dts):
            dx1 = dt[0]
            dy1 = dt[1]
            dx2 = dt[0] + dt[2]
            dy2 = dt[1] + dt[3]
            darea = dt[2] * dt[3]

            unionw = min(dx2, gx2) - max(dx1, gx1)
            if unionw <= 0:
                continue
            unionh = min(dy2, gy2) - max(dy1, gy1)
            if unionh <= 0:
                continue
            t = unionw * unionh

            ious[i, j] = float(t) / garea
    return ious

def main():
    args = parse_args()

    categories = []
    for k, v in enumerate(CLASSES, 1):
        categories.append(
            dict(id=k, name=v))

    VID_train = defaultdict(list)
    VID_train['categories'] = categories
    convert_vid(VID_train, 'train')

    VID_val = defaultdict(list)
    VID_val['categories'] = categories
    convert_vid(VID_val, 'val')




if __name__ == '__main__':
    main()