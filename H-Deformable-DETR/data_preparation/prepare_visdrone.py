import glob
import imagesize
import os
import json
from PIL import Image
from tqdm import tqdm
import argparse
import numpy as np
#import cv2


def convert(dir_data):
    train_data = dir_data + '/VisDrone2019-VID-train/'
    #val_data = dir_data + '/VisDrone2019-VID-val/'
    test_data = dir_data + '/VisDrone2019-VID-test-dev/'
    loops = [train_data, test_data]


    #train_ir_rf = dir_data + '/visdrone_new/train/'
    #test_ir = dir_data + '/visdrone_new/val/'

    for l in loops:
        if 'train' in l:
            mode = 'train'
        else:
            mode = 'test'

        print('Solving ', l)
        dict_coco = {}

        records = dict(
            vid_id=1,
            img_id=1,
            ann_id=1,
            )

        dir_imgs = 'sequences/'
        dir_labels = 'annotations/'

        ''' Key: images '''
        print('Solving images')
        dict_image_and_id = {}
        dict_coco['images'] = []
        dict_coco['videos'] = []
        dict_coco['annotations'] = []

        for vid in tqdm(sorted(os.listdir(l + dir_imgs))):
            frame_id = 1

            dict_coco['videos'].append({
                "id": records['vid_id'],
                "name": vid
            })
            for img in sorted(os.listdir(l + dir_imgs + vid)):
                # image = Image.open(img)
                file_name_save = vid + '/' + img
                width, height = imagesize.get(l + dir_imgs + file_name_save)
                # file_name = os.path.split(img)
                dict_coco['images'].append({
                    "id": records['img_id'],
                    "frame_id": frame_id,
                    "video_id": records['vid_id'],
                    "height": height,
                    "width": width,
                    "file_name": file_name_save,
                    "is_vid_train_frame": 1
                })
                dict_image_and_id[records['img_id']] = file_name_save

                """
                if mode == 'train':
                    img1 = cv2.imread(os.path.join(train_data, dir_imgs, file_name_save))
                    if not os.path.exists(os.path.join(train_ir_rf, dir_imgs, vid)):
                        os.makedirs(os.path.join(train_ir_rf, dir_imgs, vid))
                    cv2.imwrite(os.path.join(train_ir_rf, dir_imgs, file_name_save), img1)
                else:
                    img1 = cv2.imread(os.path.join(test_data, dir_imgs, file_name_save))
                    if not os.path.exists(os.path.join(test_ir, dir_imgs, vid)):
                        os.makedirs(os.path.join(test_ir, dir_imgs, vid))
                    cv2.imwrite(os.path.join(test_ir, dir_imgs, file_name_save), img1)
                """
                records['img_id'] += 1
                frame_id+=1

            offset = records['img_id'] - frame_id

            annotations = open(l + dir_labels + vid + '.txt').read()
            annotations = annotations.split('\n')
            #print(annotations)

            for i in range(0, len(annotations)):
                annotations[i] = annotations[i].split(',')

            ir_list = []

            for detection in tqdm(sorted(annotations[:-1], key=lambda x: x[7]),position=0, leave=True):
                if detection[7] == '0':
                    x,y,w,h = int(detection[2]), int(detection[3]), int(detection[4]), int(detection[5])
                    file_name = dict_image_and_id[int(detection[0])+offset]
                    """
                    if mode == 'train':
                        img2 = cv2.imread(os.path.join(train_ir_rf, dir_imgs, file_name))
                        img2[y:y + h, x:x + w] = (0, 0, 0)
                        cv2.imwrite(os.path.join(train_ir_rf, dir_imgs, file_name), img2)
                    else:
                        img2 = cv2.imread(os.path.join(test_ir, dir_imgs, file_name))
                        img2[y:y + h, x:x + w] = (0, 0, 0)
                        cv2.imwrite(os.path.join(test_ir, dir_imgs, file_name), img2)
                    """
                    ir_list.append([x,y,w,h])


                else:
                    #print(detection)
                    category_id = int(detection[7])
                    x,y,w,h = int(detection[2]), int(detection[3]), int(detection[4]), int(detection[5])
                    area = int(detection[4]) * int(detection[5])
                    iscrowd = False


                    if mode == 'train':
                        file_name = dict_image_and_id[int(detection[0]) + offset]
                        """
                        img1 = cv2.imread(os.path.join(train_data, dir_imgs, file_name))
                        img2 = cv2.imread(os.path.join(train_ir_rf, dir_imgs, file_name))

                        img2[y:y + h, x:x + w] = img1[y:y + h, x:x + w]

                        cv2.imwrite(os.path.join(train_ir_rf, dir_imgs, file_name), img2)
                    """
                    else:
                        ious = iouoverlap(ir_list, [[x, y, w, h]])

                        for i in ious:
                            if i >= 0.5:
                                iscrowd = True

                    dict_coco['annotations'].append({
                        "id": records['ann_id'],
                        "instance_id": int(detection[1]),
                        "image_id": int(detection[0])+offset,
                        "vid_id": records['vid_id'],
                        "category_id": category_id,
                        "bbox": [x,y,w,h],
                        "area": area,
                        "iscrowd": iscrowd,
                        "ignore": 0,
                        "occluded": int(detection[9])
                    })

                    records['ann_id'] += 1

            records["vid_id"] += 1

        ''' Key: categories '''

        '''
        pedestrian (1), people (2), bicycle (3), car (4), van (5), 
        truck (6), tricycle (7), awning-tricycle (8), bus (9), motor (10), others (11)
        '''
        dict_coco['categories'] = [{
            "id": 1,
            "name": "pedestrian",
            "supercategory": "none"},
            {
                "id": 2,
                "name": "people",
                "supercategory": "none"},
            {
                "id": 3,
                "name": "bicycle",
                "supercategory": "none"},
            {
                "id": 4,
                "name": "car",
                "supercategory": "none"},
            {
                "id": 5,
                "name": "van",
                "supercategory": "none"},
            {
                "id": 6,
                "name": "truck",
                "supercategory": "none"},
            {
                "id": 7,
                "name": "tricycle",
                "supercategory": "none"},
            {
                "id": 8,
                "name": "awning-tricycle",
                "supercategory": "none"},
            {
                "id": 9,
                "name": "bus",
                "supercategory": "none"},
            {
                "id": 10,
                "name": "motor",
                "supercategory": "none"},
            {
                "id": 11,
                "name": "others",
                "supercategory": "none"}
        ]

        with open('annotations_VisDrone_' + l.split('/')[-2] + '.json', 'w') as f:
            json.dump(dict_coco, f)


def get_args():
    parser = argparse.ArgumentParser('Train')
    parser.add_argument('--data_dir', type=str, default='.',
                        help='Data dir', dest='data_dir')
    args = parser.parse_args()

    return args

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


if __name__ == '__main__':
    args = get_args()
    convert(args.data_dir)