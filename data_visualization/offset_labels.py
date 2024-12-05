import cv2
import os
from tqdm import tqdm
import json

folder = './visdrone_new/val/data'
pred_sf = './visdrone_new/val/prediction_swin.json'
pred_mf = './visdrone_new/val/prediction_videoswin_enc5x1.json'
gt = './visdrone_new/val/labels.json'

with open(pred_sf, 'r') as file:
    detections_sf = json.load(file)

with open(pred_mf, 'r') as file1:
    detections_mf = json.load(file1)

with open(gt, 'r') as file2:
    gt_data = json.load(file2)

gt_annotations = gt_data['annotations'][:]
vidname_to_id = {g['name']: g['id'] for g in gt_data['videos']}
vidid_to_height = {vidname_to_id[vid]: cv2.imread(os.path.join(folder, vid, os.listdir(os.path.join(folder, vid))[0])).shape[0] for vid in os.listdir(folder)}
id_to_vidid = {g['id']: g['video_id'] for g in gt_data['images']}

total_iterations = len(detections_sf) + len(gt_annotations) + len(detections_mf)
total_annotations = len(gt_annotations)

with tqdm(total=total_iterations) as pbar:
    for d in detections_mf:
        x, y, w, h = d['bbox']
        d['bbox'] = [x, y, w, h]
        pbar.update(1)

    for d in detections_sf:
        x, y, w, h = d['bbox']
        offset = vidid_to_height[id_to_vidid[d['image_id']]]
        d['bbox'] = [x, (y + offset), w, h]
        detections_mf.append(d)
        pbar.update(1)

    for g in gt_data['annotations']:
        g_new = g.copy()
        x, y, w, h = g['bbox']
        offset = vidid_to_height[id_to_vidid[g['image_id']]]
        g['bbox'] = [x, y, w, h]
        g_new['bbox'] = [x, (y + offset), w, h]
        g_new['id'] = g['id'] + total_annotations
        # Append the modified annotation to gt_annotations
        gt_annotations.append(g_new)
        pbar.update(1)

# Update gt_data['annotations'] with the modified gt_annotations
gt_data['annotations'] = gt_annotations

with open('predict_offset.json', 'w') as f2:
    json.dump(detections_mf, f2)

with open('labels_offset.json', 'w') as f3:
    json.dump(gt_data, f3)
