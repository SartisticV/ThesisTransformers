import json
import math

with open('annotations_VisDrone_VisDrone2019-VID-test-dev.json', 'r') as file:
    data = json.load(file)

annotations = data['annotations']
images = data['images']
def calculate_speed(bbox1, bbox2):
    center1 = (bbox1[0] + bbox1[2] / 2, bbox1[1] + bbox1[3] / 2)
    center2 = (bbox2[0] + bbox2[2] / 2, bbox2[1] + bbox2[3] / 2)
    return math.sqrt((center2[0] - center1[0]) ** 2 + (center2[1] - center1[1]) ** 2)


annotation_dict = {}
video_frame_count = {}
instances_by_video = {}
imageid_to_frameid = {}

for im in images:
    imageid_to_frameid[im['id']] = im['frame_id']

for ann in annotations:
    frame = imageid_to_frameid[ann["image_id"]]
    key = (ann["instance_id"], ann["vid_id"], frame)
    annotation_dict[key] = ann

    vid_id = ann["vid_id"]
    video_frame_count[vid_id] = max(video_frame_count.get(vid_id, 0), frame)

    if vid_id not in instances_by_video:
        instances_by_video[vid_id] = set()
    instances_by_video[vid_id].add(ann["instance_id"])

print(video_frame_count)

for vid_id, instance_ids in instances_by_video.items():
    print(vid_id)
    for instance_id in instance_ids:
        prev_speed = 0
        for frame_id in range(1, video_frame_count[vid_id] + 1):
            current_key = (instance_id, vid_id, frame_id)
            next_key = (instance_id, vid_id, frame_id + 1)

            if current_key in annotation_dict:
                current_annotation = annotation_dict[current_key]

                if next_key in annotation_dict:
                    next_annotation = annotation_dict[next_key]
                    speed = calculate_speed(current_annotation["bbox"], next_annotation["bbox"])
                    if speed > 150:
                        speed = prev_speed
                else:
                    speed = prev_speed

                current_annotation["speed"] = speed
                prev_speed = speed

updated_annotations = list(annotation_dict.values())
print(updated_annotations)

# Extract all speed values from the updated annotations
speeds = [ann["speed"] for ann in updated_annotations if "speed" in ann]

data['annotations'] = updated_annotations

json.dump(data, open('speed_test.json', 'w'))
