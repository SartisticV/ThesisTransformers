import fiftyone as fo
import fiftyone.utils.coco as fouc
import json

name = "EvalBoxes"
dataset_type = fo.types.COCODetectionDataset

if fo.dataset_exists(name):
    fo.delete_dataset(name)
dataset = fo.Dataset(name)
dataset.add_dir(
    dataset_dir='./visdrone_new/val',
    dataset_type=dataset_type,
    label_field="ground_truth",
    include_id=True,
)

#Enter prediction json file here
with open('visdrone_new/val/prediction_videoswin_enc5x1.json', 'r') as fp:
    predictions_info = json.load(fp)

classes = dataset.default_classes
id_to_class = dict(zip(range(1, len(classes) + 1), classes))
print(id_to_class)

moving = ['uav0000009_03358_v', 'uav0000073_00600_v', 'uav0000073_04464_v', 'uav0000077_00720_v', 'uav0000088_00290_v', 'uav0000119_02301_v', 'uav0000188_00000_v', 'uav0000201_00000_v', 'uav0000249_00001_v', 'uav0000249_02688_v', 'uav0000297_00000_v', 'uav0000306_00230_v']
static = ['uav0000120_04775_v', 'uav0000161_00000_v', 'uav0000297_02761_v', 'uav0000355_00001_v', 'uav0000370_00001_v']

for sample in dataset:
    if sample.filepath.split('\\')[-2] in moving:
        sample.tags = ["moving"]
    else:
        sample.tags = ["static"]
    sample.save()

fouc.add_coco_labels(dataset,"predictions",predictions_info,id_to_class, coco_id_field='ground_truth_coco_id', label_type="detections")

# Update all GT detections to have iscrowd=True
for sample in dataset:
    detections = sample.ground_truth_detections.detections
    for detection in detections:
        detection.iscrowd = True  # Set iscrowd to True for each GT object
    sample.ground_truth_detections.detections = detections
    sample.save()

# Evaluate detections on moving camera images
moving_view = dataset.match_tags("moving")
results_moving = moving_view.evaluate_detections("predictions", gt_field="ground_truth_detections", eval_key="predictions", classwise=False)

# Evaluate detections on static camera images
static_view = dataset.match_tags("static")
results_static = static_view.evaluate_detections("predictions", gt_field="ground_truth_detections", eval_key="predictions", classwise=False)

results = dataset.evaluate_detections(
    "predictions",
    gt_field="ground_truth_detections",
    eval_key="predictions",
    classwise=False,
)

results.print_report(classes=classes, digits=3)
results_moving.print_report(classes=classes, digits=3)
results_static.print_report(classes=classes, digits=3)

session = fo.launch_app(dataset, desktop=True)
plot = results.plot_confusion_matrix(classes=classes)
plot.show()

