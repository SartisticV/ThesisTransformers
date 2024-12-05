import json
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from statsmodels.stats.contingency_tables import mcnemar

iou_levels = torch.linspace(0.25, 0.95, 10).tolist()
def create_function_inputs(gt_json_file, detections_json_file):
    # Read ground truth JSON file
    with open(gt_json_file, 'r') as file:
        gt_data = json.load(file)

    names = {n['id']:n['name'] for n in gt_data['categories']}
    names[0] = '0'
    print('classes: ', names)

    # Read detections/predictions JSON file
    with open(detections_json_file, 'r') as file:
        detections_data = json.load(file)

    # Extract ground truth information
    print('Extracting ground truth information...')
    gt_boxes = defaultdict(list)
    for gt in gt_data['annotations']:
        gt_boxes[gt['image_id']].append((gt['bbox'], gt['iscrowd'], gt['category_id'], gt['speed']))

    # Initialize an array to store whether each detection is a true positive
    tp = np.zeros((len(detections_data),len(iou_levels)), dtype=bool)
    tp_speed = np.zeros((len(detections_data)), dtype=bool)
    speeds = np.zeros((len(detections_data)), dtype=float)
    conf = np.zeros(len(detections_data), dtype=float)
    pred_cls = np.zeros(len(detections_data), dtype=int)
    target_cls = np.zeros(len(gt_data['annotations']), dtype=int)

    for i in range(len(gt_data['annotations'])):
        target_cls[i] = gt_data['annotations'][i]['category_id']

    # Iterate over each detection
    print('Processing detections...')
    with tqdm(total=len(detections_data)) as pbar:
        for i, detection in enumerate(detections_data):
            image_id = detection['image_id']
            category_id = detection['category_id']
            detection_bbox = detection['bbox']
            score = detection['score']

            pred_cls[i] = category_id
            conf[i] = score

            iou_curr = 0
            # Check if there is a corresponding ground truth bounding box for the same image and category
            for gt_bbox, iscrowd, gt_cat, speed in gt_boxes[image_id]:
                # Calculate IoU between the ground truth and detection bounding boxes
                iou_value = iou([detection_bbox], [gt_bbox], [iscrowd])[0][0]

                if iou_value <= iou_curr:
                    continue
                else:
                    iou_curr = iou_value

                # Mark the detection as true positive if IoU is above the threshold
                for j, curr_iou in enumerate(iou_levels):
                    if iou_value > curr_iou:
                        speeds[i] = speed
                        #print(speed)
                        if gt_cat == category_id:
                            tp[i,j] = True
                            if iou_value > .5:
                                tp_speed[i] = True

            pbar.update(1)
    return tp, tp_speed, conf, pred_cls, target_cls, names, speeds

def ap_per_class(tp,
                 conf,
                 pred_cls,
                 target_cls,
                 plot=True,
                 on_plot=None,
                 save_dir='./',
                 names=(),
                 eps=1e-16,
                 prefix=''):
    """
    Computes the average precision per class for object detection evaluation.

    Args:
        tp (np.ndarray): Binary array indicating whether the detection is correct (True) or not (False).
        conf (np.ndarray): Array of confidence scores of the detections.
        pred_cls (np.ndarray): Array of predicted classes of the detections.
        target_cls (np.ndarray): Array of true classes of the detections.
        plot (bool, optional): Whether to plot PR curves or not. Defaults to False.
        on_plot (func, optional): A callback to pass plots path and data when they are rendered. Defaults to None.
        save_dir (Path, optional): Directory to save the PR curves. Defaults to an empty path.
        names (tuple, optional): Tuple of class names to plot PR curves. Defaults to an empty tuple.
        eps (float, optional): A small value to avoid division by zero. Defaults to 1e-16.
        prefix (str, optional): A prefix string for saving the plot files. Defaults to an empty string.

    Returns:
        (tuple): A tuple of six arrays and one array of unique classes, where:
            tp (np.ndarray): True positive counts for each class.
            fp (np.ndarray): False positive counts for each class.
            p (np.ndarray): Precision values at each confidence threshold.
            r (np.ndarray): Recall values at each confidence threshold.
            f1 (np.ndarray): F1-score values at each confidence threshold.
            ap (np.ndarray): Average precision for each class at different IoU thresholds.
            unique_classes (np.ndarray): An array of unique classes that have data.

    """

    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes, nt = np.unique(target_cls, return_counts=True)
    nc = unique_classes.shape[0]  # number of classes, number of detections

    # Create Precision-Recall curve and compute AP for each class
    px, py = np.linspace(0, 1, 1000), []  # for plotting
    ap, p, r = np.zeros((nc, tp.shape[1])), np.zeros((nc, 1000)), np.zeros((nc, 1000))
    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        n_l = nt[ci]  # number of labels
        n_p = i.sum()  # number of predictions
        if n_p == 0 or n_l == 0:
            continue

        # Accumulate FPs and TPs
        fpc = (1 - tp[i]).cumsum(0)
        tpc = tp[i].cumsum(0)

        # Recall
        recall = tpc / (n_l + eps)  # recall curve
        r[ci] = np.interp(-px, -conf[i], recall[:, 0], left=0)  # negative x, xp because xp decreases

        # Precision
        precision = tpc / (tpc + fpc)  # precision curve
        valid_px = px[px <= max(conf[i])]
        #p[ci] = np.interp(-px, -conf[i], precision[:, 0], left=1)  # p at pr_score
        p[ci] = np.full(px.shape, np.nan)  # Initialize precision with NaN
        p[ci][:len(valid_px)] = np.interp(-valid_px, -conf[i], precision[:, 0], left=1)

        # AP from recall-precision curve
        for j in range(tp.shape[1]):
            ap[ci, j], mpre, mrec = compute_ap(recall[:, j], precision[:, j])
            if plot and j == 0:
                py.append(np.interp(px, mrec, mpre))  # precision at mAP@0.5

    py = np.array(py)
    f1 = 2 * p * r / (p + r + eps)
    names = [v for k, v in names.items() if k in unique_classes]  # list: only classes that have data
    names = dict(enumerate(names))  # to dict
    if plot:
        plot_pr_curve(px, py, ap, f'{prefix}PR_curve.png', names, on_plot=on_plot)
        plot_mc_curve(px, f1, f"{prefix}F1_curve.png", names, ylabel="F1", on_plot=on_plot)
        plot_mc_curve(px, p, f"{prefix}P_curve.png", names, ylabel="Precision", on_plot=on_plot)
        plot_mc_curve(px, r, f"{prefix}R_curve.png", names, ylabel="Recall", on_plot=on_plot)

def compute_ap(recall, precision):
    """
    Compute the average precision (AP) given the recall and precision curves.

    Args:
        recall (list): The recall curve.
        precision (list): The precision curve.

    Returns:
        (float): Average precision.
        (np.ndarray): Precision envelope curve.
        (np.ndarray): Modified recall curve with sentinel values added at the beginning and end.
    """

    # Append sentinel values to beginning and end
    mrec = recall#np.concatenate(([0.0], recall))
    mpre = precision#np.concatenate(([1.0], precision))

    # Compute the precision envelope
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

    # Integrate area under curve
    method = 'interp'  # methods: 'continuous', 'interp'
    if method == 'interp':
        x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
        ap = np.trapezoid(np.interp(x, mrec, mpre), x)  # integrate
    else:  # 'continuous'
        i = np.where(mrec[1:] != mrec[:-1])[0]  # points where x-axis (recall) changes
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # area under curve

    return ap, mpre, mrec

def plot_pr_curve(px, py, ap, save_dir='./pr_curve.png', names=(), on_plot=None):
    """Plots a precision-recall curve."""
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)
    py = np.stack(py, axis=1)

    if 0 < len(names) < 21:  # display per-class legend if < 21 classes
        for i, y in enumerate(py.T):
            indices = np.unique(y, return_index=True)[1]
            y_trunc = [y[index] for index in sorted(indices)]
            ax.plot(px[:len(y_trunc)], y_trunc, linewidth=1, label=f'{names[i]} {ap[i, 0]:.3f}')  # plot(recall, precision)
            #print(f'{names[i]}',y_trunc)
    else:
        ax.plot(px, py, linewidth=1, color='grey')  # plot(recall, precision)


    ax.plot(px, py.mean(1), linewidth=3, color='blue', label='all classes %.3f mAP@0.5' % ap[:, 0].mean())
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(bbox_to_anchor=(1.04, 1), loc='upper left')
    ax.set_title('Precision-Recall Curve')
    fig.savefig(save_dir, dpi=250)
    plt.close(fig)
    if on_plot:
        on_plot(save_dir)



def plot_mc_curve(px, py, save_dir='./mc_curve.png', names={}, xlabel="Confidence", ylabel="Metric", on_plot=None):
    """Plots a metric-confidence curve."""
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)

    if 0 < len(names) < 21:  # display per-class legend if < 21 classes
        for i, y in enumerate(py):
            ax.plot(px, y, linewidth=1, label=f"{names[i]}")  # plot(confidence, metric)
    else:
        ax.plot(px, py.T, linewidth=1, color="grey")  # plot(confidence, metric)

    py[np.isnan(py)] = 1
    y = smooth(py.mean(0), 0.05)
    ax.plot(px, y, linewidth=3, color="blue", label=f"all classes {y.max():.2f} at {px[y.argmax()]:.3f}")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    ax.set_title(f"{ylabel}-Confidence Curve")
    fig.savefig(save_dir, dpi=250)
    plt.close(fig)
    if on_plot:
        on_plot(save_dir)

def smooth(y, f=0.05):
    """Box filter of fraction f."""
    nf = round(len(y) * f * 2) // 2 + 1  # number of filter elements (must be odd)
    p = np.ones(nf // 2)  # ones padding
    yp = np.concatenate((p * y[0], y, p * y[-1]), 0)  # y padded
    return np.convolve(yp, np.ones(nf) / nf, mode="valid")  # y-smoothed

def iou(dts, gts, pyiscrowd):
    dts = np.asarray(dts)
    gts = np.asarray(gts)
    pyiscrowd = np.asarray(pyiscrowd)
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
            if pyiscrowd[j]:
                unionarea = darea
            else:
                unionarea = darea + garea - t
            ious[i, j] = float(t) / unionarea
    return ious


def speed_precision(tp, speeds, tp2=None, speeds2=None):
    # Define the custom bin edges
    bin_edges = [0, 0.5, 2, 5, np.inf]

    ap_per_bin_1 = []
    ap_per_bin_2 = []

    # Iterate over each bin and calculate average precision for both models
    for i in range(len(bin_edges) - 1):
        start_edge = bin_edges[i]
        end_edge = bin_edges[i + 1]

        # Create masks for the current bin for both models
        bin_mask_1 = (speeds >= start_edge) & (speeds < end_edge)
        bin_mask_2 = (speeds2 >= start_edge) & (speeds2 < end_edge) if tp2 is not None else None

        # Get the true positives for the current bin for both models
        bin_tp_1 = tp[bin_mask_1]
        bin_tp_2 = tp2[bin_mask_2] if bin_mask_2 is not None else []

        print('Swin: ' + str(len(bin_tp_1)))
        print('Video Swin: ' + str(len(bin_tp_2)))

        # Calculate precision for the current bin for both models
        precision_1 = np.sum(bin_tp_1) / len(bin_tp_1) if len(bin_tp_1) > 0 else 0
        precision_2 = np.sum(bin_tp_2) / len(bin_tp_2) if len(bin_tp_2) > 0 else 0

        ap_per_bin_1.append(precision_1)
        ap_per_bin_2.append(precision_2)

    # Format the bin labels
    bin_labels = [f'{bin_edges[i]}-{bin_edges[i + 1]}' for i in range(len(bin_edges) - 1)]

    # Plot bar chart
    width = 0.35  # Bar width for side-by-side comparison
    x = np.arange(len(bin_labels))  # X locations for the bins

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width / 2, ap_per_bin_1, width, label='SF')
    rects2 = ax.bar(x + width / 2, ap_per_bin_2, width, label='Enc5x1')



    # Add some text for labels, title, and custom x-axis tick labels
    ax.set_xlabel('Speed Ranges (px per frame)')
    ax.set_ylabel('Average Precision')
    ax.set_title('Comparison of Average Precision at IoU 0.5 for Different Speeds')
    ax.set_xticks(x)
    ax.set_xticklabels(bin_labels)
    ax.legend()

    add_labels(rects1, ax)
    add_labels(rects2, ax)

    plt.show()

def add_labels(rects, ax):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

# Add labels on top of the bars for both models

gt_json_file = 'speed_test.json'
detections_json_file = 'visdrone_new/val/prediction_swin.json'
detections_json_file_video = 'visdrone_new/val/prediction_videoswin_enc5x1.json'

tp, tp_speed, conf, pred_cls, target_cls, names, speeds = create_function_inputs(gt_json_file, detections_json_file)
print('starting ap per class...')
ap_per_class(tp, conf, pred_cls, target_cls, names=names)

print('speed plot...')
_, tp_speed2, _, _, _, _, speeds2 = create_function_inputs(gt_json_file, detections_json_file_video)
speed_precision(tp_speed, speeds, tp_speed2, speeds2)
print('done')