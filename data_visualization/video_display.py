import os
import eta.core.utils as etau
import glob
import re

import fiftyone as fo
import fiftyone.utils.video as fouv
import fiftyone.core.metadata as fom
from fiftyone import ViewField as F
import json

def parse_pattern(patt):
    """Inspects the files matching the given numeric pattern and returns the
    numeric indicies of the sequence.

    Args:
        patt: a pattern with a one or more numeric sequences like
            "/path/to/frame-%05d.jpg" or `/path/to/clips/%02d-%d.mp4`

    Returns:
        a list (or list of tuples if the pattern contains multiple sequences)
            describing the numeric indices of the files matching the pattern.
            The indices are returned in alphabetical order of their
            corresponding files. If no matches were found, an empty list is
            returned
    """
    # Extract indices from exactly matching patterns
    inds = []
    for _, match, num_inds in _iter_pattern_matches(patt):
        idx = tuple(map(int, match.groups()))
        inds.append(idx[0] if num_inds == 1 else idx)

    return inds


def get_glob_matches(glob_patt):
    """Returns a list of file paths matching the given glob pattern.

    The matches are returned in sorted order.

    Args:
        glob_patt: a glob pattern like "/path/to/files-*.jpg" or
            "/path/to/files-*-*.jpg"

    Returns:
        a list of file paths that match `glob_patt`
    """
    return sorted(p.replace('\\','/') for p in glob.glob(glob_patt) if not os.path.isdir(p))

def _iter_pattern_matches(patt):
    def _glob_escape(s):
        return re.sub(r"([\*\?\[])", r"\\\1", s)

    # Use glob to extract approximate matches
    seq_exp = re.compile(r"(%[0-9]*d)")
    glob_patt = re.sub(seq_exp, "*", _glob_escape(patt))
    files = get_glob_matches(glob_patt)

    # Create validation functions
    seq_patts = re.findall(seq_exp, patt)
    fcns = [etau.parse_int_sprintf_pattern(sp) for sp in seq_patts]
    full_exp, num_inds = re.subn(seq_exp, r"(\\s*\\d+)", patt)

    # Iterate over exactly matching patterns and files
    for f in files:
        m = re.match(full_exp, f)
        if m and all(f(p) for f, p in zip(fcns, m.groups())):
            yield f, m, num_inds

FRAMES_DIR = "./comp_visdrone"

with open("./labels_offset.json", "r") as f:
    data = json.load(f)

with open("./predict_offset.json", "r") as f:
    pred_data = json.load(f)

id_to_cat = {}
id_to_cat[0] = "no object"
for c in data['categories']:
    id_to_cat[c['id']] = c['name']

# Generate some example per-frame sequence data

video_dirs = etau.list_subdirs(FRAMES_DIR, abs_paths=True)
samples = []

with fo.ProgressBar() as pb:
    for video_dir in pb(video_dirs):
        # Generate video file
        print(video_dir.split('\\')[-1])
        video_id = [d['id'] for d in data['videos'] if d['name'] == video_dir.split('\\')[-1]][0]
        frames_patt = os.path.join(video_dir, '%07d.jpg').replace('\\','/')
        frame_numbers = parse_pattern(frames_patt) #For Windows, remove etau
        video_path = video_dir + ".mp4"
        fouv.transform_video(
            frames_patt,
            video_path,
            in_opts=["-start_number", str(min(frame_numbers))],
            out_opts=["-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2"]
        )

        sample = fo.Sample(filepath=video_path)
        fom.compute_sample_metadata(sample)
        images = [d for d in data['images'] if d['video_id'] == video_id]
        pred_frames=[]


        for frame_number in frame_numbers:
            frame = fo.Frame(filepath=frames_patt % frame_number)

            imw = sample.metadata.frame_width
            imh = sample.metadata.frame_height

            frames = [d['id'] for d in images if d['frame_id'] == frame_number]
            detections = [d for d in data['annotations'] if d['image_id'] in frames]
            predictions = [d for d in pred_data if d['image_id'] in frames and d['score']>=.05]

            det = []
            pred = []
            for d in detections:
                x1 = d['bbox'][0] / imw
                x2 = d['bbox'][2] / imw
                y1 = d['bbox'][1] / imh
                y2 = d['bbox'][3] / imh
                bbox = [x1, y1, x2, y2]
                det.append(fo.Detection(bounding_box=bbox,label=id_to_cat[d['category_id']]))

            frame['frame_detections'] = fo.Detections(detections=det)
            for p in predictions:
                x1 = p['bbox'][0] / imw
                x2 = p['bbox'][2] / imw
                y1 = p['bbox'][1] / imh
                y2 = p['bbox'][3] / imh
                bbox = [x1, y1, x2, y2]

                pred.append(fo.Detection(bounding_box=bbox, label=id_to_cat[p['category_id']]))

            frame['frame_predictions'] = fo.Detections(detections=pred)

            sample.frames[frame_number] = frame

        samples.append(sample)

dataset = fo.Dataset()
dataset.add_samples(samples)


dataset.evaluate_detections(gt_field="frames.frame_detections", pred_field="frames.frame_predictions", eval_key="eval", iou=0.25)
dataset = dataset.filter_labels("frames.frame_detections", F("iscrowd")!=True, only_matches=False)

session = fo.launch_app(dataset, desktop=True)
session.wait()

"""
data_path='/data/uadetrac_ir_rf/train',
    labels_path='/data/labels_train_small',
    """