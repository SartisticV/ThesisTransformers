import cv2
import os
from tqdm import tqdm

folder = '/scratch/6409458/data/visdrone_new/val/data'

videos = os.listdir(folder)

for vid in tqdm(videos):
    frames = os.listdir(os.path.join(folder, vid))
    for f in frames:
        f1 = os.path.join(folder, vid, f)

        img1 = cv2.imread(f1)
        img2 = cv2.imread(f1)

        cv2.putText(img1,
                    'VIDEO SWIN',
                    (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 255, 255),
                    2,
                    cv2.LINE_4)

        cv2.putText(img2,
                    'SWIN',
                    (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 255, 255),
                    2,
                    cv2.LINE_4)

        combined_frame = cv2.vconcat([img1, img2])

        folder_new = os.path.join('./comp_visdrone', vid)

        if not os.path.exists(folder_new):
            os.mkdir(folder_new)

        cv2.imwrite(os.path.join(folder_new, f), combined_frame)

