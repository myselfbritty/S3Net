import json
import os
import cv2
import numpy as np
from pycocotools import mask as maskUtils
import glob
import argparse

def parse_args():
    """Parse input arguments"""
    parser = argparse.ArgumentParser(description="Data output organization routine MaskRCNN output")
    
    parser.add_argument(
        "--annFile",
        required=True,
        type=str,
        default="test_crop/coco-annotations/instances_test_sub.json",
        help="annotations file name",
    )
    parser.add_argument(
        "--resFile",
        required=True,
        type=str,
        default="S3NET_output/fold0/output.pkl.json",
        help="result file name",
    )
    parser.add_argument(
        "--save_dir",
        required=True,
        type=str,
        default="work_dirs/predictions",
        help="path to the predicted mask",
    )
    return parser.parse_args()

instrument_factor = 32
height, width = 1024, 1280
args = parse_args()
print("Called with args:")
print(args)
f_ann = open(args.annFile, 'r')
data_ann = json.load(f_ann)
f_output = open(args.resFile, 'r')
data = json.load(f_output)
if not os.path.exists(args.save_dir):
    os.makedirs(os.path.join(args.save_dir))
tmp = 1
mask_instruments = np.zeros((height, width))
file_name = data_ann['images'][tmp-1]['file_name']
base_name = file_name.split("_")[0].split("q")[1]
file_name = file_name.split("_")[1]
print("file_name", file_name)
path_write = os.path.join(args.save_dir,"instrument_dataset_"+base_name)
if not os.path.exists(path_write):
    os.makedirs(os.path.join(path_write))
mask_folder = os.path.join(path_write,"instruments")
if not os.path.exists(mask_folder):
    os.makedirs(os.path.join(mask_folder))
count = 0
for d in data:
    score = d["score"]
    category_id = d["category_id"]
    image_id = d["image_id"]
    seg = d["segmentation"]
    mask = maskUtils.decode(seg)
    if tmp <= len(data_ann['images']):
        if (image_id == data_ann['images'][tmp-1]['id']):
            if category_id == 1:
                mask_instruments[mask > 0] = 1
            elif category_id == 2:
                mask_instruments[mask > 0] = 2
            elif category_id == 3:
                mask_instruments[mask > 0] = 3
            elif category_id == 4:
                mask_instruments[mask > 0] = 4
            elif category_id == 5:
                mask_instruments[mask > 0] = 5
            elif category_id == 6:
                mask_instruments[mask > 0] = 6
            elif category_id == 7:
                mask_instruments[mask > 0] = 7
        elif (image_id != data_ann['images'][tmp-1]['id']):
            tmp +=1
            mask_instruments = mask_instruments.astype(
                np.uint8) * instrument_factor
            cv2.imwrite(os.path.join(mask_folder,file_name), mask_instruments)
            print("image_id", image_id)
            print("data", data_ann['images'][tmp-1]['file_name'])
            file_name = data_ann['images'][tmp-1]['file_name']
            base_name = file_name.split("_")[0].split("q")[1]
            file_name = file_name.split("_")[1]
            print("file_name", file_name)
            path_write = os.path.join(args.save_dir,"instrument_dataset_"+base_name)
            if not os.path.exists(path_write):
                os.makedirs(os.path.join(path_write))
            mask_folder = os.path.join(path_write,"instruments")
            if not os.path.exists(mask_folder):
                os.makedirs(os.path.join(mask_folder))
            if category_id == 1:
                mask_instruments[mask > 0] = 1
            elif category_id == 2:
                mask_instruments[mask > 0] = 2
            elif category_id == 3:
                mask_instruments[mask > 0] = 3
            elif category_id == 4:
                mask_instruments[mask > 0] = 4
            elif category_id == 5:
                mask_instruments[mask > 0] = 5
            elif category_id == 6:
                mask_instruments[mask > 0] = 6
            elif category_id == 7:
                mask_instruments[mask > 0] = 7

file_name = data_ann['images'][tmp-1]['file_name']
base_name = file_name.split("_")[0].split("q")[1]
file_name = file_name.split("_")[1]
mask_instruments = mask_instruments.astype(
                np.uint8) * instrument_factor
cv2.imwrite(os.path.join(mask_folder,file_name), mask_instruments)
