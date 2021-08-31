# !/usr/bin/env python3
# Modified from https://github.com/waspinator/pycococreator/

import argparse
import datetime
import json
import os
import re
import fnmatch
from PIL import Image
import numpy as np
from pycococreator.pycococreatortools import pycococreatortools
from tqdm import tqdm



def parse_args():
    """Parse input arguments"""
    parser = argparse.ArgumentParser(
        description="Convert robotic segmentation dataset into COCO format"
    )

    parser.add_argument(
        "--test_dir",
        dest="test_dir",
        required=True,
        help="Path to cropped test directory",
    )
    parser.add_argument(
        "--dataset",
        dest="dataset",
        required=True,
        help="Dataset name",
    )
    return parser.parse_args()


args = parse_args()
print("Called with args:")
print(args)

# setup paths to data and annotations
TEST_DIR = args.test_dir
DATASET = args.dataset

INFO = {
    "description": "Robotic Instrument Type Segmentation",
    "url": "",
    "version": "Test",
    "year": DATASET,
    "contributor": "C.Gonzalez, L. Bravo-Sanchez, P. Arbelaez",
    "date_created": datetime.datetime.utcnow().isoformat(" "),
}

LICENSES = [{"id": 1, "name": "", "url": ""}]

CATEGORIES = [
    {"id": 1, "name": "Bipolar Forceps", "supercategory": "Instrument"},
    {"id": 2, "name": "Prograsp Forceps", "supercategory": "Instrument"},
    {"id": 3, "name": "Large Needle Driver", "supercategory": "Instrument"},
    {"id": 4, "name": "Vessel Sealer", "supercategory": "Instrument"},
    {"id": 5, "name": "Grasping Retractor", "supercategory": "Instrument"},
    {
        "id": 6,
        "name": "Monopolar Curved Scissors",
        "supercategory": "Instrument",
    },
    {"id": 7, "name": "Ultrasound Probe", "supercategory": "Instrument"},
]

if DATASET == "2017":
    CATEGORIES = CATEGORIES[:7]

def filter_for_jpeg(root, files):
    file_types = ["*.jpeg", "*.jpg"]
    file_types = r"|".join([fnmatch.translate(x) for x in file_types])
    files = [os.path.join(root, f) for f in files]
    files = [f for f in files if re.match(file_types, f)]

    return files


def filter_for_png(root, files):
    file_types = ["*.png"]
    file_types = r"|".join([fnmatch.translate(x) for x in file_types])
    files = [os.path.join(root, f) for f in files]
    files = [f for f in files if re.match(file_types, f)]

    return files


def filter_for_annotations(root, files, image_filename):
    file_types = ["*.png"]
    file_types = r"|".join([fnmatch.translate(x) for x in file_types])
    basename_no_extension = os.path.splitext(os.path.basename(image_filename))[
        0
    ]
    file_name_prefix = basename_no_extension + ".*"
    files = [os.path.join(root, f) for f in files]
    files = [f for f in files if re.match(file_types, f)]
    files = [
        f
        for f in files
        if re.match(file_name_prefix, os.path.splitext(os.path.basename(f))[0])
    ]

    return files


def main():

    coco_output = {
        "info": INFO,
        "licenses": LICENSES,
        "categories": CATEGORIES,
        "images": [],
        "annotations": [],
    }

    image_id = 1
    segmentation_id = 1
    # filter for jpeg images
    for dir in sorted(os.listdir(TEST_DIR)):
        IMAGE_DIR = os.path.join(TEST_DIR,dir)+"/images/"
        for root, _, files in os.walk(IMAGE_DIR):
            image_files = filter_for_png(root, files)
            image_files.sort()  # ensure order

            # go through each image
            for image_filename in tqdm(image_files):
                image = Image.open(image_filename)
                image_info = pycococreatortools.create_image_info(
                    image_id, os.path.basename(image_filename), image.size
                )
                coco_output["images"].append(image_info)
                ANNOTATION_DIR = os.path.join(TEST_DIR,dir)+"/binary_annotations/"
                # filter for associated png annotations
                for root, _, files in os.walk(ANNOTATION_DIR):
                    annotation_files = filter_for_annotations(
                        root, files, image_filename
                    )
                    # go through each associated annotation
                    for annotation_filename in annotation_files:
                        class_id = int(re.search(r'\d(?=_inst\d.png)', annotation_filename).group())

                        category_info = {
                            "id": class_id,
                            "is_crowd": "crowd" in image_filename,
                        }
                        binary_mask = np.asarray(
                            Image.open(annotation_filename)
                        ).astype(np.uint8)

                        annotation_info = pycococreatortools.create_annotation_info(
                            segmentation_id,
                            image_id,
                            category_info,
                            binary_mask,
                            image.size,
                            tolerance=2,
                        )

                        if annotation_info is not None:
                            coco_output["annotations"].append(annotation_info)

                        segmentation_id = segmentation_id + 1

                image_id = image_id + 1

    if not os.path.exists(os.path.join(TEST_DIR, "coco-annotations")):
        os.makedirs(os.path.join(TEST_DIR, "coco-annotations"))

    with open(
        "{0}/instances_test_sub.json".format(os.path.join(TEST_DIR, "coco-annotations")), "w"
        ) as output_json_file:
            json.dump(coco_output, output_json_file, indent=4)

if __name__ == "__main__":
    main()
