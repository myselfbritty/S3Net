#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import json
import os
from mask_classifier import make_classifier
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils
import numpy as np
import cv2
import tensorflow as tf

tf.keras.backend.set_image_data_format('channels_first')

ip_path = './S3NET_outputs/'
ip_file = 'EndoVis18_MaskRCNN_ev18_post_ep_12.pkl.segm.json'
op_path = './S3NET_outputs/S3NET_correct_class_prediction/'
annotation_file = './data/val/coco-annotations/instances_val_sub.json'

fold = './data/val'

image_path_dict = {}

model = make_classifier(num_classes=7)


@tf.function
def create_prediction(image, mask):
    op = model([image, mask], mode='softmax', training=False)
    return op


##########################################################
ip1 = np.zeros((1, 3, 224, 224))
ip2 = np.zeros((1, 25, 56, 56))

op = create_prediction(ip1, ip2)
model.load_weights('./pre-trained-weights/final_weights.h5')

############################################################


for image in os.listdir(os.path.join(fold, 'images')):
    current_image_path = os.path.join(fold, 'images', image)

    image_path_dict[image] = current_image_path

example_coco = COCO(annotation_file)
categories = example_coco.loadCats(example_coco.getCatIds())
category_names = [category['name'] for category in categories]
category_names = set([category['supercategory'] for category in categories])
category_ids = example_coco.getCatIds(catNms=['square'])

image_ids = example_coco.getImgIds(catIds=category_ids)

image_id_dict = {}

for image_id in range(len(image_ids)):
    image_data = example_coco.loadImgs(image_ids[image_id])[0]
    # print(image_data)
    current_id = image_data['id']
    current_file = image_data['file_name']
    current_path = image_path_dict[current_file]
    image_id_dict[current_id] = current_path

with open(os.path.join(ip_path, ip_file), 'r') as f:
    data = json.load(f)
data_write = []

for d in data:
    data_new = {}
    image_id = d["image_id"]
    bbox = d["bbox"]
    score = d["score"]
    category_id = d["category_id"]
    seg = d["segmentation"]
    data_new["image_id"] = image_id
    data_new["bbox"] = bbox
    data_new["score"] = score
    data_new["category_id"] = category_id

    data_new["segmentation"] = seg
    current_mask = np.ones((1, 56, 56), dtype=np.float32)
    mask = maskUtils.decode(seg)
    mask = 255 * np.ascontiguousarray(mask, dtype=np.uint8)
    mask = cv2.resize(mask, (56, 56))
    mask = mask.astype(np.float32)
    mask = mask / 255.0
    current_mask[0, :, :] = mask
    current_image = cv2.imread(image_id_dict[image_id])
    current_image = cv2.resize(current_image, (224, 224))

    current_image = current_image.copy().astype(np.float32)
    mean = [123.675, 116.28, 103.53]
    mean = np.asarray(mean)
    std = [58.395, 57.12, 57.375]
    std = np.asanyarray(std)
    mean = np.float64(mean.reshape(1, -1))
    stdinv = 1 / np.float64(std.reshape(1, -1))
    cv2.subtract(current_image, mean, current_image)
    cv2.multiply(current_image, stdinv, current_image)
    current_image = np.transpose(current_image, [2, 0, 1])
    current_image = np.expand_dims(current_image, axis=0)
    op = create_prediction(current_image, current_mask)
    new_id = np.argmax(op[0, 0, :])
    data_new["category_id"] = int(new_id + 1)
    data_write.append(data_new)

resFile = os.path.join(op_path, ip_file)
with open(resFile, 'w') as outfile:
    json.dump(data_write, outfile)
