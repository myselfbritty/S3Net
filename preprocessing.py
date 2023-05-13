import os
from pycocotools.coco import COCO
import cv2
import numpy as np
import pycocotools.mask as maskUtils
from sys import *
annotation_file = './data/EndoVis_2018/train/coco-annotations/instances_train_sub.json'
image_size = 224
images_path = './data/train'
op_path = './data/train/classifier_data'
image_path_dict = {}

for image in os.listdir(os.path.join(images_path, 'images')):
    current_image_path = os.path.join(images_path, 'images', image)
    image_path_dict[image] = current_image_path

example_coco = COCO(annotation_file)
categories = example_coco.loadCats(example_coco.getCatIds())
category_names = [category['name'] for category in categories]
category_ids = example_coco.getCatIds(catNms=['square'])

image_ids = example_coco.getImgIds(catIds=category_ids)
image_count = np.zeros((7,), dtype=np.int32)
for image_id in range(len(image_ids)):
    image_data = example_coco.loadImgs(image_ids[image_id])[0]
    current_path = image_path_dict[image_data['file_name']]
    current_image = cv2.imread(current_path)
    ht, wt = current_image.shape[:2]  # height , width
    current_image = cv2.resize(current_image, (image_size, image_size))

    annotation_ids = example_coco.getAnnIds(imgIds=image_data['id'], catIds=category_ids, iscrowd=None)
    annotations = example_coco.loadAnns(annotation_ids)
    for ann in annotations:
        if ann['iscrowd'] == 0:
            current_data = np.zeros((image_size,image_size,4), dtype=np.uint8)
            label = ann['category_id'] - 1
            segm = ann["segmentation"]
            ann_id = ann['id']
            rles = maskUtils.frPyObjects(segm, ht, wt)
            rle = maskUtils.merge(rles)  # combined rle format for the image
            segm_mask = maskUtils.decode(rle)  # decode the rle
            segm_mask[segm_mask == 1] = 255
            segm_mask = cv2.resize(segm_mask, (image_size, image_size))
            current_data[:,:,:3] = current_image
            current_data[:,:,3] = segm_mask

            current_folder = os.path.join(op_path, str(label))
            if not os.path.exists(current_folder):
                os.makedirs(current_folder)
            current_save_path = os.path.join(current_folder, str(image_count[label]))
            image_count[label] += 1
            np.save(current_save_path, current_data)

            current_image = cv2.flip(current_image,0)
            segm_mask = cv2.flip(segm_mask,0)
            current_data[:, :, :3] = current_image
            current_data[:, :, 3] = segm_mask
            current_save_path = os.path.join(current_folder, str(image_count[label]))
            image_count[label] += 1
            np.save(current_save_path, current_data)

            current_image = cv2.flip(current_image, 1)
            segm_mask = cv2.flip(segm_mask, 1)
            current_data[:, :, :3] = current_image
            current_data[:, :, 3] = segm_mask
            current_save_path = os.path.join(current_folder, str(image_count[label]))
            image_count[label] += 1
            np.save(current_save_path, current_data)

            current_image = cv2.flip(current_image, -1)
            segm_mask = cv2.flip(segm_mask, -1)
            current_data[:, :, :3] = current_image
            current_data[:, :, 3] = segm_mask
            current_save_path = os.path.join(current_folder, str(image_count[label]))
            image_count[label] += 1
            np.save(current_save_path, current_data)
