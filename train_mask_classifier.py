#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from mask_classifier import make_classifier
import tensorflow as tf
import os

from pycocotools.coco import COCO
import pycocotools.mask as maskUtils
import numpy as np
from sklearn.utils import shuffle
import cv2

tf.keras.backend.set_image_data_format('channels_first')
#######################################################

batch_size = 4

num_classes = 7
image_size = 224
WEIGHT_DECAY = 1e-4

annotation_file = './data/EndoVis_2017/Organized/fold0/coco-annotations/instances_train_sub.json'
annotation_file_val = './data/EndoVis_2017/Organized/fold0/coco-annotations/instances_val_sub.json'
images_path = './data/EndoVis_2017/Organized/'
val_folds = ['fold0']
train_folds=['fold1', 'fold2', 'fold3']
num_of_epochs = 100
loss_file_path = 'mask_loss.txt'
val_loss_file_path = 'val_loss.txt'
model_save_path = './pre-trained-weights/Stage_3/final_weights_fold0.h5'
########################################################
base_model = make_model(WEIGHT_DECAY = WEIGHT_DECAY)

temp_ip1 = tf.keras.layers.Input((3,image_size,image_size))
temp_ip2 = tf.keras.layers.Input((25,image_size//4,image_size//4))
temp_feat = base_model([temp_ip1, temp_ip2])
base_model.load_weights('./pre-trained-weights/Stage_3/base_model.h5')
classifier_model = make_classifier(num_classes = num_classes)

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)

@tf.function
def train_step(image, mask, label, gt_mask):
    
    feat = base_model([image, mask])
    with tf.GradientTape() as tape:
        
        op = classifier_model(feat, mode='softmax', training = True)
        
        loss_fn = tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
        loss = loss_fn(label,op)
        loss = tf.math.multiply(loss, gt_mask)
        loss = tf.math.reduce_sum(loss)
        denominator = tf.math.reduce_sum(gt_mask)
        loss = tf.math.divide(loss, denominator)
    grads = tape.gradient(loss,classifier_model.trainable_weights)
    optimizer.apply_gradients(zip(grads,classifier_model.trainable_weights))
    return loss

@tf.function
def val_step(image, mask, label, gt_mask):
    feat = base_model([image, mask], training = False)
    op = classifier_model(feat, mode='softmax', training = False)
    loss_fn = tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
    loss = loss_fn(label,op)
    loss = tf.math.multiply(loss, gt_mask)
    loss = tf.math.reduce_sum(loss)
    denominator = tf.math.reduce_sum(gt_mask)
    loss = tf.math.divide(loss, denominator)
    
    return loss


#######################################################################################
image_path_dict = {}


for fold in train_folds:
    for image in os.listdir(os.path.join(fold,'images')):
        current_image_path = os.path.join(fold,'images',image)
        
        image_path_dict[image] = current_image_path
        
example_coco = COCO(annotation_file)
categories = example_coco.loadCats(example_coco.getCatIds())
category_names = [category['name'] for category in categories]
category_names = set([category['supercategory'] for category in categories])
category_ids = example_coco.getCatIds(catNms=['square'])



train_images = []
train_labels = []
train_mask = []
train_mask_gt = []


image_ids = example_coco.getImgIds(catIds=category_ids)
for image_id in range(len(image_ids)):
    label = np.zeros((25,7), dtype=np.float32)
    
    
    image_data = example_coco.loadImgs(image_ids[image_id])[0]
    current_path = image_path_dict[image_data['file_name']]
    current_image = cv2.imread(current_path)
    ht, wt = current_image.shape[:2] #height , width
    current_image = cv2.resize(current_image,(image_size, image_size))
    current_image = current_image.copy().astype(np.float32)
    mean=[123.675, 116.28, 103.53]
    mean = np.asarray(mean)
    std=[58.395, 57.12, 57.375]
    std = np.asanyarray(std)
    mean = np.float64(mean.reshape(1, -1))
    stdinv = 1 / np.float64(std.reshape(1, -1))
    cv2.subtract(current_image, mean, current_image)
    cv2.multiply(current_image, stdinv, current_image)
    current_image = np.transpose(current_image, [2,0,1])
    train_images.append(current_image)
    annotation_ids = example_coco.getAnnIds(imgIds=image_data['id'], catIds=category_ids, iscrowd=None)
    annotations = example_coco.loadAnns(annotation_ids)
    current_mask_gt = np.zeros((25,), dtype=np.float32)
    current_mask = np.zeros((25,image_size//4,image_size//4), dtype=np.float32)
    count = 0
    for ann in annotations:
        if ann['iscrowd'] == 0:
            label[count,ann['category_id']-1] = 1.0
            current_mask_gt[count] = 1.0
            segm = ann["segmentation"]
            ann_id = ann['id']
            rles = maskUtils.frPyObjects(segm, ht, wt)
            rle = maskUtils.merge(rles) # combined rle format for the image 
            segm_mask = maskUtils.decode(rle) # decode the rle
            segm_mask[segm_mask==1] = 255
            segm_mask = cv2.resize(segm_mask, (image_size//4,image_size//4))
            segm_mask = segm_mask.astype(np.float32)
            segm_mask /= 127.5
            segm_mask -= 1
            current_mask[count,:,:] = segm_mask
            count += 1
    train_labels.append(label)
    
    
    current_mask_gt[0] = 1.0
    train_mask.append(current_mask)
    train_mask_gt.append(current_mask_gt)


train_images = np.asarray(train_images, dtype = np.float32)
train_labels = np.asarray(train_labels, dtype = np.float32)
train_mask = np.asarray(train_mask, dtype = np.float32)
train_mask_gt = np.asarray(train_mask_gt, dtype = np.float32)
print(train_images.shape)
print(train_labels.shape)
print(train_mask.shape)
print(train_mask_gt.shape)

#######################################################################################################
image_path_dict = {}


for fold in val_folds:
    for image in os.listdir(os.path.join(fold,'images')):
        current_image_path = os.path.join(fold,'images',image)
        
        image_path_dict[image] = current_image_path
        
example_coco = COCO(annotation_file_val)
categories = example_coco.loadCats(example_coco.getCatIds())
category_names = [category['name'] for category in categories]
category_names = set([category['supercategory'] for category in categories])
category_ids = example_coco.getCatIds(catNms=['square'])



val_images = []
val_labels = []
val_mask = []
val_mask_gt = []


image_ids = example_coco.getImgIds(catIds=category_ids)
for image_id in range(len(image_ids)):
    label = np.zeros((25,7), dtype=np.float32)
    
    
    image_data = example_coco.loadImgs(image_ids[image_id])[0]
    current_path = image_path_dict[image_data['file_name']]
    current_image = cv2.imread(current_path)
    ht, wt = current_image.shape[:2] #height , width
    current_image = cv2.resize(current_image,(image_size, image_size))
    current_image = current_image.copy().astype(np.float32)
    mean=[123.675, 116.28, 103.53]
    mean = np.asarray(mean)
    std=[58.395, 57.12, 57.375]
    std = np.asanyarray(std)
    mean = np.float64(mean.reshape(1, -1))
    stdinv = 1 / np.float64(std.reshape(1, -1))
    cv2.subtract(current_image, mean, current_image)
    cv2.multiply(current_image, stdinv, current_image)
    current_image = np.transpose(current_image, [2,0,1])
    val_images.append(current_image)
    annotation_ids = example_coco.getAnnIds(imgIds=image_data['id'], catIds=category_ids, iscrowd=None)
    annotations = example_coco.loadAnns(annotation_ids)
    current_mask_gt = np.zeros((25,), dtype=np.float32)
    current_mask = np.zeros((25,image_size//4,image_size//4), dtype=np.float32)
    count = 0
    for ann in annotations:
        if ann['iscrowd'] == 0:
            label[count,ann['category_id']-1] = 1.0
            current_mask_gt[count] = 1.0
            segm = ann["segmentation"]
            ann_id = ann['id']
            rles = maskUtils.frPyObjects(segm, ht, wt)
            rle = maskUtils.merge(rles) # combined rle format for the image 
            segm_mask = maskUtils.decode(rle) # decode the rle
            segm_mask[segm_mask==1] = 255
            segm_mask = cv2.resize(segm_mask, (image_size//4,image_size//4))
            segm_mask = segm_mask.astype(np.float32)
            segm_mask /= 127.5
            segm_mask -= 1
            current_mask[count,:,:] = segm_mask
            count += 1
    val_labels.append(label)
    
    
    current_mask_gt[0] = 1.0
    val_mask.append(current_mask)
    val_mask_gt.append(current_mask_gt)


val_images = np.asarray(val_images, dtype = np.float32)
val_labels = np.asarray(val_labels, dtype = np.float32)
val_mask = np.asarray(val_mask, dtype = np.float32)
val_mask_gt = np.asarray(val_mask_gt, dtype = np.float32)
print(val_images.shape)
print(val_labels.shape)
print(val_mask.shape)
print(val_mask_gt.shape)


# train_images = np.asarray(train_images, dtype = np.float32)
# train_labels = np.asarray(train_labels, dtype = np.float32)
# train_mask = np.asarray(train_mask, dtype = np.float32)
# train_mask_gt = np.asarray(train_mask_gt, dtype = np.float32)
# print(train_images.shape)
# print(train_labels.shape)
# print(train_mask.shape)
# print(train_mask_gt.shape)


#######################################################################################################


num_of_batches = train_images.shape[0]//batch_size
num_of_val_batches = val_images.shape[0]//1
val_count = 1
# best_val_loss = 1000.0
for epoch in range(num_of_epochs):
    this_epoch_loss = 0.0
    train_images, train_labels, train_mask, train_mask_gt = shuffle(train_images, train_labels, train_mask, train_mask_gt, random_state=666)
    for i in range(num_of_batches):
        image_batch = train_images[i*batch_size:i*batch_size+batch_size,:,:,:]
        label_batch = train_labels[i*batch_size:i*batch_size+batch_size]
        mask_batch = train_mask[i*batch_size:i*batch_size+batch_size]
        mask_gt_batch = train_mask_gt[i*batch_size:i*batch_size+batch_size]

        loss = train_step(image_batch, mask_batch, label_batch, mask_gt_batch)
        print('Epoch '+str(epoch)+'\tBatch '+str(i)+'\tLoss: '+str(loss)+'\n')
        this_epoch_loss+=loss.numpy()
                
    classifier_model.save_weights(model_save_path)
    this_epoch_loss /= num_of_batches
    with open(loss_file_path,'a') as f:
        f.write(str(this_epoch_loss)+'\n')
    this_val_loss = 0.0
    for j in range(num_of_val_batches):
        image_batch = val_images[j*1:j*1+1,:,:,:]
        label_batch = val_labels[j*1:j*1+1]
        mask_batch = val_mask[j*1:j*1+1]
        mask_gt_batch = val_mask_gt[j*1:j*1+1]
        
        loss = val_step(image_batch, mask_batch, label_batch, mask_gt_batch)
        
        loss = loss.numpy()
        this_val_loss += loss
    this_val_loss /= num_of_val_batches
    print('Validation: '+str(val_count) + '\tLoss: '+str(this_val_loss))
    val_count += 1
    with open(val_loss_file_path,'a') as f:
        f.write(str(this_val_loss)+'\n')
    