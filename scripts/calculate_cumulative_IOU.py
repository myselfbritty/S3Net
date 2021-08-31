import torch
import numpy as np
import os.path as osp
import cv2
import glob
from tqdm import tqdm
import argparse
from torch.utils.data import Dataset


def parse_args():
    """Parse input arguments"""
    parser = argparse.ArgumentParser(description="Data organization routine EndoVis 2017 dataset")
    parser.add_argument(
        "--targets_dir",
        required=True,
        type=str,
        default="data/EndoVis2017/raw_data/cropped_train_2017",   #####For four-fold evaluation
        #default="data/EndoVis2017/raw_data/cropped_test_2017",   #####For test evaluation
        help="path to the input mask",
    )
    parser.add_argument(
        "--predictions_dir",
        required=True,
        type=str,
        default="data/predictions/S3NET/instruments",
        help="path to the predicted mask",
    )
    parser.add_argument(
        "--num_classes",
        required=True,
        type=str,
        default=8,
        help="number of classes including background",
    )

    return parser.parse_args()


args = parse_args()
print("Called with args:")
print(args)

binary_factor = 255
parts_factor = 85
instrument_factor = 32

def load_image(path):
    img = cv2.imread(str(path))
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def load_prediction_mask(path, problem_type):
    if problem_type == 'binary':
        mask_folder = 'binary'
        factor = binary_factor
    elif problem_type == 'parts':
        mask_folder = 'parts'
        factor = parts_factor
    elif problem_type == 'instruments':
        factor = instrument_factor
        mask_folder = 'instruments'

    mask = cv2.imread(str(path).replace('folder', mask_folder)+ ".png", 0)
    if mask is None:
        print("File does not exist")
        mask = np.zeros((height, width))
    return (mask / factor).astype(np.uint8)

def load_target_mask(path, problem_type):
    if problem_type == 'binary':
        mask_folder = 'binary_masks'
        factor = binary_factor
    elif problem_type == 'parts':
        mask_folder = 'parts_masks'
        factor = parts_factor
    elif problem_type == 'instruments':
        factor = instrument_factor
        mask_folder = 'instruments_masks'
    mask = cv2.imread(str(path).replace('folder', mask_folder)+ ".png", 0)
    return (mask / factor).astype(np.uint8)

def calculate_confusion_matrix_from_arrays(prediction, ground_truth, nr_labels):
    replace_indices = np.vstack((
        ground_truth.flatten(),
        prediction.flatten())
    ).T
    confusion_matrix, _ = np.histogramdd(
        replace_indices,
        bins=(nr_labels, nr_labels),
        range=[(0, nr_labels), (0, nr_labels)]
    )
    confusion_matrix = confusion_matrix.astype(np.uint32)
    return confusion_matrix


def calculate_iou(confusion_matrix):
    ious = {n: -1 for n in range(1, confusion_matrix.shape[0])}
    #ignore background
    for index in range(1, confusion_matrix.shape[0]):
        true_positives = confusion_matrix[index, index]
        false_positives = confusion_matrix[1:, index].sum() - true_positives
        false_negatives = confusion_matrix[index, 1:].sum() - true_positives
        denom = true_positives + false_positives + false_negatives
        if (true_positives + false_negatives) == 0:
            continue
        else:
            iou = float(true_positives) / denom
        ious[index] = iou
    return ious


def calculate_dice(confusion_matrix):
    dices = {n: -1 for n in range(1, confusion_matrix.shape[0])}
    #ignore background
    for index in range(1, confusion_matrix.shape[0]):
        true_positives = confusion_matrix[index, index]
        false_positives = confusion_matrix[1:, index].sum() - true_positives
        false_negatives = confusion_matrix[index, 1:].sum() - true_positives
        denom = 2 * true_positives + false_positives + false_negatives
        if (true_positives + false_negatives) == 0:
            continue
        else:
            dice = 2 * float(true_positives) / denom
        dices[index]= dice
    return dices

problem_type = 'instruments'
height, width = 1024, 1280
h_start, w_start = 28, 320
num_classes = args.num_classes
targets_path = args.targets_dir
predictions_path = args.predictions_dir
confusion_matrix_cum = np.zeros(
            (int(num_classes), int(num_classes)), dtype=np.uint32)
if __name__ == '__main__':
    #####set range(9,11) for test
    for instrument_index in range(1, 9):
        instrument_folder = 'instrument_dataset_' + str(instrument_index)
        image_names = glob.glob(
            osp.join(targets_path, instrument_folder, "images", "*.png")
        )
        print("image names", image_names)
        confusion_matrix = np.zeros(
            (int(num_classes), int(num_classes)), dtype=np.uint32)

        for file_name in tqdm(image_names):
            target_file_name = osp.join(
                targets_path,
                instrument_folder, "folder", osp.basename(file_name)[:-4],
            )
            target_mask = load_target_mask(target_file_name, problem_type)
            
            prediction_file_name = osp.join(
                predictions_path, instrument_folder, 
                "folder", 
                osp.basename(file_name)[:-4],
            )
            #print("prediction_file_name", prediction_file_name)
            prediction_mask = load_prediction_mask(prediction_file_name, problem_type)
            #mask_binary = np.zeros((height, width))
            output_classes = prediction_mask#[h_start:h_start + height, w_start:w_start + width]
            target_classes = target_mask
            confusion_matrix += calculate_confusion_matrix_from_arrays(
                output_classes, target_classes, int(num_classes))

        confusion_matrix_cum += confusion_matrix
print("Final_Confusion_Matrix", confusion_matrix_cum)
ious_cum = calculate_iou(confusion_matrix_cum)
dices_cum = calculate_dice(confusion_matrix_cum)
IOU = []
DICE = []
for index in range(1, len(ious_cum)+1):
    val = ious_cum[index]
    if val != 0.0 and val != -1: #Represents that instrument is not in the set
        IOU.append(val)
for index in range(1, len(dices_cum)+1):
    val = dices_cum[index]
    if val != 0.0 and val != -1: #Represents that instrument is not in the set
        DICE.append(val)   
print("ious_cumulative_classwise", ious_cum)
print("dices_cumulative_classwise", dices_cum)

print('Cumulative IOU = ',  np.mean(IOU), np.std(IOU))
print('Cumulative Dice = ', np.mean(DICE), np.std(DICE))
