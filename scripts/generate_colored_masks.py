import torch
import numpy as np
import os.path as osp
import cv2
import glob
from tqdm import tqdm
import argparse
import os

parser = argparse.ArgumentParser(description="Validation images annotated with predicted mask")
parser.add_argument('--images_dir',help="Path of cropped_train_2017")
parser.add_argument('--predictions_dir',help="Path of input masks")
parser.add_argument('--save_dir',help="Path of save images")
args = parser.parse_args()

binary_factor = 255
parts_factor = 85
instrument_factor = 32

def load_target_mask(path, problem_type):
    if problem_type == 'binary':
        mask_folder = 'binary_masks'
        factor = binary_factor
    elif problem_type == 'parts':
        mask_folder = 'parts_masks'
        factor = parts_factor
    elif problem_type == 'instruments':
        factor = instrument_factor
        mask_folder = 'instruments'
    mask = cv2.imread(str(path).replace('folder', mask_folder)+ ".png", 0)
    if mask is None:
        print("File not found")
        mask = np.zeros((height, width))
    return (mask / factor).astype(np.uint8)

targets_path = args.images_dir
predictions_path = args.predictions_dir
image_save_path = args.save_dir 
problem_type = 'instruments'
height, width = 1024, 1280
h_start, w_start = 28, 320
colors = [(0.0,1,0.0),(0.0,0.0,1),(1,0.0,0.0),(1,1,0.0), (0.0,1,1), (1,0,1), (0.5,0.5,0)]

def apply_mask(img, mask, alpha=0.4):
    """Apply the given mask to the image.
    """
    if mask.sum()> 0:
        for c in range(3):
            img[:, :, c] = np.where(mask ==1,
                                img[:, :, c] *
                                (1 - alpha) + alpha * colors[0][c] * 255,
                                img[:, :, c])
            img[:, :, c] = np.where(mask ==2,
                                img[:, :, c] *
                                (1 - alpha) + alpha * colors[1][c] * 255,
                                img[:, :, c])
            img[:, :, c] = np.where(mask ==3,
                                img[:, :, c] *
                                (1 - alpha) + alpha * colors[2][c] * 255,
                                img[:, :, c])
            img[:, :, c] = np.where(mask ==4,
                                img[:, :, c] *
                                (1 - alpha) + alpha * colors[3][c] * 255,
                                img[:, :, c])
            img[:, :, c] = np.where(mask ==5,
                                img[:, :, c] *
                                (1 - alpha) + alpha * colors[4][c] * 255,
                                img[:, :, c])
            img[:, :, c] = np.where(mask ==6,
                                img[:, :, c] *
                                (1 - alpha) + alpha * colors[5][c] * 255,
                                img[:, :, c])
            img[:, :, c] = np.where(mask ==7,
                                img[:, :, c] *
                                (1 - alpha) + alpha * colors[6][c] * 255,
                                img[:, :, c])
    return img

if __name__ == '__main__':
    ###Use range(9,11) for test set
    for instrument_index in range(1, 9):
        instrument_folder = 'instrument_dataset_' + str(instrument_index)
        image_names = glob.glob(
            osp.join(targets_path, instrument_folder, "images", "*.png")
        )
        if not osp.exists(image_save_path):
            os.mkdir(osp.join(image_save_path))
        if not osp.exists(osp.join(image_save_path,instrument_folder)):
            os.mkdir(osp.join(image_save_path,instrument_folder))
        for file_name in tqdm(image_names):
            target_file_name = osp.join(
                predictions_path,  
                instrument_folder, "folder", osp.basename(file_name)[:-4],
            )
            img = cv2.imread(str(file_name))
            old_h, old_w, _ = img.shape
            target_mask = load_target_mask(target_file_name, problem_type)
            #target_mask = target_mask[h_start:h_start + height, w_start:w_start + width]  ##For TernausNet
            img = apply_mask(img, target_mask, 0.4)
            cv2.imwrite(osp.join(image_save_path, instrument_folder, osp.basename(file_name)) , img)