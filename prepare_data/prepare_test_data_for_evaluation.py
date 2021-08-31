# -*- coding: utf-8 -*-
"""
Created on Mon Jul 19 01:02:55 2021

@author: dell
"""

"""
[1] Merge masks with different instruments into one binary mask
[2] Crop black borders from images and masks
"""
from pathlib import Path

from tqdm import tqdm
import cv2
import numpy as np

data_path = Path('data/EndoVis_2017/raw_data')

test_path = data_path / 'test'

cropped_test_path = data_path / 'cropped_test_2017'

original_height, original_width = 1080, 1920
height, width = 1024, 1280
h_start, w_start = 28, 320

binary_factor = 255
parts_factor = 85
instrument_factor = 32


if __name__ == '__main__':
    for instrument_index in range(9, 11):
        instrument_folder = 'instrument_dataset_' + str(instrument_index)

        (cropped_test_path / instrument_folder / 'images').mkdir(exist_ok=True, parents=True)

        binary_mask_folder = (cropped_test_path / instrument_folder / 'binary_masks')
        binary_mask_folder.mkdir(exist_ok=True, parents=True)

        parts_mask_folder = (cropped_test_path / instrument_folder / 'parts_masks')
        parts_mask_folder.mkdir(exist_ok=True, parents=True)

        instrument_mask_folder = (cropped_test_path / instrument_folder / 'instruments_masks')
        instrument_mask_folder.mkdir(exist_ok=True, parents=True)

        mask_folders = list((test_path / instrument_folder / 'ground_truth').glob('*'))

        for file_name in tqdm(list((test_path / instrument_folder / 'left_frames').glob('*'))):
            img = cv2.imread(str(file_name))
            old_h, old_w, _ = img.shape

            img = img[h_start: h_start + height, w_start: w_start + width]
            cv2.imwrite(str(cropped_test_path / instrument_folder / 'images' / (file_name.name)), img)

            mask_binary = np.zeros((old_h, old_w))
            mask_parts = np.zeros((old_h, old_w))
            mask_instruments = np.zeros((old_h, old_w))
            
            for mask_folder in mask_folders:
                if 'BinarySegmentation' in str(mask_folder):
                    mask_binary = cv2.imread(str(mask_folder / file_name.name), 0)
                ####Rename the 'TypeSegmentationRescaled' folder in the ground_truth annotations to Color  
                elif 'TypeSegmentation' in str(mask_folder):
                    mask_instruments = cv2.imread(str(mask_folder / file_name.name), 0)
                    
                elif 'PartsSegmentation' in str(mask_folder):
                    mask = cv2.imread(str(mask_folder / file_name.name), 0)
                    mask_parts[mask == 30] = 1  # Shaft
                    mask_parts[mask == 100] = 2  # Wrist
                    mask_parts[mask == 255] = 3  # Claspers


            mask_binary = (mask_binary[h_start: h_start + height, w_start: w_start + width] > 0).astype(
                np.uint8) * binary_factor
            mask_instruments = (mask_instruments[h_start: h_start + height, w_start: w_start + width]).astype(
                np.uint8) * instrument_factor
            mask_parts = (mask_parts[h_start: h_start + height, w_start: w_start + width]).astype(
                np.uint8) * parts_factor
            
            
            cv2.imwrite(str(binary_mask_folder / file_name.name), mask_binary)
            cv2.imwrite(str(instrument_mask_folder / file_name.name), mask_instruments)
            cv2.imwrite(str(parts_mask_folder / file_name.name), mask_parts)