import numpy as np
import os
import os.path as osp
import glob
from skimage import io
import argparse
import warnings
from tqdm import tqdm
import cv2

import pdb

warnings.filterwarnings("ignore")


def parse_args():
    """Parse input arguments"""
    parser = argparse.ArgumentParser(description="Data organization routine EndoVis 2017 dataset")
    parser.add_argument(
        "--data_dir",
        required=True,
        type=str,
        default="2017/test",
        help="path to the data",
    )
    parser.add_argument(
        "--save_dir",
        required=True,
        type=str,
        default="2017/test_cropped",
        help="path to the save the organized dataset",
    )
    return parser.parse_args()


args = parse_args()
print("Called with args:")
print(args)

h, w = 1024, 1280
h_start, w_start = 28, 320
data_dir = args.data_dir
save_dir = args.save_dir

data_folders = ["instrument_dataset_" + str(i) for i in range(9, 11)]

names_dict = {
    1: "Bipolar Forceps",
    2: "Prograsp Forceps",
    3: "Large Needle Driver",
    4: "Vessel Sealer",
    5: "Grasping Retractor",
    6: "Monopolar Curved Scissors",
    7: "Other",
}


def get_cat_id(folder_name, names_dict):
    coincidence = [
        folder_name.find(name.replace(" ", "_"))
        for name in names_dict.values()
    ]
    coincidence_idx = np.where(np.array(coincidence) > -1)[0][0]
    cat_id = coincidence_idx + 1
    return cat_id


def get_binary_mask(mask_name):
    mask = io.imread(mask_name)
    if len(mask.shape) > 2:
        mask = mask[:, :, 0]
    bw_mask = mask > 0
    return bw_mask

def get_cat_mask(mask_name, cat_id):
    mask = io.imread(mask_name)
    if len(mask.shape) > 2:
        mask = mask[:, :, 0]
    bw_mask = mask == cat_id
    return bw_mask


def crop_image(image, h_start, w_start, h, w):
    image = image[h_start : h_start + h, w_start : w_start + w]
    return image


if __name__ == "__main__":
    if not osp.exists(osp.join(save_dir, "test_crop")):
        os.makedirs(osp.join(save_dir, "test_crop"))

    for data_f in tqdm(data_folders):
        save_path = osp.join(save_dir, "test_crop", data_f)
        if not osp.exists(save_path):
            os.makedirs(osp.join(save_path,"images"))
            os.makedirs(osp.join(save_path, "annotations"))
            os.makedirs(osp.join(save_path, "binary_annotations"))
        image_names = glob.glob(
            osp.join(data_dir, data_f, "left_frames", "*.png")
        )
        data_split = data_f.split('_')
        for im_name in tqdm(image_names):
            num_inst = {n: 0 for n in range(1, len(names_dict) + 1)}
            base_name = osp.join(
                save_path,
                "folder",
                "seq" + data_split[2] + "_" + osp.basename(im_name)[:-4],
            )
            # create empty mask w/ labels
            im = io.imread(im_name)
            im = crop_image(im, h_start, w_start, h, w)
            h, w, _ = np.shape(im)
            # index mask with binary class masks
            inner_folders = ['BinarySegmentation', 'TypeSegmentation']

            mask_name = im_name.replace(
                "left_frames", osp.join("ground_truth", inner_folders[0])
            )
            
            bw_mask = get_binary_mask(mask_name)
            
            bw_mask = crop_image(bw_mask, h_start, w_start, h, w)
            mask_name = im_name.replace(
                "left_frames", osp.join("ground_truth", inner_folders[1])
            )
            img = io.imread(mask_name)
            histg = cv2.calcHist([img],[0],None,[256],[0,256]) 
            mask_instruments = crop_image(img, h_start, w_start, h, w)
            
            cat_id = 0
            for index, ct in enumerate(histg):
                if ct > 0 and index > 0:
                    cat_id = index
                    # save binary_mask
                    if bw_mask.sum() > 0:
                        if cat_id == 8:
                            cat_id = 7
                        this_inst = num_inst[cat_id]
                        num_inst[cat_id] += 1
                        bw_filename = base_name.replace(
                            "folder", "binary_annotations"
                        ) + "_class{}_inst{}.png".format(cat_id, this_inst)
                        bw_mask = get_cat_mask(mask_name, cat_id)
                        bw_mask = crop_image(bw_mask, h_start, w_start, h, w)
                        io.imsave(bw_filename, bw_mask.astype(np.uint8)*255)

            # save mask
            destination = base_name.replace("folder", "images") + ".png"
            io.imsave(destination.replace("images", "annotations"), mask_instruments.astype(np.uint8))
            # save image
            destination = base_name.replace("folder", "images") + ".png"
            io.imsave(destination, im)
