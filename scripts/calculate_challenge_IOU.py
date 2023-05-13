from pathlib import Path
import argparse
import cv2
import numpy as np
from tqdm import tqdm

height, width = 1024, 1280
h_start, w_start = 28, 320
instrument_dataset_length = 10

def general_dice(y_true, y_pred):
    result = []

    if y_true.sum() == 0:
        if y_pred.sum() == 0:
            return 1
        else:
            return 0

    for instrument_id in set(y_true.flatten()):
        if instrument_id == 0:
            continue
        result += [dice(y_true == instrument_id, y_pred == instrument_id)]

    return np.mean(result)

def general_jaccard(y_true, y_pred):
    result = []

    if y_true.sum() == 0:
        if y_pred.sum() == 0:
            return 1
        else:
            return 0
    for instrument_id in set(y_true.flatten()):
        if instrument_id == 0:
            continue
        result += [jaccard(y_true == instrument_id, y_pred == instrument_id)]

    return np.mean(result)


def jaccard(y_true, y_pred):
    intersection = (y_true * y_pred).sum()
    union = y_true.sum() + y_pred.sum() - intersection
    return (intersection + 1e-15) / (union + 1e-15)


def dice(y_true, y_pred):
    return (2 * (y_true * y_pred).sum() + 1e-15) / (y_true.sum() + y_pred.sum() + 1e-15)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    arg = parser.add_argument

    arg('--targets_dir', type=str, default='data/cropped_train',
        help='path where train images with ground truth are located')
    arg('--predictions_dir', type=str, default='predictions/unet11', help='path with predictions')
    arg('--test', default=False, action='store_true', help='Bool type')
    args = parser.parse_args()
    problem_type = 'instruments'
    result_dice = []
    result_jaccard = []
    targets_path = args.targets_dir
    predictions_path = args.predictions_dir
    test = args.test
    IOU_dataset_wise_agg = {n: 0 for n in range(1, instrument_dataset_length+1)}
    Dice_dataset_wise_agg = {n: 0 for n in range(1, instrument_dataset_length+1)}
    IOU_dataset_wise = {n: [] for n in range(1, instrument_dataset_length+1)}
    Dice_dataset_wise = {n: [] for n in range(1, instrument_dataset_length+1)}

    # if problem_type == 'binary':
    #     #####set range(9,11) for test
    #     for instrument_id in tqdm(range(9, 11)):
    #         instrument_dataset_name = 'instrument_dataset_' + str(instrument_id)

    #         for file_name in (
    #                 Path(targets_path) / instrument_dataset_name / 'binary_masks').glob('*'):
    #             y_true = (cv2.imread(str(file_name), 0) > 0).astype(np.uint8)

    #             pred_file_name = (Path(predictions_path) / 'binary' / instrument_dataset_name / file_name.name)

    #             pred_image = (cv2.imread(str(pred_file_name), 0) > 255 * 0.5).astype(np.uint8)
    #             y_pred = pred_image#[h_start:h_start + height, w_start:w_start + width]

    #             result_dice += [dice(y_true, y_pred)]
    #             result_jaccard += [jaccard(y_true, y_pred)]

    # elif problem_type == 'parts':
    #     #####set range(9,11) for test
    #     for instrument_id in tqdm(range(9, 11)):
    #         instrument_dataset_name = 'instrument_dataset_' + str(instrument_id)
    #         for file_name in (
    #                 Path(targets_path) / instrument_dataset_name / 'parts_masks').glob('*'):
    #             y_true = cv2.imread(str(file_name), 0)

    #             pred_file_name = Path(predictions_path) / 'parts' / instrument_dataset_name / file_name.name

    #             y_pred = cv2.imread(str(pred_file_name), 0)#[h_start:h_start + height, w_start:w_start + width]

    #             result_dice += [general_dice(y_true, y_pred)]
    #             result_jaccard += [general_jaccard(y_true, y_pred)]
    if problem_type == 'instruments':
        #####set range(9,11) for test
        #print("test", test)
        if test == True:
            for instrument_id in tqdm(range(9, 11)):
                instrument_dataset_name = 'instrument_dataset_' + str(instrument_id)
                count = 0
                for file_name in (
                        Path(targets_path) / instrument_dataset_name / 'instruments_masks').glob('*'):
                    y_true = cv2.imread(str(file_name), 0)

                    pred_file_name = Path(predictions_path) / instrument_dataset_name / 'instruments' / file_name.name

                    y_pred = cv2.imread(str(pred_file_name), 0)#[h_start:h_start + height, w_start:w_start + width]
                    if y_pred is None:
                        y_pred = np.zeros((height, width))
                    result_dice += [general_dice(y_true, y_pred)]
                    result_jaccard += [general_jaccard(y_true, y_pred)]
                    count +=1
                    Dice_dataset_wise_agg[instrument_id] += general_dice(y_true, y_pred)
                    IOU_dataset_wise_agg[instrument_id] += general_jaccard(y_true, y_pred)
                    
                    # exit(0)          
                #print(count)
                Dice_dataset_wise[instrument_id] = [Dice_dataset_wise_agg[instrument_id]/count, count]
                IOU_dataset_wise[instrument_id] = [IOU_dataset_wise_agg[instrument_id]/count, count]
            
        else:
            for instrument_id in tqdm(range(1, 9)):
                instrument_dataset_name = 'instrument_dataset_' + str(instrument_id)
                count = 0
                for file_name in (
                        Path(targets_path) / instrument_dataset_name / 'instruments_masks').glob('*'):
                    y_true = cv2.imread(str(file_name), 0)

                    pred_file_name = Path(predictions_path) / instrument_dataset_name / 'instruments' / file_name.name

                    y_pred = cv2.imread(str(pred_file_name), 0)#[h_start:h_start + height, w_start:w_start + width]
                    if y_pred is None:
                        y_pred = np.zeros((height, width))
                    result_dice += [general_dice(y_true, y_pred)]
                    result_jaccard += [general_jaccard(y_true, y_pred)]
                    count +=1
                    Dice_dataset_wise_agg[instrument_id] += general_dice(y_true, y_pred)
                    IOU_dataset_wise_agg[instrument_id] += general_jaccard(y_true, y_pred)
                    
                    # exit(0)          
                #print(count)
                Dice_dataset_wise[instrument_id] = [Dice_dataset_wise_agg[instrument_id]/count, count]
                IOU_dataset_wise[instrument_id] = [IOU_dataset_wise_agg[instrument_id]/count, count]
            

    print('Challenge Dice as TernausNet = ', np.mean(result_dice), np.std(result_dice))
    print('Challenge IOU as TernausNet = ', np.mean(result_jaccard), np.std(result_jaccard))

    print("Dice_individual_dataset", Dice_dataset_wise)
    print("IoU_individual_dataset", IOU_dataset_wise)
    Dice_dataset_wise_cal= []
    IOU_dataset_wise_cal = []
    for i, (D_val, I_val) in enumerate(zip(Dice_dataset_wise, IOU_dataset_wise)):
        if Dice_dataset_wise[D_val] != []:
            if Dice_dataset_wise[D_val][0] != 0:
                Dice_dataset_wise_cal.append([Dice_dataset_wise[D_val][0], Dice_dataset_wise[D_val][1]])
        if IOU_dataset_wise[I_val] != []:
            if IOU_dataset_wise[I_val][0] != 0:
                IOU_dataset_wise_cal.append([IOU_dataset_wise[I_val][0], IOU_dataset_wise[I_val][1]])
    Dice_challenge_num = 0
    IOU_challenge_num = 0
    Dice_challenge_den = 0
    IOU_challenge_den = 0
    #print(IOU_dataset_wise_cal[0][0])
    for ind, (D_v, I_v) in enumerate(zip(Dice_dataset_wise_cal, IOU_dataset_wise_cal)):
        #print(ind)
        Dice_challenge_num += D_v[1]*D_v[0]
        Dice_challenge_den += D_v[1]
        IOU_challenge_num += I_v[1]*I_v[0]
        IOU_challenge_den += I_v[1]
    print("Dice_challenge", Dice_challenge_num/Dice_challenge_den)
    print("IOU_challenge", IOU_challenge_num/IOU_challenge_den)