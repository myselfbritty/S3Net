S3NET: From Forks to Forceps: A New Architecture forInstance Segmentation of Surgical Instruments
===========================================

Overview
--------
Computer-assisted medical surgeries require accurate instance segmentation of surgical instruments in the endoscopic camera view to carry out many downstream perception and control tasks.
We observe that, due to the typical orientation and aspect ratio of medical instruments, the cross-domain fine-tuning of the instance segmentation model detects and segments the object regions correctly but is insufficient to classify the segmented regions accurately. 
We propose using cumulative IoU over the entire test dataset as the evaluation metric. Using cumulative IoU provides many insights about the low performance of state-of-the-art instance segmentation techniques on the dataset.
We propose a novel three-stage deep neural network architecture to augment a third stage in a standard instance segmentation pipeline to perform mask-based classification of the segmented object. To handle small datasets with visually similar classes, we train the proposed third stage using ideas from metric learning.


Data
----
EndoVis 2017 consisting of 8 X 225-frame sequences is used as train set and 2 X 300-frame sequences is used as test set.
Instrument labels are 
Bipolar Forceps 
Prograsp Forceps
Large Needle Driver
Vessel Sealer
Grasping Retractor
Monopolar Curved Scissors 
Ultrasound Probe

EndoVis 2018 consisting of 11 X 149-frame sequences is used as train set and 4 X 149-frame sequences is used as test set.
Instrument labels are 
Bipolar Forceps 
Prograsp Forceps
Large Needle Driver
Monopolar Curved Scissors 
Ultrasound Probe
Suction Instrument
Clip Applier

EETS consisting of 20 X 125-frame sequences is used as train set and 10 X 125-frame sequences is used as test set.
Instrument labels are 
Suction
Irrigation
Dissector
Scissors
Knife
Navigation
Biopsy
Curette
Drill
Tumor_biopsy

Method
------
S3NET
The basic S3NET pipeline is divided into two:
1) Stage_1_2
3) Stage 3

These two are set on different environments:

Installation: Stage_1_2
------------
To install you can run
(Tested on Ubuntu 16.04. For Ubuntu 18 and 20, install gcc 9)

* conda create -n S3NET_Stage_1_2 python=3.7
* conda activate S3NET_Stage_1_2
* conda install pytorch==1.1.0 torchvision==0.3.0 cudatoolkit=9.2 -c pytorch
* conda install cython
* pip install "git+https://github.com/philferriere/cocoapi.git#egg=pycocotools&subdirectory=PythonAPI"
* pip install mmcv==0.2.14
* pip install tqdm
* pip install opencv-python
* pip install scikit-image
* git clone https://bitbucket.org/anonymouscvprs3net/s3net_3stage S3NET
* cd S3NET
* bash compile.sh
* python setup.py install
* pip install .

Installation: Stage 3
------------
To install you can run
(Tested on Ubuntu 16.04. For Ubuntu 18 and 20, install gcc 9)
Dependecies:
Nvidia Driver >= 460

* conda create -n S3NET_Stage_3 python=3.7
* conda activate S3NET_Stage_3
* conda install cython
* conda install -c anaconda tensorflow-gpu=2.4.1
* pip install "git+https://github.com/philferriere/cocoapi.git#egg=pycocotools&subdirectory=PythonAPI"
* pip install mmcv==0.2.14
* pip install tqdm
* pip install opencv-python
* pip install sklearn
* pip install matplotlib


Organize data
----------

Here we describe the steps for using the Endoscopic Vision 2017 [1] for instrument-type segmentation.

Download the 2017 dataset from here [2]. Arrange the data in the folder format

::

    ├── data
    │   ├── EndoVis2018
    │   	├── train
    │   		├── annotations
    │   		├── binary_annotations
    │   		├── coco-annotations
    │   		├── images
    		├── val
    │   		├── annotations
    │   		├── binary_annotations
    │   		├── coco-annotations
    │   		├── images
- The pre-trained weights of all stages are available at [Google drive](https://drive.google.com/drive/folders/1k7WxHMq60CkMneHb6e8lzGY4RUxFlZfW?usp=sharing)

Stage 1_2
------------------------------

- Organize the data of the dataset into the appropriate splits for coco format.

``python organize2017.py --data-dir /path/to/raw_train_data/ \
--save-dir /path/to/save/organized/data/ --cropped``

``python organize2017_test.py --data_dir /path/to/raw_test_data/\
--save_dir /path/to/save/data/EndoVis2017/ --cropped``

Convert the dataset to the MS-COCO format. Required for Mask R-CNN transfer learning.

``python prepare_data/convert_to_coco_foldwise.py --root_dir /path/to/organized/data/
    --dataset <dataset_name> --fold_name <dataset_split>``
    
``python prepare_data/convert_to_coco_test.py --test_dir /path/to/data/EndoVis2017/test_crop/
    --dataset <dataset_name>``

mkdir at data/EndoVis2017/train

mkdir at data/EndoVis2017/test

- Copy all images from /path/to/organized/data/fold#/images to /path/to/data/EndoVis2017/train

- Copy all images from /path/to/EndoVis2017/test_crop/images to /path/to/data/EndoVis2017/test

- The pre-trained weights are in [Google drive](https://drive.google.com/drive/folders/1k7WxHMq60CkMneHb6e8lzGY4RUxFlZfW?usp=sharing)

- Kindly download them and add to pre-trained-weights folder.

Training
---------------

Resized weights of pre-trained MaskRCNN (ImageNet) are in pre-trained-weights folder for training afresh. 

``python training_routine.py``

Testing
----------
Run the testing by 

``python testing_routine.py``


At this point, Stage 1 and 2 is over and we need to now improve the classification of the instances generated.

Stage 3
----------

Training
---------------
Resized weights are in pre-trained-weights folder
``python preprocessing.py''
``python train_mask_classifier.py``

Testing
----------
Run the testing by 

``python test_mask.py``



Evaluation
----------
Organize the data of the dataset into the appropriate splits for final evaluation of combined dataset

``python prepare_data/prepare_data_for_evaluation.py`` for four-fold cross validation 

``python prepare_data/prepare_test_data_for_evaluation.py`` for the test data preparation
