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


EndoVis 2018 consisting of 11 X 149-frame sequences is used as train set and 4 X 149-frame sequences is used as test set.
Instrument labels are 
Bipolar Forceps 
Prograsp Forceps
Large Needle Driver
Monopolar Curved Scissors 
Ultrasound Probe
Suction Instrument
Clip Applier


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

Organize the data, dataset is available in [Google drive] (https://drive.google.com/drive/folders/1pU1eWmYPJOwiaP7XKPjhZriZ1Anc-9dm?usp=share_link)

::

    ├── data
    │   ├── EndoVis2018
    │   	├── train
    │   		├── annotations
    │   		├── binary_annotations
    │   		├── coco-annotations
    │   		├── images
    	|	├── val
    │   		├── annotations
    │   		├── binary_annotations
    │   		├── coco-annotations
    │   		├── images

Stage 1_2
------------------------------

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
Evaluation was performed using ISINet evaluation framework as mentioned in [1].

References

[1] González, Cristina, Laura Bravo-Sánchez, and Pablo Arbelaez. "Isinet: an instance-based approach for surgical instrument segmentation." International Conference on Medical Image Computing and Computer-Assisted Intervention. Springer, Cham, 2020.
