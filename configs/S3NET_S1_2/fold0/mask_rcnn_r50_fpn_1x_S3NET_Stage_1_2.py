# model settings
model = dict(
    type='MaskRCNN',
    pretrained='modelzoo://resnet50',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        style='pytorch'),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_scales=[8],
        anchor_ratios=[0.5, 1.0, 2.0],
        anchor_strides=[4, 8, 16, 32, 64],
        target_means=[.0, .0, .0, .0],
        target_stds=[1.0, 1.0, 1.0, 1.0],
        use_sigmoid_cls=True),
    bbox_roi_extractor=dict(
        type='SingleRoIExtractor',
        roi_layer=dict(type='RoIAlign', out_size=7, sample_num=2),
        out_channels=256,
        featmap_strides=[4, 8, 16, 32]),
    bbox_head=dict(
        type='SharedFCBBoxHead',
        num_fcs=2,
        in_channels=256,
        fc_out_channels=1024,
        roi_feat_size=7,
        num_classes=8,     ######Class number = 8
        target_means=[0., 0., 0., 0.],
        target_stds=[0.1, 0.1, 0.2, 0.2],
        reg_class_agnostic=False),
    mask_roi_extractor=dict(
        type='SingleRoIExtractor',
        roi_layer=dict(type='RoIAlign', out_size=14, sample_num=2),
        out_channels=256,
        featmap_strides=[4, 8, 16, 32]),
    mask_head=dict(
        type='FCNMaskHead',
        num_convs=4,
        in_channels=256,
        conv_out_channels=256,
        num_classes=8))    ######Class number = 8
# model training and testing settings
train_cfg = dict(
    rpn=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.7,
            neg_iou_thr=0.3,
            min_pos_iou=0.3,
            ignore_iof_thr=-1),
        sampler=dict(
            type='RandomSampler',
            num=256,
            pos_fraction=0.5,
            neg_pos_ub=-1,
            add_gt_as_proposals=False),
        allowed_border=0,
        pos_weight=-1,
        smoothl1_beta=1 / 9.0,
        debug=False),
    rcnn=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.5,
            min_pos_iou=0.5,
            ignore_iof_thr=-1),
        sampler=dict(
            type='RandomSampler',
            num=512,
            pos_fraction=0.25,
            neg_pos_ub=-1,
            add_gt_as_proposals=True),
        mask_size=28,
        pos_weight=-1,
        debug=False))
test_cfg = dict(
    rpn=dict(
        nms_across_levels=False,
        nms_pre=2000,
        nms_post=2000,
        max_num=2000,
        nms_thr=0.7,
        min_bbox_size=0),
    rcnn=dict(
        score_thr=0.0,    #####Threshold set to zero to include all bboxes
        nms=dict(type='nms', iou_thr=0.5),
        max_per_img=5,   ######Total instances per image set to 5 for EndoVis_2017
        mask_thr_binary=0.5))
# dataset settings
dataset_type = 'CocoDataset'
data_root = 'data/EndoVis_2017/'  #####Path to the root folder 
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
data = dict(
    imgs_per_gpu=4, 
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'Organized/fold0/coco-annotations/instances_train_sub.json', #######Annotation file coco format for training dataset
        img_prefix=data_root +'train/images/',  #######Images for training dataset
        img_scale=(1333, 800),
        img_norm_cfg=img_norm_cfg,
        size_divisor=32,
        flip_ratio=0.5,
        with_mask=True,
        with_crowd=True,
        with_label=True),
    val=dict(
        type=dataset_type,
        ####For testing foldwise
        ann_file=data_root + 'Organized/fold0/coco-annotations/instances_val_sub.json', #######Annotation file coco format for testing dataset
        img_prefix=data_root +'train/images/', #######Images for train dataset
        # ####For testing seq 9 10
        # ann_file=data_root + 'test_crop/coco-annotations/instances_test_sub.json', #######Annotation file coco format for testing dataset
        # img_prefix=data_root +'test/images/', #######Images for testing dataset
        img_scale=(1333, 800),
        img_norm_cfg=img_norm_cfg,
        size_divisor=32,
        flip_ratio=0,
        with_mask=True,
        with_crowd=True,
        with_label=True),
    test=dict(
        type=dataset_type,
        ####For testing foldwise
        ann_file=data_root + 'Organized/fold0/coco-annotations/instances_val_sub.json', #######Annotation file coco format for testing dataset
        img_prefix=data_root +'train/images/', #######Images for train dataset
        ####For testing seq 9 10
        # ann_file=data_root + 'test_crop/coco-annotations/instances_test_sub.json', #######Annotation file coco format for testing dataset
        # img_prefix=data_root +'test/images/', #######Images for testing dataset
        img_scale=(1333, 800),
        img_norm_cfg=img_norm_cfg,
        size_divisor=32,
        flip_ratio=0,
        with_mask=False,
        with_label=False,
        test_mode=True))
# optimizer
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[8, 11])
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])

# runtime settings
total_epochs = 12
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/S3NET_Stage_1_2/fold0/' #####Path to the ouput folder
######Use load_from for training the new dataset 
load_from ='pre-trained-weights/basic_chkpoints/resized_weights_r50_class_8.pth' #####ImageNet weights resized to class 8 
#####[0:Background, 1:Bipolar Forceps, 2:Prograsp Forceps, 3:Large Needle Driver, 4:Vessel Sealer, 5:Grasping Retractor, 6:Monopolar Curved Scissors, 7:Others/Ultrasound Probe]
######Use resume_from for training from a different epoch number/testing the dataset 
#resume_from = './work_dirs/pre-trained-weights/EndoVis_2017/fold0/epoch_12.pth'  ####Pre-trained weights for testing
resume_from = None
workflow = [('train', 1)]
