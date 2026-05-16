# dataset settings for dataset1027 foreground (1-class) instance segmentation

dataset_type = 'CocoDataset'
classes = ('foreground',)
data_root = 'data/dataset1027_tentative_combined/'
ann_root = 'data/dataset1027_tentative_combined_processed/annotations/'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
    train=dict(
        type=dataset_type,
        classes=classes,
        ann_file=ann_root + 'annotations_train.json',
        img_prefix=data_root,
        pipeline=train_pipeline,
        filter_empty_gt=False),
    val=dict(
        type=dataset_type,
        classes=classes,
        ann_file=ann_root + 'annotations_val.json',
        img_prefix=data_root,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        classes=classes,
        ann_file=ann_root + 'annotations_val.json',
        img_prefix=data_root,
        pipeline=test_pipeline))

evaluation = dict(interval=1, metric=['bbox', 'segm'])
