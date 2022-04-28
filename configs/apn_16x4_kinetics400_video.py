_base_ = ["./Adam.py", "./default_runtime.py"]

# model settings
model = dict(
    type='APN',
    backbone=dict(type='X3D', gamma_w=1, gamma_b=2.25, gamma_d=2.2),
    cls_head=dict(
        type='APNHead',
        in_channels=432,
        spatial_type='avg3d'))

# dataset settings
dataset_type = 'CustomVideoDataset'
data_root = 'my_data/kinetics400/videos_train'
data_root_val = 'my_data/kinetics400/videos_val'
ann_file_train = 'my_data/kinetics400/kinetics400_train_list_videos.txt'
ann_file_val = 'my_data/kinetics400/kinetics400_val_list_videos.txt'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)
train_pipeline = [
    dict(type='DecordInit'),
    dict(type='SampleFrames', clip_len=16, frame_interval=4),
    dict(type='DecordDecode'),
    dict(type='ProgLabel', ordinal=True),
    dict(type='Resize', scale=(-1, 128)),
    dict(type='RandomResizedCrop'),
    dict(type='Resize', scale=(112, 112), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'prog_labels'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'prog_labels'])
]
val_pipeline = [
    dict(type='DecordInit'),
    dict(type='SampleFrames', clip_len=16, frame_interval=4),
    dict(type='DecordDecode'),
    dict(type='ProgLabel'),
    dict(type='Resize', scale=(-1, 128)),
    dict(type='CenterCrop', crop_size=112),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'prog_labels'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'prog_labels'])
]
data = dict(
    videos_per_gpu=32,
    workers_per_gpu=2,
    test_dataloader=dict(videos_per_gpu=1),
    train=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=data_root,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=data_root_val,
        pipeline=val_pipeline))

# evaluation
evaluation = dict(interval=1, metrics=['MAE'], save_best='MAE', rule='less')

# work_dir
work_dir = './work_dirs/apn_16x4_kinetics400_video/'
