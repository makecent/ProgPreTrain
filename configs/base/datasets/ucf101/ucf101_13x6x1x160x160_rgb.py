clip_len = 13
frame_interval = 6
num_clips = 1

# dataset settings
dataset_type = 'RawframeDataset'
data_root = 'my_data/ucf101/rawframes/'
data_root_val = 'my_data/ucf101/rawframes/'
split = 1  # official train/test splits. valid numbers: 1, 2, 3
ann_file_train = f'my_data/ucf101/ucf101_train_split_{split}_rawframes.txt'
ann_file_val = f'my_data/ucf101/ucf101_val_split_{split}_rawframes.txt'
ann_file_test = f'my_data/ucf101/ucf101_val_split_{split}_rawframes.txt'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)

train_pipeline = [
    dict(type='SampleFrames', clip_len=clip_len, frame_interval=frame_interval, num_clips=num_clips),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 182)),
    dict(type='RandomResizedCrop'),
    dict(type='Resize', scale=(160, 160), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
val_pipeline = [
    dict(type='SampleFrames', clip_len=clip_len, frame_interval=frame_interval, num_clips=num_clips, test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 182)),
    dict(type='CenterCrop', crop_size=160),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
test_pipeline = [
    dict(type='SampleFrames', clip_len=clip_len, frame_interval=frame_interval, num_clips=10, test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 182)),
    dict(type='ThreeCrop', crop_size=182),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
data = dict(
    videos_per_gpu=32,
    workers_per_gpu=8,
    test_dataloader=dict(videos_per_gpu=1),
    train=dict(
        type=dataset_type,
        start_index=0,
        ann_file=ann_file_train,
        data_prefix=data_root,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        start_index=0,
        ann_file=ann_file_val,
        data_prefix=data_root_val,
        pipeline=val_pipeline),
    test=dict(
        type=dataset_type,
        start_index=0,
        ann_file=ann_file_test,
        data_prefix=data_root_val,
        pipeline=test_pipeline))
