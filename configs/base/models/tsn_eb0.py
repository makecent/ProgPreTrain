# model settings
model = dict(
    type='Recognizer2D',
    backbone=dict(
        type='torchvision.efficientnet_b0',
        pretrained=True),
    cls_head=dict(
        type='TSNHead',
        num_classes=400,
        in_channels=1280,
        spatial_type=None,
        consensus=dict(type='AvgConsensus', dim=1),
        dropout_ratio=0.4,
        init_std=0.01),
    # model training and testing settings
    train_cfg=None,
    test_cfg=dict(average_clips=None))

