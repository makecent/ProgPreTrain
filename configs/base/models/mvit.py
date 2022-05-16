model = dict(
    type='Recognizer3D',
    backbone=dict(type='MViT', arch='base_16x4', pretrained=False),
    cls_head=dict(type='MViTHead', num_classes=400, in_channels=768),
    train_cfg=None,
    test_cfg=dict(average_clips='prob'))
