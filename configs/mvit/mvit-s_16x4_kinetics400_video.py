_base_ = ["adamw_3e-4_200.py",
          "default_runtime.py",
          "kinetics400_16x4x1x224x224_video.py"]

# model settings
model = dict(
    type='Recognizer3D',
    backbone=dict(type='MViT', arch='small_16x4', pretrained=False),
    cls_head=dict(type='MViTHead', num_classes=400, in_channels=512, label_smooth_eps=0.1),
    train_cfg=dict(blending=dict(type='BatchAugBlending')),
    # train_cfg=None,
    test_cfg=dict(average_clips='prob'))

optimizer = dict(lr=1.6e-3 * 8*8*2/512)
log_config = dict(interval=1000)
# lr settings
data = dict(videos_per_gpu=8)
