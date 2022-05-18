_base_ = ["../base/schedules/adamw_3e-4_30.py",
          "default_runtime.py",
          "../base/datasets/kinetics400/video/kinetics400_16x4x1x224x224_video.py"]

# model settings
model = dict(
    type='Recognizer3D',
    backbone=dict(type='MViT', arch='small_16x4', pretrained=False),
    cls_head=dict(type='MViTHead', num_classes=400, in_channels=512),
    train_cfg=dict(blending=dict(type='MixupBlending', num_classes=400, alpha=.8)),
    test_cfg=dict(average_clips='prob'))

optimizer = dict(lr=1e-2)
log_config = dict(interval=1000)
# lr settings
data = dict(videos_per_gpu=8)
