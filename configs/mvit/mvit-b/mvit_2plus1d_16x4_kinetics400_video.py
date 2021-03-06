_base_ = ["../base/schedules/adamw_3e-4_30.py",
          "../base/default_runtime.py",
          "../base/datasets/kinetics400/video/kinetics400_16x4x1x224x224_video.py"]

# model settings
model = dict(
    type='Recognizer3D',
    backbone=dict(type='MViT2Plus1D', arch='base_16x4', pretrained=False),
    cls_head=dict(type='MViTHead', num_classes=400, in_channels=768, label_smooth_eps=0.1),
    train_cfg=dict(blending=dict(type='MixupBlending', num_classes=400, alpha=.2)),
    test_cfg=dict(average_clips='prob'))
optimizer = dict(lr=1e-3)
checkpoint_config = dict(interval=1)
evaluation = dict(interval=1, metrics=['top_k_accuracy', 'mean_class_accuracy'], by_epoch=True)
log_config = dict(interval=1000)
data = dict(videos_per_gpu=4)
