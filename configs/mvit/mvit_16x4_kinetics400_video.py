_base_ = ["../base/schedules/adamw_3e-4_30.py",
          "../base/default_runtime.py",
          "../base/datasets/kinetics400/video/kinetics400_16x4x1x224x224_video.py"]

# model settings
model = dict(
    type='Recognizer3D',
    backbone=dict(type='MViT', arch='base_16x4', pretrained=False),
    cls_head=dict(type='MViTHead', num_classes=400, in_channels=768),
    train_cfg=None,
    test_cfg=dict(average_clips='prob'))

checkpoint_config = dict(interval=1)
evaluation = dict(interval=1, metrics=['top_k_accuracy', 'mean_class_accuracy'], by_epoch=False)
# lr settings
data = dict(videos_per_gpu=8)
# work_dir
work_dir = './work_dirs/mvit_16x4_kinetics400_video/'
