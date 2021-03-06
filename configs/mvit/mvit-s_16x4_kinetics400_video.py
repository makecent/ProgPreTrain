_base_ = ["adamw_3e-4_200.py",
          "default_runtime.py",
          "kinetics400.py"]

# model settings
model = dict(
    type='Recognizer3D',
    backbone=dict(type='MViT', arch='small_16x4', pretrained=False),
    cls_head=dict(type='MViTHead', num_classes=400, in_channels=512, label_smooth_eps=0.1),
    train_cfg=dict(blending=dict(type='BatchAugBlending')),
    # train_cfg=None,
    test_cfg=dict(average_clips='prob'))

optimizer = dict(lr=1.6e-3 * 16*8*2/512)  # 1.6e-3 x mini_batch x num_gpus x num_blending / 512
log_config = dict(interval=1000)
# lr settings
data = dict(videos_per_gpu=16)
