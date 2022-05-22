_base_ = ["adamw_3e-4_200.py",
          "default_runtime.py",
          "kinetics400_unregl.py"]

# model settings
model = dict(
    type='Recognizer3D',
    backbone=dict(type='MViT', arch='small_16x4', pretrained=False, head_dropout_rate=0.0, droppath_rate_block=0.0),
    cls_head=dict(type='MViTHead', num_classes=400, in_channels=512),
    train_cfg=None,
    test_cfg=dict(average_clips='prob'))

evaluation = dict(interval=1)
checkpoint_config = dict(interval=1)
# optimizer
optimizer = dict(type='AdamW', lr=1.6e-3 * 32*8/512, weight_decay=0.05)
optimizer_config = dict(grad_clip=dict(max_norm=20.0))
# learning policy
lr_config = dict(policy='CosineAnnealing',
                 min_lr_ratio=0.01,
                 warmup='linear',
                 warmup_ratio=0.01,
                 warmup_iters=5,
                 warmup_by_epoch=True)
total_epochs = 50
log_config = dict(interval=250)
data = dict(videos_per_gpu=32)
