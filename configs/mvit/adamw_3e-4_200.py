# optimizer
optimizer = dict(type='AdamW', lr=1.6e-3, weight_decay=0.05)  # this lr is for batch-size=64
optimizer_config = dict(grad_clip=dict(max_norm=1.0))
# learning policy
lr_config = dict(policy='CosineAnnealing',
                 min_lr_ratio=0.01,
                 warmup='linear',
                 warmup_ratio=0.01,
                 warmup_iters=30,
                 warmup_by_epoch=False)
total_epochs = 200
