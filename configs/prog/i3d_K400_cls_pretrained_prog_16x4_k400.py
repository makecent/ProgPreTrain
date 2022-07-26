_base_ = ["./default_runtime.py", "./kinetics400_prog_16x4x224x224_video.py"]

# model settings
model = dict(
    type='Recognizer3DWithProg',
    backbone=dict(
        type='ResNet3d',
        pretrained2d=True,
        pretrained='torchvision://resnet50',
        depth=50,
        conv1_kernel=(5, 7, 7),
        conv1_stride_t=2,
        pool1_stride_t=2,
        conv_cfg=dict(type='Conv3d'),
        norm_eval=False,
        inflate=((1, 1, 1), (1, 0, 1, 0), (1, 0, 1, 0, 1, 0), (0, 1, 0)),
        zero_init_residual=False),
    cls_head=dict(
        type='I3DHeadWithProg',
        num_classes=400,
        num_stages=100,
        in_channels=2048,
        spatial_type='avg',
        dropout_ratio=0.5,
        init_std=0.01),
    # model training and testing settings
    train_cfg=dict(aux_info=('prog_label',)),
    test_cfg=dict(average_clips='prob'))

log_config = dict(interval=2000)
data = dict(videos_per_gpu=16)

# optimizer
optimizer = dict(
    type='SGD',
    lr=5e-5,
    momentum=0.9,
    weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))
evaluation = dict(interval=1, metrics=['top_k_accuracy', 'mean_class_accuracy', 'MAE'])
# learning policy
# lr_config = dict(policy='step', step=[4, 8])
lr_config = dict(policy='Fixed')
total_epochs = 10
load_from = "work_dirs/i3d_16x4_k400/epoch_100.pth"
