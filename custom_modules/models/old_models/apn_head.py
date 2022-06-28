from abc import ABCMeta

import torch
import torch.nn as nn

from mmcv.cnn import kaiming_init, normal_init, constant_init
from mmaction.models.builder import HEADS, build_loss


@HEADS.register_module()
class APNHead(nn.Module, metaclass=ABCMeta):
    """Regression head for APN.

    Args:
        num_stages (int): Number of stages to be predicted.
        in_channels (int): Number of channels in input feature.
        loss (dict): Config for building loss.
            Default: dict(type='CrossEntropyLoss')
        spatial_type (str): Pooling type in spatial dimension. Default: 'avg'.
        dropout_ratio (float): Probability of dropout layer. Default: 0.5.
        kwargs (dict, optional): Any keyword argument to be used to initialize
            the head.
    """

    def __init__(self,
                 num_classes=20,
                 num_stages=10,
                 in_channels=2048,
                 loss_cls=dict(type='CrossEntropyLoss'),
                 loss_reg=dict(type='BCELossWithLogits'),
                 spatial_type='avg',
                 dropout_ratio=0.5,
                 init_std=0.01):
        super().__init__()

        self.num_classes = num_classes
        self.in_channels = in_channels
        self.loss_cls = build_loss(loss_cls)
        self.loss_reg = build_loss(loss_reg)
        self.num_stages = num_stages
        self.spatial_type = spatial_type
        self.dropout_ratio = dropout_ratio
        self.init_std = init_std

        if self.spatial_type == 'avg':
            # use `nn.AdaptiveAvgPool3d` to adaptively match the in_channels.
            self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        else:
            self.avg_pool = nn.Identity()

        if self.dropout_ratio > 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = nn.Identity()

        self.cls_fc = nn.Linear(self.in_channels, self.num_classes)

        self.coral_fc = nn.Linear(self.in_channels, 1, bias=False)
        self.coral_bias = nn.Parameter(torch.zeros(1, self.num_stages), requires_grad=True)

    def init_weights(self):
        normal_init(self.cls_fc, std=self.init_std)
        normal_init(self.coral_fc, std=self.init_std)
        constant_init(self.coral_bias, 0)

    def forward(self, x):
        x = self.avg_pool(x)
        x = x.view(x.shape[0], -1)
        x = self.dropout(x)
        cls_score = self.cls_fc(x)
        reg_score = self.coral_fc(x) + self.coral_bias
        return cls_score, reg_score
