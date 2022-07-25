# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
from mmcv.cnn import normal_init, kaiming_init
import torch

from mmaction.models.builder import HEADS, build_loss
from mmaction.models.heads import I3DHead


@HEADS.register_module()
class I3DHeadWithProg(I3DHead):
    """Classification head for I3D.

    Args:
        num_classes (int): Number of classes to be classified.
        in_channels (int): Number of channels in input feature.
        loss_cls (dict): Config for building loss.
            Default: dict(type='CrossEntropyLoss')
        spatial_type (str): Pooling type in spatial dimension. Default: 'avg'.
        dropout_ratio (float): Probability of dropout layer. Default: 0.5.
        init_std (float): Std value for Initiation. Default: 0.01.
        kwargs (dict, optional): Any keyword argument to be used to initialize
            the head.
    """

    def __init__(self, num_stages=100,
                 loss_reg=dict(type='BCELossWithLogits'),
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_stages = num_stages
        self.loss_reg = build_loss(loss_reg)
        self.coral_fc = nn.Linear(self.in_channels, 1, bias=False)
        self.coral_bias = nn.Parameter(torch.zeros(1, num_stages), requires_grad=True)

    def init_weights(self):
        """Initiate the parameters from scratch."""
        normal_init(self.fc_cls, std=self.init_std)
        kaiming_init(self.coral_fc, a=0, nonlinearity='relu', distribution='uniform')
        nn.init.constant_(self.coral_bias, 0)

    def forward(self, x):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The classification scores for input samples.
        """
        # [N, in_channels, 4, 7, 7]
        if self.avg_pool is not None:
            x = self.avg_pool(x)
        # [N, in_channels, 1, 1, 1]
        if self.dropout is not None:
            x = self.dropout(x)
        # [N, in_channels, 1, 1, 1]
        x = x.view(x.shape[0], -1)
        # [N, in_channels]
        cls_score = self.fc_cls(x)
        # [N, num_classes]
        reg_score = self.coral_fc(x) + self.coral_bias
        return cls_score, reg_score
