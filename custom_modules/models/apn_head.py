from abc import ABCMeta

import torch
import torch.nn as nn
from mmaction.models.heads.base import AvgConsensus

from mmcv.cnn import kaiming_init, normal_init, constant_init
from mmaction.models.builder import HEADS, build_loss


class BiasLayer(nn.Module, metaclass=ABCMeta):
    def __init__(self, num_bias):
        super(BiasLayer, self).__init__()
        self.num_bias = num_bias
        self.bias = nn.Parameter(torch.zeros(num_bias).float(), requires_grad=True)

    def forward(self, x):
        return x + self.bias


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
    """

    def __init__(self,
                 num_stages=100,
                 in_channels=2048,
                 output_type='coral',
                 loss=dict(type='BCELossWithLogits'),
                 spatial_type='avg3d',
                 dropout_ratio=0.5):
        super().__init__()

        self.in_channels = in_channels
        self.output_type = output_type
        self.loss = build_loss(loss)
        self.num_stages = num_stages
        self.spatial_type = spatial_type
        self.dropout_ratio = dropout_ratio

        if self.dropout_ratio:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None

        if self.spatial_type == 'avg3d':
            self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        elif self.spatial_type == 'avg2d':
            self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        else:
            self.avg_pool = None

        if self.output_type == 'regression':
            self.layers = nn.Linear(self.in_channels, 1)
        elif self.output_type == 'classification':  # including cost-sensitive classification
            self.layers = nn.Linear(self.in_channels, self.num_stages + 1)
        elif self.output_type == 'binary_decomposition':
            self.bi_cls_fc = nn.Linear(self.in_channels, self.num_stages * 2)
            self.view = nn.Unflatten(dim=-1, unflattened_size=(self.num_stages, 2))
            self.layers = nn.Sequential(self.bi_cls_fc, self.view)
        elif self.output_type == 'coral':
            self.coral_fc = nn.Linear(self.in_channels, 1, bias=False)
            self.coral_bias = BiasLayer(self.num_stages)
            self.layers = nn.Sequential(self.coral_fc, self.coral_bias)
        else:
            raise ValueError(f"output_type: {self.output_type} not allowed")

    def init_weights(self):
        if self.output_type == 'coral':
            kaiming_init(self.coral_fc, a=0, nonlinearity='relu', distribution='uniform')
            constant_init(self.coral_bias, 0)
        else:
            normal_init(self.layers, std=0.01)

    def forward(self, x):
        x = self.avg_pool(x)
        x = x.view(x.shape[0], -1)
        if self.dropout is not None:
            x = self.dropout(x)
        score = self.layers(x)
        return score

    def decode_output(self, output):
        if self.output_type == 'coral':
            output = torch.sigmoid(output)
            progressions = torch.count_nonzero(output > 0.5, dim=-1)
        else:
            raise TypeError("other than CORAL Not supported yet")
        return progressions
