# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from einops import rearrange, repeat
from mmcv.cnn import build_norm_layer, constant_init
from mmcv.cnn.bricks.registry import ATTENTION, FEEDFORWARD_NETWORK, TRANSFORMER_LAYER
from mmcv.cnn.bricks.transformer import FFN, build_dropout
from mmcv.runner.base_module import BaseModule
from mmcv.utils import digit_version
from pytorchvideo.layers.utils import set_attributes
from torch.nn.init import constant_, xavier_normal_, xavier_uniform_
from torch.nn import MultiheadAttention

from mmcv import ConfigDict
from mmcv.cnn.bricks.transformer import build_transformer_layer, build_attention


@ATTENTION.register_module()
class Distillation(BaseModule):
    """Temporal Attention in Divided Space Time Attention.
    Args:
        embed_dims (int): Dimensions of embedding.
        num_heads (int): Number of parallel attention heads in
            TransformerCoder.
        num_frames (int): Number of frames in the video.
        attn_drop (float): A Dropout layer on attn_output_weights. Defaults to
            0..
        proj_drop (float): A Dropout layer after `nn.MultiheadAttention`.
            Defaults to 0..
        dropout_layer (dict): The dropout_layer used when adding the shortcut.
            Defaults to `dict(type='DropPath', drop_prob=0.1)`.
        norm_cfg (dict): Config dict for normalization layer. Defaults to
            `dict(type='LN')`.
        init_cfg (dict | None): The Config for initialization. Defaults to
            None.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 num_frames,
                 dropout_layer=0.1,
                 init_cfg=None,
                 distill_prob=0.5,
                 **kwargs):
        super().__init__(init_cfg)
        set_attributes(self, locals())
        temporal_attn_cfg = ConfigDict(
            dict(
                type='DividedTemporalAttentionWithNorm',
                embed_dims=embed_dims,
                num_heads=num_heads,
                num_frames=num_frames,
                dropout_layer=dict(type='DropPath', drop_prob=dropout_layer),
                norm_cfg=dict(type='LN', eps=1e-6)))
        spatial_attn_cfg = ConfigDict(
            dict(
                type='DividedSpatialAttentionWithNorm',
                embed_dims=embed_dims,
                num_heads=num_heads,
                num_frames=num_frames,
                dropout_layer=dict(type='DropPath', drop_prob=dropout_layer),
                norm_cfg=dict(type='LN', eps=1e-6)))
        self.temporal_attn = build_attention(temporal_attn_cfg)
        self.spatial_attn = build_attention(spatial_attn_cfg)
        joint_attn_cfg = ConfigDict(
            dict(
                type='MultiheadAttention',
                embed_dims=embed_dims,
                num_heads=num_heads,
                batch_first=True,
                dropout_layer=dict(type='DropPath', drop_prob=dropout_layer)))
        norm_cfg = dict(type='LN', eps=1e-6)
        self.joint_attn_norm = build_norm_layer(norm_cfg, self.embed_dims)[1]
        self.joint_attn = build_attention(joint_attn_cfg)

    def forward(self, x, *args, **kwargs):
        toss_coin = (self.distill_prob + torch.rand(1)).floor() if self.training else 0
        if toss_coin:
            x = self.spatial_attn(self.temporal_attn(x, *args, **kwargs))
        else:
            identity = x
            x = self.joint_attn_norm(x)
            x = self.joint_attn(x, x, x, identity=identity, **kwargs)
        return x
