# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from functools import partial
from typing import Callable, List, Optional, Tuple

import numpy
import torch
import torch.nn as nn
from mmaction.models.builder import BACKBONES
from pytorchvideo.layers import SpatioTemporalClsPositionalEncoding
from pytorchvideo.layers.drop_path import DropPath
from pytorchvideo.layers.utils import round_width, set_attributes
from pytorchvideo.models.hub.utils import MODEL_ZOO_ROOT_DIR
from pytorchvideo.models.stem import create_conv_patch_embed
from pytorchvideo.models.weight_init import init_net_weights
from torch.hub import load_state_dict_from_url
from torch.nn.common_types import _size_2_t, _size_3_t
from .stem import create_early_conv_patch_embed


class Mlp(nn.Module):
    """
    A MLP block that contains two linear layers with a normalization layer. The MLP
    block is used in a transformer model after the attention block.
    ::
                         Linear (in_features, hidden_features)
                                           ↓
                                 Normalization (act_layer)
                                           ↓
                                Dropout (p=dropout_rate)
                                           ↓
                         Linear (hidden_features, out_features)
                                           ↓
                                Dropout (p=dropout_rate)
    """

    def __init__(
            self,
            in_features: int,
            hidden_features: Optional[int] = None,
            out_features: Optional[int] = None,
            act_layer: Callable = nn.GELU,
            dropout_rate: float = 0.0,
            bias_on: bool = True,
    ) -> None:
        """
        Args:
            in_features (int): Input feature dimension.
            hidden_features (Optional[int]): Hidden feature dimension. By default,
                hidden feature is set to input feature dimension.
            out_features (Optional[int]): Output feature dimension. By default, output
                features dimension is set to input feature dimension.
            act_layer (Callable): Activation layer used after the first linear layer.
            dropout_rate (float): Dropout rate after each linear layer. Dropout is not used
                by default.
        """
        super().__init__()
        self.dropout_rate = dropout_rate
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias_on)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias_on)
        if self.dropout_rate > 0.0:
            self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (tensor): Input tensor.
        """
        x = self.fc1(x)
        x = self.act(x)
        if self.dropout_rate > 0.0:
            x = self.dropout(x)
        x = self.fc2(x)
        if self.dropout_rate > 0.0:
            x = self.dropout(x)
        return x


def _attention_pool(
        tensor: torch.Tensor,
        pool: Optional[Callable],
        thw_shape: List[int],
        has_cls_embed: bool = True,
        norm: Optional[Callable] = None,
) -> torch.Tensor:
    """
    Apply pool to a flattened input (given pool operation and the unflattened shape).
                                         Input
                                           ↓
                                        Reshape
                                           ↓
                                          Pool
                                           ↓
                                        Reshape
                                           ↓
                                          Norm
    Args:
        tensor (torch.Tensor): Input tensor.
        pool (Optional[Callable]): Pool operation that is applied to the input tensor.
            If pool is none, return the input tensor.
        thw_shape (List): The shape of the input tensor (before flattening).
        has_cls_embed (bool): Whether the input tensor contains cls token. Pool
            operation excludes cls token.
        norm: (Optional[Callable]): Optional normalization operation applied to
         tensor after pool.
    Returns:
        tensor (torch.Tensor): Input tensor after pool.
        thw_shape (List[int]): Output tensor shape (before flattening).
    """
    if pool is None:
        return tensor, thw_shape
    tensor_dim = tensor.ndim
    if tensor_dim == 4:
        pass
    elif tensor_dim == 3:
        tensor = tensor.unsqueeze(1)
    else:
        raise NotImplementedError(f"Unsupported input dimension {tensor.shape}")

    if has_cls_embed:
        cls_tok, tensor = tensor[:, :, :1, :], tensor[:, :, 1:, :]

    B, N, L, C = tensor.shape
    T, H, W = thw_shape
    tensor = tensor.reshape(B * N, T, H, W, C).permute(0, 4, 1, 2, 3).contiguous()

    if isinstance(norm, (nn.BatchNorm3d, nn.Identity)):
        # If use BN, we apply norm before pooling instead of after pooling.
        tensor = norm(tensor)
        # We also empirically find that adding a GELU here is beneficial.
        tensor = nn.functional.gelu(tensor)

    tensor = pool(tensor)

    thw_shape = [tensor.shape[2], tensor.shape[3], tensor.shape[4]]
    L_pooled = tensor.shape[2] * tensor.shape[3] * tensor.shape[4]
    tensor = tensor.reshape(B, N, C, L_pooled).transpose(2, 3)
    if has_cls_embed:
        tensor = torch.cat((cls_tok, tensor), dim=2)
    if norm is not None and not isinstance(norm, nn.BatchNorm3d):
        tensor = norm(tensor)

    if tensor_dim == 4:
        pass
    else:  # For the case tensor_dim == 3.
        tensor = tensor.squeeze(1)
    return tensor, thw_shape


class MultiScaleAttention(nn.Module):
    """
    Implementation of a multiscale attention block. Compare to a conventional attention
    block, a multiscale attention block optionally supports pooling (either
    before or after qkv projection). If pooling is not used, a multiscale attention
    block is equivalent to a conventional attention block.
    ::
                                   Input
                                     |
                    |----------------|-----------------|
                    ↓                ↓                 ↓
                  Linear           Linear            Linear
                    &                &                 &
                 Pool (Q)         Pool (K)          Pool (V)
                    → -------------- ←                 |
                             ↓                         |
                       MatMul & Scale                  |
                             ↓                         |
                          Softmax                      |
                             → ----------------------- ←
                                         ↓
                                   MatMul & Scale
                                         ↓
                                      DropOut
    """

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            dropout_rate: float = 0.0,
            kernel_q: _size_3_t = (1, 1, 1),
            kernel_kv: _size_3_t = (1, 1, 1),
            stride_q: _size_3_t = (1, 1, 1),
            stride_kv: _size_3_t = (1, 1, 1),
            norm_layer: Callable = nn.LayerNorm,
            has_cls_embed: bool = True,
            pool_mode: str = "conv",
            pool_first: bool = False,
            residual_pool: bool = True,
            depthwise_conv: bool = True,
            bias_on: bool = True,
            separate_qkv: bool = True,
    ) -> None:
        """
        Args:
            dim (int): Input feature dimension.
            num_heads (int): Number of heads in the attention layer.
            qkv_bias (bool): If set to False, the qkv layer will not learn an additive
                bias. Default: False.
            dropout_rate (float): Dropout rate.
            kernel_q (_size_3_t): Pooling kernel size for q. If both pooling kernel
                size and pooling stride size are 1 for all the dimensions, pooling is
                disabled.
            kernel_kv (_size_3_t): Pooling kernel size for kv. If both pooling kernel
                size and pooling stride size are 1 for all the dimensions, pooling is
                disabled.
            stride_q (_size_3_t): Pooling kernel stride for q.
            stride_kv (_size_3_t): Pooling kernel stride for kv.
            norm_layer (nn.Module): Normalization layer used after pooling.
            has_cls_embed (bool): If set to True, the first token of the input tensor
                should be a cls token. Otherwise, the input tensor does not contain a
                cls token. Pooling is not applied to the cls token.
            pool_mode (str): Pooling mode. Option includes "conv" (learned pooling), "avg"
                (average pooling), and "max" (max pooling).
            pool_first (bool): If set to True, pool is applied before qkv projection.
                Otherwise, pool is applied after qkv projection. Default: False.
            residual_pool (bool): If set to True, use Improved Multiscale Vision
                Transformer's pooling residual connection.
            depthwise_conv (bool): Whether use depthwise or full convolution for pooling.
            bias_on (bool): Whether use biases for linear layers.
            separate_qkv (bool): Whether to use separate or one layer for qkv projections.
        """

        super().__init__()
        assert pool_mode in ["conv", "avg", "max"]

        self.pool_first = pool_first
        self.dropout_rate = dropout_rate
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.has_cls_embed = has_cls_embed
        self.residual_pool = residual_pool
        self.separate_qkv = separate_qkv
        padding_q = [int(q // 2) for q in kernel_q]
        padding_kv = [int(kv // 2) for kv in kernel_kv]

        if self.pool_first or self.separate_qkv:
            self.q = nn.Linear(dim, dim, bias=qkv_bias)
            self.k = nn.Linear(dim, dim, bias=qkv_bias)
            self.v = nn.Linear(dim, dim, bias=qkv_bias)
        else:
            self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim, bias=True if bias_on else False)
        if dropout_rate > 0.0:
            self.proj_drop = nn.Dropout(dropout_rate)

        # Skip pooling with kernel and stride size of (1, 1, 1).
        if (
                kernel_q is not None
                and numpy.prod(kernel_q) == 1
                and numpy.prod(stride_q) == 1
        ):
            kernel_q = None
        if (
                kernel_kv is not None
                and numpy.prod(kernel_kv) == 1
                and numpy.prod(stride_kv) == 1
        ):
            kernel_kv = None

        if pool_mode in ("avg", "max"):
            pool_op = nn.MaxPool3d if pool_mode == "max" else nn.AvgPool3d
            self.pool_q = (
                pool_op(kernel_q, stride_q, padding_q, ceil_mode=False)
                if kernel_q is not None
                else None
            )
            self.pool_k = (
                pool_op(kernel_kv, stride_kv, padding_kv, ceil_mode=False)
                if kernel_kv is not None
                else None
            )
            self.pool_v = (
                pool_op(kernel_kv, stride_kv, padding_kv, ceil_mode=False)
                if kernel_kv is not None
                else None
            )
        elif pool_mode == "conv":
            self.pool_q = (
                nn.Conv3d(
                    head_dim,
                    head_dim,
                    kernel_q,
                    stride=stride_q,
                    padding=padding_q,
                    groups=head_dim if depthwise_conv else 1,
                    bias=False,
                )
                if kernel_q is not None
                else None
            )
            self.norm_q = norm_layer(head_dim) if kernel_q is not None else None
            self.pool_k = (
                nn.Conv3d(
                    head_dim,
                    head_dim,
                    kernel_kv,
                    stride=stride_kv,
                    padding=padding_kv,
                    groups=head_dim if depthwise_conv else 1,
                    bias=False,
                )
                if kernel_kv is not None
                else None
            )
            self.norm_k = norm_layer(head_dim) if kernel_kv is not None else None
            self.pool_v = (
                nn.Conv3d(
                    head_dim,
                    head_dim,
                    kernel_kv,
                    stride=stride_kv,
                    padding=padding_kv,
                    groups=head_dim if depthwise_conv else 1,
                    bias=False,
                )
                if kernel_kv is not None
                else None
            )
            self.norm_v = norm_layer(head_dim) if kernel_kv is not None else None
        else:
            raise NotImplementedError(f"Unsupported model {pool_mode}")

    def _qkv_proj(
            self,
            q: torch.Tensor,
            q_size: List[int],
            k: torch.Tensor,
            k_size: List[int],
            v: torch.Tensor,
            v_size: List[int],
            batch_size: List[int],
            chan_size: List[int],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q = (
            self.q(q)
                .reshape(batch_size, q_size, self.num_heads, chan_size // self.num_heads)
                .permute(0, 2, 1, 3)
        )
        k = (
            self.k(k)
                .reshape(batch_size, k_size, self.num_heads, chan_size // self.num_heads)
                .permute(0, 2, 1, 3)
        )
        v = (
            self.v(v)
                .reshape(batch_size, v_size, self.num_heads, chan_size // self.num_heads)
                .permute(0, 2, 1, 3)
        )
        return q, k, v

    def _qkv_pool(
            self,
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            thw_shape: Tuple[torch.Tensor, List[int]],
    ) -> Tuple[
        torch.Tensor, List[int], torch.Tensor, List[int], torch.Tensor, List[int]
    ]:
        q, q_shape = _attention_pool(
            q,
            self.pool_q,
            thw_shape,
            has_cls_embed=self.has_cls_embed,
            norm=self.norm_q if hasattr(self, "norm_q") else None,
        )
        k, k_shape = _attention_pool(
            k,
            self.pool_k,
            thw_shape,
            has_cls_embed=self.has_cls_embed,
            norm=self.norm_k if hasattr(self, "norm_k") else None,
        )
        v, v_shape = _attention_pool(
            v,
            self.pool_v,
            thw_shape,
            has_cls_embed=self.has_cls_embed,
            norm=self.norm_v if hasattr(self, "norm_v") else None,
        )
        return q, q_shape, k, k_shape, v, v_shape

    def _get_qkv_length(
            self,
            q_shape: List[int],
            k_shape: List[int],
            v_shape: List[int],
    ) -> Tuple[int]:
        q_N = numpy.prod(q_shape) + 1 if self.has_cls_embed else numpy.prod(q_shape)
        k_N = numpy.prod(k_shape) + 1 if self.has_cls_embed else numpy.prod(k_shape)
        v_N = numpy.prod(v_shape) + 1 if self.has_cls_embed else numpy.prod(v_shape)
        return q_N, k_N, v_N

    def _reshape_qkv_to_seq(
            self,
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            q_N: int,
            v_N: int,
            k_N: int,
            B: int,
            C: int,
    ) -> Tuple[int]:
        q = q.permute(0, 2, 1, 3).reshape(B, q_N, C)
        v = v.permute(0, 2, 1, 3).reshape(B, v_N, C)
        k = k.permute(0, 2, 1, 3).reshape(B, k_N, C)
        return q, k, v

    def forward(
            self, x: torch.Tensor, thw_shape: List[int]
    ) -> Tuple[torch.Tensor, List[int]]:
        """
        Args:
            x (torch.Tensor): Input tensor.
            thw_shape (List): The shape of the input tensor (before flattening).
        """

        B, N, C = x.shape
        if self.pool_first:
            x = x.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            q = k = v = x
            q, q_shape, k, k_shape, v, v_shape = self._qkv_pool(q, k, v, thw_shape)
            q_N, k_N, v_N = self._get_qkv_length(q_shape, k_shape, v_shape)
            q, k, v = self._reshape_qkv_to_seq(q, k, v, q_N, v_N, k_N, B, C)
            q, k, v = self._qkv_proj(q, q_N, k, k_N, v, v_N, B, C)
        else:
            if self.separate_qkv:
                q = k = v = x
                q, k, v = self._qkv_proj(q, N, k, N, v, N, B, C)
            else:
                qkv = (
                    self.qkv(x)
                        .reshape(B, N, 3, self.num_heads, -1)
                        .permute(2, 0, 3, 1, 4)
                )
                q, k, v = qkv[0], qkv[1], qkv[2]
            q, q_shape, k, k_shape, v, v_shape = self._qkv_pool(q, k, v, thw_shape)

        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)

        N = q.shape[2]

        if self.residual_pool:
            x = (attn @ v + q).transpose(1, 2).reshape(B, N, C)
        else:
            x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        if self.dropout_rate > 0.0:
            x = self.proj_drop(x)
        return x, q_shape


class MultiScaleBlock(nn.Module):
    """
    Implementation of a multiscale vision transformer block. Each block contains a
    multiscale attention layer and a Mlp layer.
    ::
                                      Input
                                        |-------------------+
                                        ↓                   |
                                       Norm                 |
                                        ↓                   |
                                MultiScaleAttention        Pool
                                        ↓                   |
                                     DropPath               |
                                        ↓                   |
                                    Summation ←-------------+
                                        |
                                        |-------------------+
                                        ↓                   |
                                       Norm                 |
                                        ↓                   |
                                       Mlp                 Proj
                                        ↓                   |
                                     DropPath               |
                                        ↓                   |
                                    Summation  ←------------+
    """

    def __init__(
            self,
            dim: int,
            dim_out: int,
            num_heads: int,
            mlp_ratio: float = 4.0,
            qkv_bias: bool = False,
            dropout_rate: float = 0.0,
            droppath_rate: float = 0.0,
            act_layer: nn.Module = nn.GELU,
            norm_layer: nn.Module = nn.LayerNorm,
            attn_norm_layer: nn.Module = nn.LayerNorm,
            kernel_q: _size_3_t = (1, 1, 1),
            kernel_kv: _size_3_t = (1, 1, 1),
            stride_q: _size_3_t = (1, 1, 1),
            stride_kv: _size_3_t = (1, 1, 1),
            pool_mode: str = "conv",
            has_cls_embed: bool = True,
            pool_first: bool = False,
            residual_pool: bool = False,
            depthwise_conv: bool = True,
            bias_on: bool = True,
            separate_qkv: bool = True,
    ) -> None:
        """
        Args:
            dim (int): Input feature dimension.
            dim_out (int): Output feature dimension.
            num_heads (int): Number of heads in the attention layer.
            mlp_ratio (float): Mlp ratio which controls the feature dimension in the
                hidden layer of the Mlp block.
            qkv_bias (bool): If set to False, the qkv layer will not learn an additive
                bias. Default: False.
            dropout_rate (float): DropOut rate. If set to 0, DropOut is disabled.
            droppath_rate (float): DropPath rate. If set to 0, DropPath is disabled.
            act_layer (nn.Module): Activation layer used in the Mlp layer.
            norm_layer (nn.Module): Normalization layer.
            attn_norm_layer (nn.Module): Normalization layer in the attention module.
            kernel_q (_size_3_t): Pooling kernel size for q. If pooling kernel size is
                1 for all the dimensions, pooling is not used (by default).
            kernel_kv (_size_3_t): Pooling kernel size for kv. If pooling kernel size
                is 1 for all the dimensions, pooling is not used. By default, pooling
                is disabled.
            stride_q (_size_3_t): Pooling kernel stride for q.
            stride_kv (_size_3_t): Pooling kernel stride for kv.
            pool_mode (str): Pooling mode. Option includes "conv" (learned pooling), "avg"
                (average pooling), and "max" (max pooling).
            has_cls_embed (bool): If set to True, the first token of the input tensor
                should be a cls token. Otherwise, the input tensor does not contain a
                cls token. Pooling is not applied to the cls token.
            pool_first (bool): If set to True, pool is applied before qkv projection.
                Otherwise, pool is applied after qkv projection. Default: False.
            residual_pool (bool): If set to True, use Improved Multiscale Vision
                Transformer's pooling residual connection.
            depthwise_conv (bool): Whether use depthwise or full convolution for pooling.
            bias_on (bool): Whether use biases for linear layers.
            separate_qkv (bool): Whether to use separate or one layer for qkv projections.
        """
        super().__init__()
        self.dim = dim
        self.dim_out = dim_out
        self.norm1 = norm_layer(dim)
        kernel_skip = [s + 1 if s > 1 else s for s in stride_q]
        stride_skip = stride_q
        padding_skip = [int(skip // 2) for skip in kernel_skip]
        self.attn = MultiScaleAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            dropout_rate=dropout_rate,
            kernel_q=kernel_q,
            kernel_kv=kernel_kv,
            stride_q=stride_q,
            stride_kv=stride_kv,
            norm_layer=attn_norm_layer,
            has_cls_embed=has_cls_embed,
            pool_mode=pool_mode,
            pool_first=pool_first,
            residual_pool=residual_pool,
            bias_on=bias_on,
            depthwise_conv=depthwise_conv,
            separate_qkv=separate_qkv,
        )
        self.drop_path = (
            DropPath(droppath_rate) if droppath_rate > 0.0 else nn.Identity()
        )
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.has_cls_embed = has_cls_embed
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            out_features=dim_out,
            act_layer=act_layer,
            dropout_rate=dropout_rate,
            bias_on=bias_on,
        )
        if dim != dim_out:
            self.proj = nn.Linear(dim, dim_out, bias=bias_on)

        self.pool_skip = (
            nn.MaxPool3d(kernel_skip, stride_skip, padding_skip, ceil_mode=False)
            if len(stride_skip) > 0 and numpy.prod(stride_skip) > 1
            else None
        )

    def forward(
            self, x: torch.Tensor, thw_shape: List[int]
    ) -> Tuple[torch.Tensor, List[int]]:
        """
        Args:
            x (torch.Tensor): Input tensor.
            thw_shape (List): The shape of the input tensor (before flattening).
        """

        x_block, thw_shape_new = self.attn(
            (
                self.norm1(x.permute(0, 2, 1)).permute(0, 2, 1)
                if isinstance(self.norm1, nn.BatchNorm1d)
                else self.norm1(x)
            ),
            thw_shape,
        )
        x_res, _ = _attention_pool(
            x, self.pool_skip, thw_shape, has_cls_embed=self.has_cls_embed
        )
        x = x_res + self.drop_path(x_block)
        x_norm = (
            self.norm2(x.permute(0, 2, 1)).permute(0, 2, 1)
            if isinstance(self.norm2, nn.BatchNorm1d)
            else self.norm2(x)
        )
        x_mlp = self.mlp(x_norm)
        if self.dim != self.dim_out:
            x = self.proj(x_norm)
        x = x + self.drop_path(x_mlp)
        return x, thw_shape_new


@BACKBONES.register_module(name="MViT")
class MultiscaleVisionTransformers(nn.Module):
    """
    Multiscale Vision Transformers
    Haoqi Fan, Bo Xiong, Karttikeya Mangalam, Yanghao Li, Zhicheng Yan, Jitendra Malik,
    Christoph Feichtenhofer
    https://arxiv.org/abs/2104.11227

    ::

                                       PatchEmbed
                                           ↓
                                   PositionalEncoding
                                           ↓
                                        Dropout
                                           ↓
                                     Normalization
                                           ↓
                                         Block 1
                                           ↓
                                           .
                                           .
                                           .
                                           ↓
                                         Block N
                                           ↓
                                     Normalization
                                           ↓
                                          Head


    The builder can be found in `create_mvit`.
    """

    def __init__(self, arch='base_16x4', pretrained=True):
        super().__init__()
        mvit_video_small_config = {
            "spatial_size": 224,
            "temporal_size": 16,
            "patch_embed_dim": 128,
            "conv_patch_embed_kernel": (3, 8, 8),
            "conv_patch_embed_stride": (2, 8, 8),
            "embed_dim_mul": [[3, 2.0], [10, 2.0]],
            "atten_head_mul": [[3, 2.0], [10, 2.0]],
            "pool_q_stride_size": [[3, 1, 2, 2], [10, 1, 2, 2]],
            "pool_kv_stride_adaptive": [1, 8, 8],
            "pool_kvq_kernel": [3, 3, 3],
        }
        mvit_video_base_config = {
            "spatial_size": 224,
            "temporal_size": 16,
            "embed_dim_mul": [[1, 2.0], [3, 2.0], [14, 2.0]],
            "atten_head_mul": [[1, 2.0], [3, 2.0], [14, 2.0]],
            "pool_q_stride_size": [[1, 1, 2, 2], [3, 1, 2, 2], [14, 1, 2, 2]],
            "pool_kv_stride_adaptive": [1, 8, 8],
            "pool_kvq_kernel": [3, 3, 3],
        }
        mvit_video_base_32x3_config = {
            "spatial_size": 224,
            "temporal_size": 32,
            "embed_dim_mul": [[1, 2.0], [3, 2.0], [14, 2.0]],
            "atten_head_mul": [[1, 2.0], [3, 2.0], [14, 2.0]],
            "pool_q_stride_size": [[1, 1, 2, 2], [3, 1, 2, 2], [14, 1, 2, 2]],
            "pool_kv_stride_adaptive": [1, 8, 8],
            "pool_kvq_kernel": [3, 3, 3],
        }

        mvit_image_base_16_config = {
            "spatial_size": 224,
            "temporal_size": 1,
            "depth": 16,
            "conv_patch_embed_kernel": [7, 7],
            "conv_patch_embed_stride": [4, 4],
            "conv_patch_embed_padding": [3, 3],
            "use_2d_patch": True,
            "embed_dim_mul": [[1, 2.0], [3, 2.0], [14, 2.0]],
            "atten_head_mul": [[1, 2.0], [3, 2.0], [14, 2.0]],
            "pool_q_stride_size": [[1, 1, 2, 2], [3, 1, 2, 2], [14, 1, 2, 2]],
            "pool_kv_stride_adaptive": [1, 4, 4],
            "pool_kvq_kernel": [1, 3, 3],
        }
        checkpoint_paths = {
            "mvit_base_16x4": "{}/kinetics/MVIT_B_16x4.pyth".format(MODEL_ZOO_ROOT_DIR),
            "mvit_base_32x3": "{}/kinetics/MVIT_B_32x3_f294077834.pyth".format(
                MODEL_ZOO_ROOT_DIR
            ),
            "mvit_base_16": "{}/imagenet/MVIT_B_16_f292487636.pyth".format(MODEL_ZOO_ROOT_DIR),
        }
        if arch == 'small_16x4':
            mvit_config = mvit_video_small_config
        elif arch == 'base_16x4':
            mvit_config = mvit_video_base_config
        elif arch == 'base_32x3':
            mvit_config = mvit_video_base_32x3_config
        elif arch == 'base_16':
            mvit_config = mvit_image_base_16_config
        else:
            raise TypeError(f"{arch} not supported")
        set_attributes(self, self.create_multiscale_vision_transformers(**mvit_config))
        if pretrained:
            checkpoint = load_state_dict_from_url(
                checkpoint_paths[f"mvit_{arch}"], progress=True, map_location="cpu")
            state_dict = checkpoint["model_state"]
            self.load_state_dict(state_dict)
        else:
            init_net_weights(self, init_std=0.02, style="vit")

    def _get_bn_w_b(self, bn, repeat=1):
        w_bn = torch.diag(
            bn.weight.div(torch.sqrt(bn.eps + bn.running_var)).repeat(repeat)
        )

        b_bn = (
            bn.bias
            - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
        ).repeat(repeat)
        return w_bn, b_bn

    def fuse_norm_before_linear(self, bn, linear):
        if bn is None:
            return linear
        w_bn, b_bn = self._get_bn_w_b(bn)
        fused_linear = nn.Linear(linear.in_features, linear.out_features, bias=True)
        fused_linear.weight.data[:] = torch.mm(linear.weight, w_bn)
        fused_linear.bias.data[:] = (
            torch.matmul(linear.weight, b_bn) + linear.bias
            if linear.bias is not None
            else torch.matmul(linear.weight, b_bn)
        )
        return fused_linear

    def fuse_norm_after_linear(self, linear, bn):
        if bn is None:
            return linear
        assert linear.in_features % bn.bias.shape[0] == 0
        num_heads = linear.in_features // bn.bias.shape[0]
        w_bn, b_bn = self._get_bn_w_b(bn, repeat=num_heads)

        fused_linear = nn.Linear(linear.in_features, linear.out_features, bias=True)
        fused_linear.weight.data[:] = torch.mm(w_bn, linear.weight)
        fused_linear.bias.data[:] = (
            torch.matmul(w_bn, linear.bias) + b_bn if linear.bias is not None else b_bn
        )
        return fused_linear

    def fuse_bn(self):
        assert not self.training
        for blk in self.blocks:
            # fuse self.norm1
            if blk.attn.separate_qkv:
                blk.attn.q = self.fuse_norm_before_linear(blk.norm1, blk.attn.q)
                blk.attn.k = self.fuse_norm_before_linear(blk.norm1, blk.attn.k)
                blk.attn.v = self.fuse_norm_before_linear(blk.norm1, blk.attn.v)
            else:
                blk.attn.qkv = self.fuse_norm_before_linear(blk.norm1, blk.attn.qkv)
            blk.norm1 = nn.Identity()

            # fuse the bn in attention
            if blk.attn.separate_qkv:
                blk.attn.q = self.fuse_norm_after_linear(blk.attn.q, blk.attn.norm_q)
                blk.attn.k = self.fuse_norm_after_linear(blk.attn.k, blk.attn.norm_k)
                blk.attn.v = self.fuse_norm_after_linear(blk.attn.v, blk.attn.norm_v)
            else:
                w_q, w_k, w_v = blk.attn.qkv.weight.chunk(3)
                b_q, b_k, b_v = blk.attn.qkv.bias.chunk(3)
                tmp_q = nn.Linear(w_q.shape[1], w_q.shape[0], bias=True)
                tmp_k = nn.Linear(w_k.shape[1], w_k.shape[0], bias=True)
                tmp_v = nn.Linear(w_v.shape[1], w_v.shape[0], bias=True)
                tmp_q.weight.data[:] = w_q
                tmp_k.weight.data[:] = w_k
                tmp_v.weight.data[:] = w_v
                tmp_q.bias.data[:] = b_q
                tmp_k.bias.data[:] = b_k
                tmp_v.bias.data[:] = b_v
                tmp_q = self.fuse_norm_after_linear(tmp_q, blk.attn.norm_q)
                tmp_k = self.fuse_norm_after_linear(tmp_k, blk.attn.norm_k)
                tmp_v = self.fuse_norm_after_linear(tmp_v, blk.attn.norm_v)
                blk.attn.qkv.weight.data[:] = torch.cat(
                    [tmp_q.weight.data, tmp_k.weight.data, tmp_v.weight.data], dim=0
                )
                blk.attn.qkv.bias.data[:] = torch.cat(
                    [tmp_q.bias.data, tmp_k.bias.data, tmp_v.bias.data], dim=0
                )

            blk.attn.norm_q = nn.Identity()
            blk.attn.norm_k = nn.Identity()
            blk.attn.norm_v = nn.Identity()

            # fuse self.norm2
            blk.mlp.fc1 = self.fuse_norm_before_linear(blk.norm2, blk.mlp.fc1)
            if blk.dim != blk.dim_out:
                blk.proj = self.fuse_norm_before_linear(blk.norm2, blk.proj)
            blk.norm2 = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.patch_embed is not None:
            x = self.patch_embed(x)
        x = self.cls_positional_encoding(x)

        if self.pos_drop is not None:
            x = self.pos_drop(x)

        thw = self.cls_positional_encoding.patch_embed_shape
        for blk in self.blocks:
            x, thw = blk(x, thw)
        if self.norm_embed is not None:
            x = self.norm_embed(x)
        if self.head is not None:
            x = self.head(x)
        return x


    def create_multiscale_vision_transformers(self,
                                              *,
                                              spatial_size: _size_2_t,
                                              temporal_size: int,
                                              cls_embed_on: bool = True,
                                              sep_pos_embed: bool = True,
                                              depth: int = 16,
                                              norm: str = "layernorm",
                                              # Patch embed config.
                                              enable_patch_embed: bool = True,
                                              input_channels: int = 3,
                                              patch_embed_dim: int = 96,
                                              conv_patch_embed_kernel: Tuple[int] = (3, 7, 7),
                                              conv_patch_embed_stride: Tuple[int] = (2, 4, 4),
                                              conv_patch_embed_padding: Tuple[int] = (1, 3, 3),
                                              early_convolution=False,                                # Modified
                                              enable_patch_embed_norm: bool = False,
                                              use_2d_patch: bool = False,
                                              # Attention block config.
                                              num_heads: int = 1,
                                              mlp_ratio: float = 4.0,
                                              qkv_bias: bool = True,
                                              dropout_rate_block: float = 0.0,
                                              droppath_rate_block: float = 0.0,
                                              pooling_mode: str = "conv",
                                              pool_first: bool = False,
                                              residual_pool: bool = False,
                                              depthwise_conv: bool = True,
                                              bias_on: bool = True,
                                              separate_qkv: bool = True,
                                              embed_dim_mul: Optional[List[List[int]]] = None,
                                              atten_head_mul: Optional[List[List[int]]] = None,
                                              pool_q_stride_size: Optional[List[List[int]]] = None,
                                              pool_kv_stride_size: Optional[List[List[int]]] = None,
                                              pool_kv_stride_adaptive: Optional[_size_3_t] = None,
                                              pool_kvq_kernel: Optional[_size_3_t] = None,
                                              # Head config.
                                              head: Optional[Callable] = None,
                                              head_dropout_rate: float = 0.5,
                                              head_activation: Callable = None,
                                              head_num_classes: int = 400,
                                              # The default model definition is not TorchScript-friendly.
                                              # Set create_scriptable_model=True to create a TorchScriptable model.
                                              create_scriptable_model: bool = False):
        if use_2d_patch:
            assert temporal_size == 1, "If use_2d_patch, temporal_size needs to be 1."
        if pool_kv_stride_adaptive is not None:
            assert (
                    pool_kv_stride_size is None
            ), "pool_kv_stride_size should be none if pool_kv_stride_adaptive is set."
        if norm == "layernorm":
            norm_layer = partial(nn.LayerNorm, eps=1e-6)
            block_norm_layer = partial(nn.LayerNorm, eps=1e-6)
            attn_norm_layer = partial(nn.LayerNorm, eps=1e-6)
        elif norm == "batchnorm":
            norm_layer = None
            block_norm_layer = nn.BatchNorm1d
            attn_norm_layer = nn.BatchNorm3d
        else:
            raise NotImplementedError("Only supports layernorm.")
        if create_scriptable_model:
            assert (
                    norm == "batchnorm"
            ), "The scriptable model supports only the batchnorm-based model."

        if isinstance(spatial_size, int):
            spatial_size = (spatial_size, spatial_size)

        conv_patch_op = nn.Conv2d if use_2d_patch else nn.Conv3d

        if early_convolution:
            patch_embed = (
                create_early_conv_patch_embed(
                    in_channels=input_channels,
                    out_channels=patch_embed_dim,
                )
                if enable_patch_embed
                else None
            )
        else:
            patch_embed = (
                create_conv_patch_embed(
                    in_channels=input_channels,
                    out_channels=patch_embed_dim,
                    conv_kernel_size=conv_patch_embed_kernel,
                    conv_stride=conv_patch_embed_stride,
                    conv_padding=conv_patch_embed_padding,
                    conv=conv_patch_op,
                )
                if enable_patch_embed
                else None
            )

        input_dims = [temporal_size, spatial_size[0], spatial_size[1]]
        input_stirde = (
            (1,) + tuple(conv_patch_embed_stride)
            if use_2d_patch
            else conv_patch_embed_stride
        )

        patch_embed_shape = (
            [input_dims[i] // input_stirde[i] for i in range(len(input_dims))]
            if enable_patch_embed
            else input_dims
        )

        pos_func = SpatioTemporalClsPositionalEncoding
        cls_positional_encoding = pos_func(
            embed_dim=patch_embed_dim,
            patch_embed_shape=patch_embed_shape,
            sep_pos_embed=sep_pos_embed,
            has_cls=cls_embed_on,
        )

        dpr = [
            x.item() for x in torch.linspace(0, droppath_rate_block, depth)
        ]  # stochastic depth decay rule

        if dropout_rate_block > 0.0:
            pos_drop = nn.Dropout(p=dropout_rate_block)

        dim_mul, head_mul = torch.ones(depth + 1), torch.ones(depth + 1)
        if embed_dim_mul is not None:
            for i in range(len(embed_dim_mul)):
                dim_mul[embed_dim_mul[i][0]] = embed_dim_mul[i][1]
        if atten_head_mul is not None:
            for i in range(len(atten_head_mul)):
                head_mul[atten_head_mul[i][0]] = atten_head_mul[i][1]

        mvit_blocks = nn.ModuleList()

        pool_q = [[] for i in range(depth)]
        pool_kv = [[] for i in range(depth)]
        stride_q = [[] for i in range(depth)]
        stride_kv = [[] for i in range(depth)]

        if pool_q_stride_size is not None:
            for i in range(len(pool_q_stride_size)):
                stride_q[pool_q_stride_size[i][0]] = pool_q_stride_size[i][1:]
                if pool_kvq_kernel is not None:
                    pool_q[pool_q_stride_size[i][0]] = pool_kvq_kernel
                else:
                    pool_q[pool_q_stride_size[i][0]] = [
                        s + 1 if s > 1 else s for s in pool_q_stride_size[i][1:]
                    ]

        # If POOL_KV_STRIDE_ADAPTIVE is not None, initialize POOL_KV_STRIDE.
        if pool_kv_stride_adaptive is not None:
            _stride_kv = pool_kv_stride_adaptive
            pool_kv_stride_size = []
            for i in range(depth):
                if len(stride_q[i]) > 0:
                    _stride_kv = [
                        max(_stride_kv[d] // stride_q[i][d], 1)
                        for d in range(len(_stride_kv))
                    ]
                pool_kv_stride_size.append([i] + _stride_kv)

        if pool_kv_stride_size is not None:
            for i in range(len(pool_kv_stride_size)):
                stride_kv[pool_kv_stride_size[i][0]] = pool_kv_stride_size[i][1:]
                if pool_kvq_kernel is not None:
                    pool_kv[pool_kv_stride_size[i][0]] = pool_kvq_kernel
                else:
                    pool_kv[pool_kv_stride_size[i][0]] = [
                        s + 1 if s > 1 else s for s in pool_kv_stride_size[i][1:]
                    ]

        for i in range(depth):
            num_heads = round_width(num_heads, head_mul[i], min_width=1, divisor=1)
            patch_embed_dim = round_width(patch_embed_dim, dim_mul[i], divisor=num_heads)
            dim_out = round_width(
                patch_embed_dim,
                dim_mul[i + 1],
                divisor=round_width(num_heads, head_mul[i + 1]),
            )

            block_func = MultiScaleBlock

            mvit_blocks.append(
                block_func(
                    dim=patch_embed_dim,
                    dim_out=dim_out,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    dropout_rate=dropout_rate_block,
                    droppath_rate=dpr[i],
                    norm_layer=block_norm_layer,
                    attn_norm_layer=attn_norm_layer,
                    kernel_q=pool_q[i],
                    kernel_kv=pool_kv[i],
                    stride_q=stride_q[i],
                    stride_kv=stride_kv[i],
                    pool_mode=pooling_mode,
                    has_cls_embed=cls_embed_on,
                    pool_first=pool_first,
                    residual_pool=residual_pool,
                    bias_on=bias_on,
                    depthwise_conv=depthwise_conv,
                    separate_qkv=separate_qkv,
                )
            )

        embed_dim = dim_out
        norm_embed = None if norm_layer is None else norm_layer(embed_dim)
        if head is not None:
            head_model = head(
                in_features=embed_dim,
                out_features=head_num_classes,
                seq_pool_type="cls" if cls_embed_on else "mean",
                dropout_rate=head_dropout_rate,
                activation=head_activation,
            )
        else:
            head_model = None

        return dict(patch_embed=patch_embed,
                    cls_positional_encoding=cls_positional_encoding,
                    pos_drop=pos_drop if dropout_rate_block > 0.0 else None,
                    blocks=mvit_blocks,
                    norm_embed=norm_embed,
                    head=head_model)

    def init_weights(self):
        pass
