# mimic the pytorchvideo.model.stem.py, revised to support early convolution
from typing import Callable
import torch.nn as nn
from pytorchvideo.models.stem import PatchEmbed


def create_early_conv_patch_embed(
        *,
        in_channels: int,
        out_channels: int,
        conv_bias: bool = True,
        conv: Callable = nn.Conv3d,
) -> nn.Module:
    conv_module = nn.Sequential(
        conv(in_channels=in_channels,
             out_channels=out_channels,
             kernel_size=(3, 3, 3),
             stride=(2, 2, 2),
             padding=(1, 1, 1),
             bias=conv_bias),
        conv(in_channels=out_channels,
             out_channels=out_channels,
             kernel_size=(1, 3, 3),
             stride=(1, 2, 2),
             padding=(0, 1, 1),
             bias=conv_bias),
        # conv(in_channels=out_channels,
        #      out_channels=out_channels,
        #      kernel_size=(1, 3, 3),
        #      stride=(1, 2, 2),
        #      padding=(0, 1, 1),
        #      bias=conv_bias),
        conv(in_channels=out_channels,
             out_channels=out_channels,
             kernel_size=(1, 1, 1),
             stride=(1, 1, 1),
             bias=conv_bias)
    )
    return PatchEmbed(patch_model=conv_module)
