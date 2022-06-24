# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.modules.linear import NonDynamicallyQuantizableLinear
from einops import rearrange, repeat
from mmcv.cnn import build_norm_layer, constant_init
from mmcv.cnn.bricks.registry import ATTENTION, FEEDFORWARD_NETWORK
from mmcv.cnn.bricks.transformer import FFN, build_dropout, MultiheadAttention
from mmcv.runner.base_module import BaseModule
import warnings
from mmcv.utils import digit_version
from pytorchvideo.layers.utils import set_attributes
from torch.nn.init import constant_, xavier_normal_, xavier_uniform_


@ATTENTION.register_module()
class DecomposedAttentionWithNorm(BaseModule):
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
                 attn_drop=0.,
                 proj_drop=0.,
                 dropout_layer=dict(type='DropPath', drop_prob=0.1),
                 norm_cfg=dict(type='LN'),
                 in_proj=True,
                 in_sharing=False,
                 out_proj=True,
                 out_sharing=False,
                 mid_residual=True,
                 temporal_first=True,
                 init_cfg=None,
                 temporal_cls_attn=True,
                 spatial_cls_attn=True,
                 mid_fc=False,
                 parallel=False,
                 **kwargs):
        super().__init__(init_cfg)
        set_attributes(self, locals())
        self.head_dim = embed_dims // num_heads
        self.scale = self.head_dim ** -0.5
        self.norm1 = build_norm_layer(norm_cfg, self.embed_dims)[1]
        self.norm2 = build_norm_layer(norm_cfg, self.embed_dims)[1] if mid_residual else nn.Identity()
        self.drop_path1 = build_dropout(
            dropout_layer) if dropout_layer else nn.Identity()
        self.drop_path2 = build_dropout(
            dropout_layer) if dropout_layer else nn.Identity()
        self.drop_out1 = nn.Dropout(attn_drop)
        self.drop_out2 = nn.Dropout(attn_drop)
        self.proj_drop1 = nn.Dropout(proj_drop)
        self.proj_drop2 = nn.Dropout(proj_drop)
        self.modes = ('temporal', 'spatial') if temporal_first else ('spatial', 'temporal')
        cls_attn = (spatial_cls_attn, temporal_cls_attn)
        self.cls_attn = cls_attn[::-1] if temporal_first else cls_attn
        if self.parallel:
            self.alpha = nn.Parameter(torch.tensor(1.0, dtype=torch.float), requires_grad=True)

        self.block_in = nn.Linear(self.embed_dims, 3 * self.embed_dims)
        if self.out_proj:
            self.out_proj = self.block_out if self.out_sharing else nn.Linear(self.embed_dims, self.embed_dims)
        else:
            self.out_proj = nn.Identity()
        if self.in_proj:
            self.in_proj = self.block_in if self.in_sharing else nn.Linear(self.embed_dims, 3 * self.embed_dims)
        else:
            self.in_proj = nn.Identity()
        self.block_out = nn.Linear(self.embed_dims, self.embed_dims)

        # used in TimeSformer, which I think is unnecessary because the MultiHeadAttention includes an OUT projection.
        self.mid_fc = nn.Linear(self.embed_dims, self.embed_dims) if mid_fc else nn.Identity()

        self.init_weights()

    def init_weights(self):
        xavier_uniform_(self.block_in.weight)
        constant_(self.block_in.bias, 0.)
        if not isinstance(self.in_proj, nn.Identity):
            xavier_uniform_(self.in_proj.weight)
            constant_(self.in_proj.bias, 0.)
        if not isinstance(self.mid_fc, nn.Identity):
            constant_init(self.mid_fc, val=0, bias=0)

    def forward(self, x, *args, **kwargs):
        identity = x if not self.mid_residual else 0

        x2 = self.attention(x, mode=self.modes[0], cls_attn=self.cls_attn[0],
                            in_proj=self.block_in, norm=self.norm1, drop_out=self.drop_out1,
                            drop_path=self.drop_path1, out_proj=self.out_proj, proj_drop=self.proj_drop1,
                            residual=self.mid_residual, extra_fc=self.mid_fc)
        if self.parallel:
            x3 = x
        else:
            x3 = x2
        x4 = self.attention(x3, mode=self.modes[1], cls_attn=self.cls_attn[1],
                            in_proj=self.in_proj, norm=self.norm2, drop_out=self.drop_out2,
                            drop_path=self.drop_path2, out_proj=self.block_out, proj_drop=self.proj_drop2,
                            residual=self.mid_residual)
        if self.parallel:
            alpha = self.alpha.sigmoid()
            x = (1 - alpha) * x2 + alpha * x4
        else:
            x = x4
        x += identity
        return x

    def attention(self, x, mode, cls_attn, in_proj, norm, drop_out, drop_path, out_proj, proj_drop, residual,
                  extra_fc=None):
        if residual:
            identity = x if cls_attn else x[:, 1:, :]
        else:
            identity = 0

        cls_token = x[:, 0, :].unsqueeze(1)
        x = x[:, 1:, :]

        h, c = self.num_heads, self.head_dim
        b, tp, m = x.size()
        t, p = self.num_frames, tp // self.num_frames

        if mode == 'temporal':
            x = rearrange(x, 'b (p t) m -> (b p) t m', p=p, t=t)
            b1, n = p, t
        else:
            x = rearrange(x, 'b (p t) m -> (b t) p m', p=p, t=t)
            b1, n = t, p

        if cls_attn:
            cls_token = repeat(cls_token, 'b i m -> (b b1) i m', b1=b1, i=1)
            x = torch.cat((cls_token, x), 1)

        x = in_proj(norm(x))
        x = rearrange(x, '(b b1) n (i h c) -> i (b b1) h n c', b1=b1, h=h, c=c)

        q, k, v = x if x.size(0) == 3 else (x[0], x[0], x[0])
        attn = (q @ k.transpose(-1, -2)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = drop_out(attn)
        x = rearrange(attn @ v, '(b b1) h n c -> (b b1) n (h c)', b1=b1)

        x = proj_drop(out_proj(x))
        x = drop_path(x.contiguous())

        if cls_attn:
            cls_token = x[:, 0, :].reshape(b, b1, m)
            cls_token = torch.mean(cls_token, 1, True)
            x = x[:, 1:, :]

        if mode == 'temporal':
            x = rearrange(x, '(b p) t m -> b (p t) m', p=p, t=t)
        else:
            x = rearrange(x, '(b t) p m -> b (p t) m', p=p, t=t)

        if cls_attn:
            x = torch.cat((cls_token, x), 1)
            if extra_fc:
                x = extra_fc(x)
            x += identity
        else:
            if extra_fc:
                x = extra_fc(x)
            x += identity
            x = torch.cat((cls_token, x), 1)
        return x

    # def spatial_attention(self, x, in_proj, norm, drop_out, drop_path, out_proj, proj_drop):
    #     init_cls_token = x[:, 0, :].unsqueeze(1)
    #     x = x[:, 1:, :]
    #
    #     # query [batch_size, num_frames * num_patches, embed_dims]
    #     h, c = self.num_heads, self.head_dim
    #     b, tp, m = x.size()
    #     t, p = self.num_frames, tp // self.num_frames
    #
    #     cls_token = repeat(init_cls_token, 'b i m -> (b t) i m', t=t)
    #
    #     # query [batch_size * num_frames, num_patches + 1, embed_dims]
    #     x = rearrange(x, 'b (p t) m -> (b t) p m', p=p, t=t)
    #     x = torch.cat((cls_token, x), 1)
    #
    #     x = in_proj(norm(x))
    #     x = rearrange(x, '(b t) p (i h c) -> i (b t) h p c', t=t, h=h, c=c)
    #
    #     q, k, v = x if x.size(0) == 3 else (x[0], x[0], x[0])
    #     attn = (q @ k.transpose(-1, -2)) * self.scale
    #     attn = attn.softmax(dim=-1)
    #     attn = drop_out(attn)
    #     x = rearrange(attn @ v, '(b t) h p c -> (b t) p (h c)', t=t)
    #
    #     x = proj_drop(out_proj(x))
    #     x = drop_path(x.contiguous())
    #
    #     # cls_token [batch_size, 1, embed_dims]
    #     cls_token = x[:, 0, :].reshape(b, t, m)
    #     cls_token = torch.mean(cls_token, 1, True)
    #
    #     # res_spatial [batch_size * num_frames, num_patches + 1, embed_dims]
    #     x = rearrange(x[:, 1:, :], '(b t) p m -> b (p t) m', t=t)
    #     x = torch.cat((cls_token, x), 1)
    #
    #     return x


class TimeSpaceAttention(nn.Module):
    def __init__(self, embed_dims, num_heads, temporal_dim):
        super(TimeSpaceAttention, self).__init__()
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.head_dim = embed_dims // num_heads
        self.temporal_dim = temporal_dim
        self.scale = self.head_dim ** -0.5

        self.in_proj_weight = nn.Parameter(torch.empty((3 * embed_dims, embed_dims)))
        self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dims))
        self.out_proj = NonDynamicallyQuantizableLinear(embed_dims, embed_dims)
        self._reset_parameters()

    def _reset_parameters(self):
        if self._qkv_same_embed_dim:
            xavier_uniform_(self.in_proj_weight)
        else:
            xavier_uniform_(self.q_proj_weight)
            xavier_uniform_(self.k_proj_weight)
            xavier_uniform_(self.v_proj_weight)

        if self.in_proj_bias is not None:
            constant_(self.in_proj_bias, 0.)
            constant_(self.out_proj.bias, 0.)
        if self.bias_k is not None:
            xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            xavier_normal_(self.bias_v)

    def forward(self, query):
        # cls_token = query[:, 0, :].unsqueeze(1)
        # query = query[:, 1:, :]
        tgt_len, bsz, embed_dim = query.shape
        q, k, v = F._in_projection_packed(query, query, query, self.in_proj_weight, self.in_proj_bias)
        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        k = k.contiguous().view(k.shape[0], bsz * self.num_heads, self.head_dim).transpose(0, 1)
        v = v.contiguous().view(v.shape[0], bsz * self.num_heads, self.head_dim).transpose(0, 1)

        # Temporal attention
        x = self.temporal_attention(q, k, v)
        x = self.spatial_attention(x, x, x)
        x = x.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        x = F.linear(x, self.out_proj.weight, self.out_proj.bias)
        return x

    def temporal_attention(self, q, k, v):
        bh, l, d = q.size()
        t, s = self.temporal_dim, l // self.temporal_dim

        q = rearrange(q, 'bh (s t) d -> (bh s) t d', bh=bh, s=s, t=t, d=d)
        k = rearrange(k, 'bh (s t) d -> (bh s) t d', bh=bh, s=s, t=t, d=d)
        v = rearrange(v, 'bh (s t) d -> (bh s) t d', bh=bh, s=s, t=t, d=d)

        attn = (q @ k.transpose(-1, -2)) * self.scale
        attn = attn.softmax(dim=-1)
        x = attn @ v

        x = rearrange(x, '(bh s) t d -> bh (s t) d', bh=bh, s=s, t=t, d=d)
        return x

    def spatial_attention(self, q, k, v):
        bh, l, d = q.size()
        t, s = self.temporal_dim, l // self.temporal_dim

        q = rearrange(q, 'bh (s t) d -> (bh t) s d', bh=bh, s=s, t=t, d=d)
        k = rearrange(k, 'bh (s t) d -> (bh t) s d', bh=bh, s=s, t=t, d=d)
        v = rearrange(v, 'bh (s t) d -> (bh t) s d', bh=bh, s=s, t=t, d=d)

        attn = (q @ k.transpose(-1, -2)) * self.scale
        attn = attn.softmax(dim=-1)
        x = attn @ v

        x = rearrange(x, '(bh t) s m -> bh (s t) d', bh=bh, t=t, s=s, d=d)
        return x


@ATTENTION.register_module()
class MultiheadTimeSpaceAttention(MultiheadAttention):
    def __init__(self,
                 temporal_dim,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.attn = TimeSpaceAttention(self.embed_dims, self.num_heads, temporal_dim=temporal_dim)

    def forward(self,
                query,
                key=None,
                value=None,
                identity=None,
                query_pos=None,
                key_pos=None,
                attn_mask=None,
                key_padding_mask=None,
                **kwargs):

        if key is None:
            key = query
        if value is None:
            value = key
        if identity is None:
            identity = query
        if key_pos is None:
            if query_pos is not None:
                # use query_pos if key_pos is not available
                if query_pos.shape == key.shape:
                    key_pos = query_pos
                else:
                    warnings.warn(f'position encoding of key is'
                                  f'missing in {self.__class__.__name__}.')
        if query_pos is not None:
            query = query + query_pos
        if key_pos is not None:
            key = key + key_pos

        if self.batch_first:
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)

        out = self.attn(
            query=query,
            key=key,
            value=value,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask)

        if self.batch_first:
            out = out.transpose(0, 1)

        return identity + self.dropout_layer(self.proj_drop(out))
