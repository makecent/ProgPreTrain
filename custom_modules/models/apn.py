import torch

from mmaction.models.builder import LOCALIZERS, build_backbone, build_head
from mmaction.models.recognizers import Recognizer3D
from torch import nn
import einops
import numpy as np


@LOCALIZERS.register_module()
class Recognizer3DWithProg(Recognizer3D):

    def forward_train(self, imgs, labels, prog_label=None, **kwargs):
        """Defines the computation performed at every call when training."""

        assert self.with_cls_head
        imgs = imgs.reshape((-1,) + imgs.shape[2:])
        losses = dict()

        x = self.extract_feat(imgs)
        if self.with_neck:
            x, loss_aux = self.neck(x, labels.squeeze())
            losses.update(loss_aux)

        cls_score, reg_score = self.cls_head(x)
        gt_labels = labels.squeeze()
        loss_cls = self.cls_head.loss(cls_score, gt_labels, **kwargs)
        losses.update(loss_cls)

        gt_prog = prog_label.squeeze(dim=-1)
        losses['loss_reg'] = self.cls_head.loss_reg(reg_score, gt_prog, **kwargs)

        return losses

    def _do_test(self, imgs, prog_label=None):
        """Defines the computation performed at every call when evaluation,
        testing and gradcam."""
        batches = imgs.shape[0]
        num_segs = imgs.shape[1]
        imgs = imgs.reshape((-1,) + imgs.shape[2:])

        if self.max_testing_views is not None:
            total_views = imgs.shape[0]
            assert num_segs == total_views, (
                'max_testing_views is only compatible '
                'with batch_size == 1')
            view_ptr = 0
            feats = []
            while view_ptr < total_views:
                batch_imgs = imgs[view_ptr:view_ptr + self.max_testing_views]
                x = self.extract_feat(batch_imgs)
                if self.with_neck:
                    x, _ = self.neck(x)
                feats.append(x)
                view_ptr += self.max_testing_views
            # should consider the case that feat is a tuple
            if isinstance(feats[0], tuple):
                len_tuple = len(feats[0])
                feat = [
                    torch.cat([x[i] for x in feats]) for i in range(len_tuple)
                ]
                feat = tuple(feat)
            else:
                feat = torch.cat(feats)
        else:
            feat = self.extract_feat(imgs)
            if self.with_neck:
                feat, _ = self.neck(feat)

        if self.feature_extraction:
            feat_dim = len(feat[0].size()) if isinstance(feat, tuple) else len(
                feat.size())
            assert feat_dim in [
                5, 2
            ], ('Got feature of unknown architecture, '
                'only 3D-CNN-like ([N, in_channels, T, H, W]), and '
                'transformer-like ([N, in_channels]) features are supported.')
            if feat_dim == 5:  # 3D-CNN architecture
                # perform spatio-temporal pooling
                avg_pool = nn.AdaptiveAvgPool3d(1)
                if isinstance(feat, tuple):
                    feat = [avg_pool(x) for x in feat]
                    # concat them
                    feat = torch.cat(feat, axis=1)
                else:
                    feat = avg_pool(feat)
                # squeeze dimensions
                feat = feat.reshape((batches, num_segs, -1))
                # temporal average pooling
                feat = feat.mean(axis=1)
            return feat

        # should have cls_head if not extracting features
        assert self.with_cls_head
        cls_score, reg_score = self.cls_head(feat)
        cls_score = self.average_clip(cls_score, num_segs)
        if prog_label is not None:
            prog_mae = progression_mae(reg_score, prog_label)
            return list(zip(cls_score.cpu().numpy(), prog_mae.cpu().numpy()))
        else:
            return cls_score.cpu().numpy()

    def forward_test(self, imgs, prog_label=None):
        """Defines the computation performed at every call when evaluation and
        testing."""
        return self._do_test(imgs, prog_label)


def decode_progression(reg_score):
    batch_size, num_stage = reg_score.shape
    if isinstance(reg_score, torch.Tensor):
        progression = torch.count_nonzero(reg_score > 0.5, dim=-1)
        # x1 = torch.cat([torch.ones((batch_size, 1), device=reg_score.device), reg_score], dim=-1)
        # x2 = torch.cat([reg_score, torch.zeros((batch_size, 1), device=reg_score.device)], dim=-1)
        # p = (x1 - x2).clamp(0)
        # v = torch.arange(num_stage+1, device=reg_score.device).repeat((batch_size, 1))
        # progression = (p * v).sum(dim=-1)
    elif isinstance(reg_score, np.ndarray):
        progression = np.count_nonzero(reg_score > 0.5, axis=-1)
    else:
        raise TypeError(f"unsupported reg_score type: {type(reg_score)}")
    progression = progression * 100 / num_stage
    return progression


def progression_mae(reg_score, progression_label):
    progression = decode_progression(reg_score)
    progression_label = decode_progression(progression_label)
    print(progression.shape, progression_label.shape)
    if isinstance(reg_score, torch.Tensor):
        mae = torch.abs(progression - progression_label)
    elif isinstance(reg_score, np.ndarray):
        mae = np.abs(progression - progression_label)
    else:
        raise TypeError(f"unsupported reg_score type: {type(reg_score)}")
    return mae
