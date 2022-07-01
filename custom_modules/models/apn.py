import torch

from mmaction.models.builder import LOCALIZERS, build_backbone, build_head
from mmaction.models.recognizers import Recognizer3D
from torch import nn
import einops


@LOCALIZERS.register_module()
class Recognizer3DWithProg(Recognizer3D):

    def forward_train(self, imgs, labels, prog_label=None, **kwargs):
        """Defines the computation performed at every call when training."""

        assert self.with_cls_head
        imgs = imgs.reshape((-1, ) + imgs.shape[2:])
        losses = dict()

        x = self.extract_feat(imgs)

        cls_score = self.cls_head(x)
        cls_score = cls_score.view((-1, 400, 10))

        cls_score1 = cls_score.max(dim=-1).values
        cls_score2 = cls_score.gather(index=einops.repeat(labels, 'b i-> b i k', k=10), dim=-2).squeeze(dim=1)

        loss_cls = self.cls_head.loss(cls_score2, prog_label.squeeze(), **kwargs)
        losses['loss_prg'] = loss_cls.pop('loss_cls')
        losses['top1_prg'] = loss_cls.pop('top1_acc')
        loss_cls = self.cls_head.loss(cls_score1, labels.squeeze(), **kwargs)
        losses.update(loss_cls)

        return losses

    def _do_test(self, imgs, prog_label=None):
        """Defines the computation performed at every call when evaluation,
        testing and gradcam."""
        batches = imgs.shape[0]
        num_segs = imgs.shape[1]
        imgs = imgs.reshape((-1, ) + imgs.shape[2:])

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
        cls_score = self.cls_head(feat)

        if prog_label is not None and False:
            assert cls_score.shape == (30, 4000), "currently only support our config"
            cls_score = cls_score.reshape(1, 10, 3, 400, 10)
            cls_score = cls_score.gather(dim=-1, index=einops.repeat(prog_label, 'b (n i)-> b n i j k', i=3, j=400, k=1))
            cls_score = cls_score.reshape(30, 400)
        else:
            cls_score = cls_score.reshape(-1, 400, 10).max(dim=-1).values
        cls_score = self.average_clip(cls_score, num_segs)
        return cls_score

    def forward_test(self, imgs, prog_label=None):
        """Defines the computation performed at every call when evaluation and
        testing."""
        return self._do_test(imgs, prog_label).cpu().numpy()