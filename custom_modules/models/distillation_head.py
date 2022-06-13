from mmaction.models.builder import HEADS
from mmaction.models.heads import BaseHead
from pytorchvideo.models.head import SequencePool
from torch import nn
import torch.nn.functional as F
from mmcv.cnn import normal_init, trunc_normal_init
import torch
from mmaction.core import top_k_accuracy


@HEADS.register_module()
class DistillationHead(BaseHead):
    def __init__(self,
                 num_classes,
                 in_channels,
                 loss_cls=dict(type='CrossEntropyLoss'),
                 init_std=0.02,
                 **kwargs):
        super().__init__(num_classes, in_channels, loss_cls, **kwargs)
        self.init_std = init_std
        self.fc_cls = nn.Linear(self.in_channels, self.num_classes)

    def init_weights(self):
        """Initiate the parameters from scratch."""
        trunc_normal_init(self.fc_cls, std=self.init_std)

    def forward(self, x1, x2):
        # [N, in_channels]
        cls_score = self.fc_cls(x)
        # [N, num_classes]
        return cls_score

    def loss(self, cls_score, labels, **kwargs):
        """overwrite the function of the same name in BaseHead to correct the label smoothing"""
        losses = dict()
        if labels.shape == torch.Size([]):
            labels = labels.unsqueeze(0)
        elif labels.dim() == 1 and labels.size()[0] == self.num_classes \
                and cls_score.size()[0] == 1:
            # Fix a bug when training with soft labels and batch size is 1.
            # When using soft labels, `labels` and `cls_socre` share the same
            # shape.
            labels = labels.unsqueeze(0)

        if cls_score.size() != labels.size():
            top_k_acc = top_k_accuracy(cls_score.detach().cpu().numpy(),
                                       labels.detach().cpu().numpy(),
                                       self.topk)
            for k, a in zip(self.topk, top_k_acc):
                losses[f'top{k}_acc'] = torch.tensor(
                    a, device=cls_score.device)
        if self.label_smooth_eps != 0:
            if cls_score.size() != labels.size():
                labels = F.one_hot(labels, num_classes=self.num_classes)
            labels = ((1 - self.label_smooth_eps) * labels +
                      self.label_smooth_eps / self.num_classes)

        loss_cls = self.loss_cls(cls_score, labels, **kwargs)
        # loss_cls may be dictionary or single tensor
        if isinstance(loss_cls, dict):
            losses.update(loss_cls)
        else:
            losses['loss_cls'] = loss_cls

        return losses
