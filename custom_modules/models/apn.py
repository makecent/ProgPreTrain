import torch
from torch.nn import functional as F
import numpy as np

from mmaction.models.builder import LOCALIZERS, build_backbone, build_head
from mmaction.models.localizers import BaseTAGClassifier
from mmaction.core import top_k_accuracy


def decode_progression(reg_score):
    num_stage = reg_score.shape[-1]
    if isinstance(reg_score, torch.Tensor):
        progression = torch.count_nonzero(reg_score > 0.5, dim=-1)
    elif isinstance(reg_score, np.ndarray):
        progression = np.count_nonzero(reg_score > 0.5, axis=-1)
    else:
        raise TypeError(f"unsupported reg_score type: {type(reg_score)}")
    progression = progression * 100 / num_stage
    return progression


def progression_mae(reg_score, progression_label):
    progression = decode_progression(reg_score)
    if progression_label.ndim > 2:
        progression_label = decode_progression(progression_label)
    if isinstance(reg_score, torch.Tensor):
        mae = torch.abs(progression - progression_label)
    elif isinstance(reg_score, np.ndarray):
        mae = np.abs(progression - progression_label)
    else:
        raise TypeError(f"unsupported reg_score type: {type(reg_score)}")
    return mae


def binary_accuracy(pred, label):
    acc = np.count_nonzero((pred > 0.5) == label) / label.size
    return acc


@LOCALIZERS.register_module()
class APN(BaseTAGClassifier):
    """APN model framework."""

    def __init__(self,
                 backbone,
                 cls_head,
                 train_cfg=None,
                 test_cfg=None):
        super(BaseTAGClassifier, self).__init__()
        self.backbone = build_backbone(backbone)
        self.cls_head = build_head(cls_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.init_weights()

    def _forward(self, imgs):
        # [N, S, C, T, H, W] -> [N*S, C, T, H, W]
        imgs = imgs.reshape((-1,) + imgs.shape[2:])

        x = self.extract_feat(imgs)
        output = self.cls_head(x)

        return output

    def forward_train(self, imgs, prog_label=None, label=None):
        cls_score, reg_score = self._forward(imgs)
        losses = {'loss_cls': self.cls_head.loss_cls(cls_score, label.squeeze(-1)),
                  'loss_reg': self.cls_head.loss_reg(reg_score, prog_label)}

        cls_acc = top_k_accuracy(cls_score.detach().cpu().numpy(),
                                 label.detach().cpu().numpy(),
                                 topk=(1,))
        reg_score = reg_score.sigmoid()
        reg_acc = binary_accuracy(reg_score.detach().cpu().numpy(), prog_label.detach().cpu().numpy())
        reg_mae = progression_mae(reg_score.detach().cpu().numpy(), prog_label.detach().cpu().numpy())
        losses[f'top1_acc'] = torch.tensor(cls_acc, device=cls_score.device)
        losses[f'reg_acc'] = torch.tensor(reg_acc, device=reg_score.device)
        losses[f'reg_mae'] = torch.tensor(reg_mae.mean(), device=reg_score.device)

        return losses

    def forward_test(self, imgs, prog_label):
        """Defines the computation performed at every call when evaluation and testing."""
        cls_score, reg_score = self._forward(imgs)
        reg_score = reg_score.sigmoid()
        reg_mae = progression_mae(reg_score, prog_label.flatten())

        cls_score = self.average_clip(cls_score, num_segs=imgs.shape[1])
        reg_mae = self.average_clip(reg_mae, num_segs=imgs.shape[1])
        return list(zip(cls_score.cpu().numpy(), reg_mae.cpu().numpy()))

    def average_clip(self, cls_score, num_segs=1):
        average_clips = self.test_cfg['average_clips']
        batch_size = cls_score.shape[0]
        cls_score = cls_score.view(batch_size // num_segs, num_segs, -1)

        if average_clips == 'prob':
            cls_score = F.softmax(cls_score, dim=2).mean(dim=1)
        elif average_clips == 'score':
            cls_score = cls_score.mean(dim=1)

        return cls_score
