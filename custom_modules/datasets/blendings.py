from mmaction.datasets import BaseMiniBatchBlending, BLENDINGS
from mmcv.utils import build_from_cfg
import torch
from torch.distributions.beta import Beta


@BLENDINGS.register_module()
class BatchAugBlending:
    """Implementing
        https://openaccess.thecvf.com/content_CVPR_2020/papers/Hoffer_Augment_Your_Batch_Improving_Generalization_Through_Instance_Repetition_CVPR_2020_paper.pdf
        Only support repeated blending.
    """

    def __init__(self,
                 blendings=(dict(type='MixupBlending', num_classes=400, alpha=.8),
                            dict(type='CutmixBlending', num_classes=400, alpha=1.))):
        self.blendings = [build_from_cfg(bld, BLENDINGS) for bld in blendings]

    def __call__(self, imgs, label):
        repeated_imgs = []
        repeated_label = []

        for bld in self.blendings:
            mixed_imgs, mixed_label = bld(imgs, label)
            repeated_imgs.append(mixed_imgs)
            repeated_label.append(mixed_label)
        return torch.cat(repeated_imgs), torch.cat(repeated_label)


# Modify Cutmix to avoid __floordiv__ warning
@BLENDINGS.register_module(force=True)
class CutmixBlending(BaseMiniBatchBlending):
    """Implementing Cutmix in a mini-batch.

    This module is proposed in `CutMix: Regularization Strategy to Train Strong
    Classifiers with Localizable Features <https://arxiv.org/abs/1905.04899>`_.
    Code Reference https://github.com/clovaai/CutMix-PyTorch

    Args:
        num_classes (int): The number of classes.
        alpha (float): Parameters for Beta distribution.
    """

    def __init__(self, num_classes, alpha=.2):
        super().__init__(num_classes=num_classes)
        self.beta = Beta(alpha, alpha)

    @staticmethod
    def rand_bbox(img_size, lam):
        """Generate a random boudning box."""
        w = img_size[-1]
        h = img_size[-2]
        cut_rat = torch.sqrt(1. - lam)
        cut_w = torch.tensor(int(w * cut_rat))
        cut_h = torch.tensor(int(h * cut_rat))

        # uniform
        cx = torch.randint(w, (1,))[0]
        cy = torch.randint(h, (1,))[0]

        bbx1 = torch.clamp(cx - torch.div(cut_w, 2, rounding_mode='floor'), 0, w)
        bby1 = torch.clamp(cy - torch.div(cut_h, 2, rounding_mode='floor'), 0, h)
        bbx2 = torch.clamp(cx + torch.div(cut_w, 2, rounding_mode='floor'), 0, w)
        bby2 = torch.clamp(cy + torch.div(cut_h, 2, rounding_mode='floor'), 0, h)

        return bbx1, bby1, bbx2, bby2

    def do_blending(self, imgs, label, **kwargs):
        """Blending images with cutmix."""
        assert len(kwargs) == 0, f'unexpected kwargs for cutmix {kwargs}'

        batch_size = imgs.size(0)
        rand_index = torch.randperm(batch_size)
        lam = self.beta.sample()

        bbx1, bby1, bbx2, bby2 = self.rand_bbox(imgs.size(), lam)
        imgs[:, ..., bby1:bby2, bbx1:bbx2] = imgs[rand_index, ..., bby1:bby2,
                                             bbx1:bbx2]
        lam = 1 - (1.0 * (bbx2 - bbx1) * (bby2 - bby1) /
                   (imgs.size()[-1] * imgs.size()[-2]))

        label = lam * label + (1 - lam) * label[rand_index, :]

        return imgs, label
