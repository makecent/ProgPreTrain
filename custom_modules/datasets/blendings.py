from mmaction.datasets.builder import BLENDINGS
from mmcv.utils import build_from_cfg
import torch


@BLENDINGS.register_module()
class BatchAugBlending:
    """Implementing
        https://openaccess.thecvf.com/content_CVPR_2020/papers/Hoffer_Augment_Your_Batch_Improving_Generalization_Through_Instance_Repetition_CVPR_2020_paper.pdf
        Only support repeated blending.
    """

    def __init__(self,
                 blendings=(dict(type='MixupBlending', num_classes=400, alpha=.8),
                            dict(type='CutupBlending', num_classes=400, alpha=.2))):
        self.blendings = (build_from_cfg(bld, BLENDINGS) for bld in blendings)

    def __call__(self, imgs, label):
        repeated_imgs = []
        repeated_label = []

        for bld in self.blendings:
            mixed_imgs, mixed_label = bld(imgs, label)
            repeated_imgs.append(mixed_imgs)
            repeated_label.append(mixed_label)

        return torch.cat(repeated_imgs), torch.cat(repeated_label)
