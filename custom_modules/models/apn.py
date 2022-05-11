from mmaction.models.builder import LOCALIZERS, build_backbone, build_head
from mmaction.models.localizers.base import BaseTAGClassifier


@LOCALIZERS.register_module()
class APN(BaseTAGClassifier):
    """APN model framework."""

    def _forward(self, imgs):
        # [N, num_clips, C, T, H, W] -> [N*num_clips, C, T, H, W], which make clips training parallely (For TSN).
        # For 2D backbone, there is no 'T' dimension. For our APN, num_clips is always equal to 1.
        imgs = imgs.reshape((-1,) + imgs.shape[2:])
        x = self.extract_feat(imgs)
        output = self.cls_head(x)

        return output

    def forward_train(self, imgs, prog_labels=None):
        output = self._forward(imgs)
        losses = {'loss': self.cls_head.loss(output, prog_labels)}
        return losses

    def forward_test(self, imgs, prog_labels):
        output = self._forward(imgs)
        progs = self.cls_head.decode_output(output)
        mae = (progs - prog_labels.squeeze()).abs()
        return mae.cpu().numpy()
