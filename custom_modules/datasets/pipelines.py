import numpy as np
from mmaction.datasets.builder import PIPELINES
import warnings
from mmaction.datasets.pipelines.augmentations import ThreeCrop as _ThreeCrop
from mmaction.datasets.pipelines.augmentations import _init_lazy_if_proper


# @PIPELINES.register_module()
# class ProgLabel:
#
#     def __init__(self,
#                  num_stages=10,
#                  ordinal=False):
#         self.num_stages = num_stages
#         self.ordinal = ordinal
#
#     def __call__(self, results):
#         clip_center = results['frame_inds'].reshape([results['num_clips'], results['clip_len']]).mean(axis=-1)
#         prog_label = np.rint(clip_center / results['total_frames'] * self.num_stages)
#         if self.ordinal:
#             assert results['num_clips'] == 1
#             prog_label = prog_label.astype(int)[0]
#             ordinal_label = np.full(self.num_stages, fill_value=0.0, dtype='float32')
#             ordinal_label[:prog_label] = 1.0
#             results['prog_label'] = ordinal_label
#         else:
#             results['prog_label'] = prog_label / self.num_stages * 100
#         return results


@PIPELINES.register_module()
class ProgLabel:

    def __init__(self, num_stages=100):
        self.num_stages = num_stages

    @staticmethod
    def _get_prog(results):
        assert results['num_clips'] == 1, "progression label should only used in training"
        clip_center = results['frame_inds'].mean()
        prog = np.clip(clip_center / results['total_frames'], a_min=0, a_max=1)
        return prog

    def __call__(self, results):
        """Convert progression_label to ordinal label. e.g., 0.031 => [1, 1, 1, 0, ...]."""
        ordinal_label = np.full(self.num_stages, fill_value=0.0, dtype='float32')
        prog = results['progression_label'] if 'progression_label' in results else self._get_prog(results)

        denormalized_prog = round(prog * self.num_stages)
        ordinal_label[:denormalized_prog] = 1.0
        results['prog_label'] = ordinal_label
        return results


@PIPELINES.register_module()
class CenterLabel(ProgLabel):
    def __call__(self, results):

        clip_center = results['frame_inds'].mean()
        video_center = results['total_frames'] / 2
        centerness = 1 - 2 * abs(clip_center - video_center) / results['total_frames']
        centerness = np.clip(centerness, a_min=0, a_max=1)
        prog_label = round(centerness * self.num_stages)
        if self.ordinal:
            ordinal_label = np.full(self.num_stages, fill_value=0.0, dtype='float32')
            ordinal_label[:prog_label] = 1.0
        results['prog_label'] = ordinal_label if self.ordinal else float(prog_label)
        return results


@PIPELINES.register_module(force=True)
class ThreeCrop(_ThreeCrop):
    """
    Support short-side center crop now;
    Repeat progression label three times;
    """
    def __call__(self, results):
        _init_lazy_if_proper(results, False)
        if 'gt_bboxes' in results or 'proposals' in results:
            warnings.warn('ThreeCrop cannot process bounding boxes')

        imgs = results['imgs']
        img_h, img_w = results['imgs'][0].shape[:2]
        crop_w, crop_h = self.crop_size
        assert crop_h <= img_h and crop_w <= img_w

        if img_w >= img_h:
            w_step = (img_w - crop_w) // 2
            h_offset = (img_h - crop_h) // 2
            offsets = [
                (0, h_offset),  # left
                (2 * w_step, h_offset),  # right
                (w_step, h_offset),  # middle
            ]
        else:
            h_step = (img_h - crop_h) // 2
            w_offset = (img_w - crop_w) // 2
            offsets = [
                (w_offset, 0),  # top
                (w_offset, 2 * h_step),  # down
                (w_offset, h_step),  # middle
            ]

        cropped = []
        crop_bboxes = []
        for x_offset, y_offset in offsets:
            bbox = [x_offset, y_offset, x_offset + crop_w, y_offset + crop_h]
            crop = [
                img[y_offset:y_offset + crop_h, x_offset:x_offset + crop_w]
                for img in imgs
            ]
            cropped.extend(crop)
            crop_bboxes.extend([bbox for _ in range(len(imgs))])

        crop_bboxes = np.array(crop_bboxes)
        results['imgs'] = cropped
        results['crop_bbox'] = crop_bboxes
        results['img_shape'] = results['imgs'][0].shape[:2]
        if 'prog_label' in results:
            results['prog_label'] = np.repeat(results['prog_label'], 3)

        return results

    def __repr__(self):
        repr_str = f'{self.__class__.__name__}(crop_size={self.crop_size})'
        return repr_str