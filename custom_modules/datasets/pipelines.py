import numpy as np
from mmaction.datasets.builder import PIPELINES


@PIPELINES.register_module()
class ProgLabel:

    def __init__(self,
                 num_stages=100,
                 ordinal=False):
        self.num_stages = num_stages
        self.ordinal = ordinal

    def __call__(self, results):
        clip_center = results['frame_inds'].mean()
        prog_label = round(clip_center / results['total_frames'] * self.num_stages)
        if self.ordinal:
            ordinal_label = np.full(self.num_stages, fill_value=0.0, dtype='float32')
            ordinal_label[:prog_label] = 1.0
        results['prog_labels'] = ordinal_label if self.ordinal else float(prog_label)
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
        results['prog_labels'] = ordinal_label if self.ordinal else float(prog_label)
        return results