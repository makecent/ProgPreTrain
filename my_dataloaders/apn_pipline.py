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
        """Convert progression_label to Ordinal matrix.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        clip_center = results['frame_inds'].mean()
        prog_label = round(clip_center / results['total_frames'] * self.num_stages)
        if self.ordinal:
            prog_label = np.full(self.num_stages, fill_value=0.0, dtype='float32')
            prog_label[:prog_label] = 1.0
        else:
            prog_label = float(prog_label)
        results['prog_labels'] = prog_label
        return results
