import numpy as np
from mmaction.datasets.builder import DATASETS
from mmaction.datasets.video_dataset import VideoDataset
from mmcv.utils import print_log


@DATASETS.register_module()
class CustomVideoDataset(VideoDataset):
    def evaluate(self,
                 results,
                 metrics='MAE',
                 metric_options=None,
                 logger=None,
                 **deprecated_kwargs):
        eval_results = dict()

        msg = f'Evaluating MAE ...'
        if logger is None:
            msg = '\n' + msg
        print_log(msg, logger=logger)

        MAE = np.array(results).mean()
        eval_results[f'MAE'] = MAE
        print_log(f'\nMAE\t{MAE:.2f}', logger=logger)

        return eval_results
