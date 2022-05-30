import numpy as np
from mmaction.datasets.builder import DATASETS
from mmaction.datasets.video_dataset import VideoDataset
from mmcv.utils import print_log


@DATASETS.register_module()
class CustomVideoDataset(VideoDataset):
    def evaluate(self, results, **kwargs):
        if 'MAE' in kwargs.get('metrics', []):
            kwargs['metrics'].remove('MAE')
            logger = kwargs.get('logger', None)
            eval_results = VideoDataset.evaluate(results, **kwargs)

            msg = f'Evaluating MAE ...'
            if logger is None:
                msg = '\n' + msg
            print_log(msg, logger=logger)

            MAE = np.array(results).mean()
            eval_results[f'MAE'] = MAE
            print_log(f'\nMAE\t{MAE:.2f}', logger=logger)
        else:
            eval_results = VideoDataset.evaluate(results, **kwargs)

        return eval_results
