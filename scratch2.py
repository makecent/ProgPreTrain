from mmaction.datasets import build_dataset, build_dataloader
from mmcv.utils import Config


cfg = Config.fromfile("configs/mvit/mvit-s_16x4_kinetics400_video.py")
datasets = build_dataset(cfg.data.train)
dataloader_setting = dict(
    videos_per_gpu=cfg.data.get('videos_per_gpu', 1),
    workers_per_gpu=cfg.data.get('workers_per_gpu', 1),
    persistent_workers=cfg.data.get('persistent_workers', False))
dataloader_setting = dict(dataloader_setting,
                          **cfg.data.get('train_dataloader', {}))
data_loader = iter(build_dataloader(datasets, **dataloader_setting))
samples1 = [next(data_loader) for i in range(240618)]
samples2 = [next(data_loader) for i in range(240618)]
print('s')