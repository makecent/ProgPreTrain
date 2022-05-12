import custom_modules
from mmaction.models.builder import build_model
from mmaction.datasets.builder import build_dataset
from mmcv import Config
import torch
cfg = Config.fromfile("configs/mvit/mvit_16x4_kinetics400_video.py")

model = build_model(cfg.model)
dataset = build_dataset(cfg.data.train)
input = dataset[0]
y = model(imgs=input['imgs'][None, :], label=input['label'][None, :])

print("fi")
