# %% fvcore
import torch
from fvcore.nn import FlopCountAnalysis, parameter_count, flop_count_table
from mmcv import Config
from mmaction.models.builder import build_model, build_backbone

inputs = torch.randn(1, 3, 8, 224, 224)
# model = build_backbone(Config.fromfile(
#     "configs/timesformer/timesformer_divST_8x4x1_15e_sthv2_rgb.py").model.backbone)
# model = build_backbone(Config.fromfile(
#     "configs/timesformer/in_proj/timesformer_decST_no_inproj_8x4_1_15e_sthv2_rgb.py").model.backbone)
model = build_backbone(Config.fromfile(
    "configs/timesformer/timesformer_jointST_equalflop_8x4x1_15e_sthv2_rgb.py").model.backbone)

# inputs = (torch.randn(1, 1, 3, 16, 224, 224), torch.randint(10, (1, 1)))
# model = build_model(Config.fromfile("configs/mvit/mvit_16x4_kinetics400_video.py").model)

flops = FlopCountAnalysis(model, inputs)
print(flop_count_table(flops, max_depth=10))
params = parameter_count(model)

print(f"GFLOPS:\t{flops.total()/1e9:.2f} G")
print(f"Params:\t{params['']/1e6:.2f} M")