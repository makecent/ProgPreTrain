import torch
from mmaction.datasets.pipelines import SampleFrames, DenseSampleFrames
# dummy_video = torch.randn(256, 224, 224)

# s1 = SampleFrames(clip_len=16, frame_interval=2, num_clips=10, test_mode=True)
s1 = SampleFrames(clip_len=16, frame_interval=2, num_clips=1)
v = {'filename': 'dummy',
     'total_frames': 256,
     'start_index': 0}
results = s1(v)
print('haha')