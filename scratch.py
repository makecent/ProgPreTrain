from pytorchvideo.models import create_multiscale_vision_transformers
import torch

spatial_size = 224
temporal_size = 16
embed_dim_mul = [[1, 2.0], [3, 2.0], [14, 2.0]]
atten_head_mul = [[1, 2.0], [3, 2.0], [14, 2.0]]
pool_q_stride_size = [[1, 1, 2, 2], [3, 1, 2, 2], [14, 1, 2, 2]]
pool_kv_stride_adaptive = [1, 8, 8]
pool_kvq_kernel = [3, 3, 3]
head_num_classes = 400
MViT_B = create_multiscale_vision_transformers(
    spatial_size=spatial_size,
    temporal_size=temporal_size,
    embed_dim_mul=embed_dim_mul,
    atten_head_mul=atten_head_mul,
    pool_q_stride_size=pool_q_stride_size,
    pool_kv_stride_adaptive=pool_kv_stride_adaptive,
    pool_kvq_kernel=pool_kvq_kernel,
    head_num_classes=head_num_classes)

x = torch.randn(1, 3, 16, 224, 224)
y = MViT_B(x)
