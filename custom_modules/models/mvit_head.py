from mmaction.models.builder import HEADS
from mmaction.models.heads import BaseHead
from pytorchvideo.models.head import SequencePool
from torch import nn
from mmcv.cnn import normal_init


@HEADS.register_module()
class MViTHead(BaseHead):
    def __init__(self, dropout_rate=0.5, seq_pool_type="cls", init_std=0.01, *args, **kwargs):
        super(MViTHead, self).__init__(*args, **kwargs)
        self.seq_pool_type = seq_pool_type
        self.sequence_pool = SequencePool(seq_pool_type)
        self.proj = nn.Linear(self.in_channels, self.num_classes)
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0.0 else None
        self.init_std = init_std

    def forward(self, x):
        # Performs pooling.
        if self.sequence_pool is not None:
            x = self.sequence_pool(x)

        # Performs dropout.
        if self.dropout is not None:
            x = self.dropout(x)
        # Performs projection.
        x = self.proj(x)
        return x

    def init_weights(self):
        normal_init(self.proj, std=self.init_std)
