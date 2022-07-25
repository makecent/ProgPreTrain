Note that the below setting is used and maybe not the dafault:
    dict(type='RandomRescale', scale_range=(256, 320)),
    dict(type='RandomCrop', size=224),

1. Comparing the performance 8x4, 8x8, 8x32, 8x8 get the best accuracy. 8x32 is too long for sthv2 because the average duration of sthv2 is about 4.03 seconds (fps=12, about 48.36 frames)

2. The original paper does NOT use RandomFlip for the sthv2. We use RandFlip (with flip_label_map) and it slightly improves the performance.
