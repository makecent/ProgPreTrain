# imports
custom_imports = dict(imports=['my_models', 'my_dataloaders'], allow_failed_imports=False)

# others
checkpoint_config = dict(interval=5)
log_config = dict(interval=1, hooks=[dict(type='TensorboardLoggerHook'), dict(type='TextLoggerHook')])

# runtime settings
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]

# disable opencv multithreading to avoid system being overloaded
opencv_num_threads = 0
# set multi-process start method as `fork` to speed up the training
mp_start_method = 'fork'



