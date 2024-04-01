import os
import torch


def get_config():
    config = {
        'debug': False,
        'step_size': 6,
        'num_workers': 2,
        'n_channels': 32,
        'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        'batch_size': 128,
        'epochs': 20,
        'lr': 0.01,
        'lr_scheduler': 'step_lr',
        'dropout': 0.1,
        'norm': 'bn',
        'classes': ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'),
    }
    return config