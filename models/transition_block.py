import torch
import torch.nn as nn
import torch.nn.functional as F

class TransitionBlock(nn.Module):
    def __init__(self, in_channels, out_channels,kernel_size=(1,1), **kwargs):
        super().__init__()
        
        self.conv1d = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, bias=False, **kwargs),
        )
        
    def forward(self, x):
        x = self.conv1d(x)
        return x