import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from config import get_config
# import sys
# # add parent directory to path
# sys.path.append('/Users/aakashvardhan/Library/CloudStorage/GoogleDrive-vardhan.aakash1@gmail.com/My Drive/ERA v2/s8-normalization/config.py')


GROUP_SIZE_GN = 2
GROUP_SIZE_LN = 1
config = get_config()

class LayerNorm(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.layer_norm = nn.GroupNorm(num_groups=GROUP_SIZE_LN, num_channels=num_features)
        
    def forward(self, x):
        return self.layer_norm(x)


class DepthwiseSeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3,3), padding=0, bias=False, **kwargs):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=padding, groups=in_channels, bias=bias, **kwargs)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=(1,1), bias=bias, **kwargs)
        
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels,norm, kernel_size=(3,3),dropout_value=0,dilation=1, depthwise_seperable=False, **kwargs):
        super().__init__()
        
        if norm == 'bn':
            self.norm = lambda num_features: nn.BatchNorm2d(num_features)
        elif norm == 'gn':
            self.norm = lambda num_features: nn.GroupNorm(GROUP_SIZE_GN, num_features)
        elif norm == 'ln':
            self.norm = lambda num_features: LayerNorm(num_features)
        else:
            raise ValueError('Norm type {} not supported'.format(norm))
        
        if depthwise_seperable:
            self.conv = DepthwiseSeparableConv2d(in_channels, out_channels, kernel_size=kernel_size, padding=1, bias=False,dilation=dilation, **kwargs)
        else:
            self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,dilation=dilation,bias=False, **kwargs),
        
        self.convblock = nn.Sequential(
            self.conv,
            nn.ReLU(),
            self.norm(out_channels),
            nn.Dropout(dropout_value)
        )
        
    def forward(self, x):
        return self.convblock(x)