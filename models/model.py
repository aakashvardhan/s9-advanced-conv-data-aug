import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from models.conv_block import ConvBlock
from models.transition_block import TransitionBlock

torch.manual_seed(1)

class Net(nn.Module):
    def __init__(self,config):
        super().__init__()
        n_channels = config['n_channels']
        dropout = config['dropout']
        norm = config['norm']
        
        # Convolution Block 1
        self.conv1 = ConvBlock(in_channels=3, out_channels=n_channels // 4, norm=norm, padding=1) # output_size = 32, RF = 3
        self.conv2 = ConvBlock(in_channels=n_channels // 4, out_channels=n_channels // 2, norm=norm, padding=1) # output_size = 32, RF = 5
        self.conv3 = ConvBlock(in_channels=n_channels // 2, out_channels=n_channels, norm=norm, padding=0,dilation=2) # output_size = 30, RF = 9
        
        # Transition Block 1
        self.trans1 = TransitionBlock(in_channels=n_channels, out_channels=n_channels // 4, padding=0) # output_size = 30, RF = 9
        
        # Convolution Block 2
        self.conv4 = ConvBlock(in_channels=n_channels // 4, out_channels=n_channels // 2, norm=norm, padding=1) # output_size = 30, RF = 13
        self.conv5 = ConvBlock(in_channels=n_channels // 2, out_channels=n_channels, norm=norm, padding=1) # output_size = 30, RF = 17
        self.conv6 = ConvBlock(in_channels=n_channels, out_channels=n_channels, norm=norm, padding=0,dilation=4) # output_size = 26, RF = 25
        
        # Transition Block 2
        self.trans2 = TransitionBlock(in_channels=n_channels, out_channels=n_channels // 4, padding=0) # output_size = 26, RF = 25
        
        # Convolution Block 3
        self.conv7 = ConvBlock(in_channels=n_channels // 4, out_channels=n_channels // 2, norm=norm, padding=1) # output_size = 26, RF = 33 
        self.conv8 = ConvBlock(in_channels=n_channels // 2, out_channels=n_channels, norm=norm, padding=1) # output_size = 26, RF = 41
        self.conv9 = ConvBlock(in_channels=n_channels, out_channels=n_channels, norm=norm, padding=0,dilation=8) # output_size = 18, RF = 57
        
        # Transition Block 3
        self.trans3 = TransitionBlock(in_channels=n_channels, out_channels=n_channels // 4, padding=0) # output_size = 18, RF = 57
        
        # Convolution Block 4 (with Depthwise Separable Convolution)
        self.conv10 = ConvBlock(in_channels=n_channels // 4, out_channels=n_channels // 2, norm=norm, padding=1, depthwise_seperable=True) # output_size = 18, RF = 75 
        self.conv11 = ConvBlock(in_channels=n_channels // 2, out_channels=n_channels, norm=norm, padding=1) # output_size = 18, RF = 93
        self.conv12 = ConvBlock(in_channels=n_channels, out_channels=n_channels, norm=norm, padding=0) # output_size = 16, RF = 109
        
        # Output Block
        self.gap = nn.AdaptiveAvgPool2d(1) # output_size = 1, RF = 125
        self.fc = nn.Linear(32, 10) # output_size = 1, RF = 125
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x3 = self.conv3(x)
        x = self.trans1(x3)
        x = self.conv4(x)
        x = self.conv5(x) + x3
        x6 = self.conv6(x)
        x = self.trans2(x6)
        x = self.conv7(x)
        x = self.conv8(x) + x6
        x9 = self.conv9(x)
        x = self.trans3(x9)
        x = self.conv10(x)
        x = self.conv11(x) + x9
        x = self.conv12(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return F.log_softmax(x, dim=-1)