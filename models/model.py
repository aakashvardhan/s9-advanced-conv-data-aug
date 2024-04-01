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
        
        self.config = config
        
        # Convolution Block 1
        self.conv1 = ConvBlock(in_channels=3, out_channels=n_channels // 2, norm=norm, padding=3,kernel_size=(7,7)) # output_size = 32, RF = 7
        self.conv2 = ConvBlock(in_channels=n_channels // 2, out_channels=n_channels // 2, norm=norm, padding=1) # output_size = 32, RF = 7
        self.conv3 = ConvBlock(in_channels=n_channels // 2, out_channels=n_channels, norm=norm, padding=0,dilation=2) # output_size = 30, RF = 9
        
        # Transition Block 1
        self.trans1 = TransitionBlock(in_channels=n_channels, out_channels=n_channels // 2, padding=0) # output_size = 30, RF = 9
        
        # Convolution Block 2
        self.conv4 = ConvBlock(in_channels=n_channels // 2, out_channels=n_channels, norm=norm, padding=1,dropout_value=dropout) # output_size = 30, RF = 17
        self.conv5 = ConvBlock(in_channels=n_channels, out_channels=n_channels, norm=norm, padding=1,dropout_value=dropout) # output_size = 30, RF = 25
        self.conv6 = ConvBlock(in_channels=n_channels, out_channels=n_channels, norm=norm, padding=0,dilation=4,dropout_value=dropout) # output_size = 26, RF = 33
        
        # Transition Block 2
        self.trans2 = TransitionBlock(in_channels=n_channels, out_channels=n_channels // 2, padding=0) # output_size = 26, RF = 33
        
        # Convolution Block 3 (with Depthwise Separable Convolution)
        self.conv7 = ConvBlock(in_channels=n_channels // 2, out_channels=n_channels, norm=norm, padding=1,dropout_value=dropout,depthwise_seperable=True) # output_size = 26, RF = 41
        self.conv8 = ConvBlock(in_channels=n_channels, out_channels=n_channels, norm=norm, padding=1,dropout_value=dropout) # output_size = 26, RF = 49
        self.conv9 = ConvBlock(in_channels=n_channels, out_channels=n_channels, norm=norm, padding=0,dilation=8,dropout_value=dropout) # output_size = 18, RF = 57
        
        # Transition Block 3
        self.trans3 = TransitionBlock(in_channels=n_channels, out_channels=n_channels // 2, padding=0) # output_size = 18, RF = 57
        
        # Convolution Block 4 (with Depthwise Separable Convolution)
        self.conv10 = ConvBlock(in_channels=n_channels // 2, out_channels=n_channels, norm=norm, padding=1, depthwise_seperable=True) # output_size = 18, RF = 75
        self.conv11 = ConvBlock(in_channels=n_channels, out_channels=n_channels, norm=norm, padding=1) # output_size = 18, RF = 93
        self.conv12 = ConvBlock(in_channels=n_channels, out_channels=n_channels, norm=norm, padding=0) # output_size = 16, RF = 109
        
        # Output Block
        self.gap = nn.AdaptiveAvgPool2d((1,1)) # output_size = 1, RF = 125
        self.fc = None
    def forward(self, x):
        x = self.conv1(x)
        if self.config['debug']:
            print("After Conv1: ", x.shape)
        x = x + self.conv2(x)
        if self.config['debug']:
            print("After Conv2: ", x.shape)
        x3 = self.conv3(x)
        if self.config['debug']:
            print("After Conv3: ", x3.shape)
        x = self.trans1(x3)
        if self.config['debug']:
            print("After Transition1: ", x.shape)
        x = self.conv4(x)
        if self.config['debug']:
            print("After Conv4: ", x.shape)
        x = x + self.conv5(x)
        if self.config['debug']:
            print("After Conv5: ", x.shape)
        x6 = self.conv6(x)
        if self.config['debug']:
            print("After Conv6: ", x6.shape)
        x = self.trans2(x6)
        if self.config['debug']:
            print("After Transition2: ", x.shape)
        x = self.conv7(x)
        if self.config['debug']:
            print("After Conv7: ", x.shape)
        x = x + self.conv8(x)
        if self.config['debug']:
            print("After Conv8: ", x.shape)
        x9 = self.conv9(x)
        if self.config['debug']:
            print("After Conv9: ", x9.shape)
        x = self.trans3(x9)
        if self.config['debug']:
            print("After Transition3: ", x.shape)
        x = self.conv10(x)
        if self.config['debug']:
            print("After Conv10: ", x.shape)
        x = x + self.conv11(x)
        if self.config['debug']:
            print("After Conv11: ", x.shape)
        x = self.conv12(x)
        if self.config['debug']:
            print("After Conv12: ", x.shape)
        x = self.gap(x)
        if self.config['debug']:
            print("After GAP: ", x.shape)
        
        if self.fc is None:
            in_features = x.shape[1] * x.shape[2] * x.shape[3]
            self.fc = nn.Linear(in_features, 10).to(x.device)
        
        x = x.view(x.size(0), -1)
        if self.config['debug']:
            print("After View: ", x.shape)
        x = self.fc(x)
        if self.config['debug']:
            print("After FC: ", x.shape)
        return F.log_softmax(x, dim=-1)
    
    
