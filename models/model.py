import torch
import torch.nn as nn
import torch.nn.functional as F
from models.conv_block import ConvBlock
from models.transition_block import TransitionBlock

torch.manual_seed(1)


class Net(nn.Module):
    """
    A convolutional neural network model.

    Args:
        config (dict): Configuration parameters for the model.

    Attributes:
        conv1 (ConvBlock): Convolutional block 1.
        conv2 (ConvBlock): Convolutional block 2.
        conv3 (ConvBlock): Convolutional block 3.
        trans1 (TransitionBlock): Transition block 1.
        conv4 (ConvBlock): Convolutional block 4.
        conv5 (ConvBlock): Convolutional block 5.
        conv6 (ConvBlock): Convolutional block 6.
        trans2 (TransitionBlock): Transition block 2.
        conv7 (ConvBlock): Convolutional block 7.
        conv8 (ConvBlock): Convolutional block 8.
        conv9 (ConvBlock): Convolutional block 9.
        trans3 (TransitionBlock): Transition block 3.
        conv10 (ConvBlock): Convolutional block 10.
        conv11 (ConvBlock): Convolutional block 11.
        conv12 (ConvBlock): Convolutional block 12.
        gap (nn.AdaptiveAvgPool2d): Global average pooling layer.
        fc (nn.Linear): Fully connected layer.

    """

    def __init__(self, config):
        """
        Initializes the Net class.

        Args:
            config (dict): Configuration parameters for the model.

        """
        super().__init__()
        n_channels = config["n_channels"]
        dropout = config["dropout"]
        norm = config["norm"]

        self.config = config

        # Convolution Block 1
        self.conv1 = ConvBlock(
            in_channels=3, out_channels=n_channels // 2, norm=norm, padding=1
        )  # output_size = 32, RF = 3
        self.conv2 = ConvBlock(
            in_channels=n_channels // 2,
            out_channels=n_channels // 2,
            norm=norm,
            padding=1,
        )  # output_size = 32, RF = 5
        self.conv3 = ConvBlock(
            in_channels=n_channels // 2,
            out_channels=n_channels,
            norm=norm,
            padding=0,
            stride=2,
        )  # output_size = 30, RF = 7

        # Transition Block 1
        self.trans1 = TransitionBlock(
            in_channels=n_channels, out_channels=n_channels // 2, padding=0
        )  # output_size = 15, RF = 9

        # Convolution Block 2
        self.conv4 = ConvBlock(
            in_channels=n_channels // 2,
            out_channels=n_channels,
            norm=norm,
            padding=1,
            dropout_value=dropout,
        )  # output_size = 15, RF = 13
        self.conv5 = ConvBlock(
            in_channels=n_channels,
            out_channels=n_channels,
            norm=norm,
            padding=1,
            dropout_value=dropout,
        )  # output_size = 15, RF = 17
        self.conv6 = ConvBlock(
            in_channels=n_channels,
            out_channels=n_channels,
            norm=norm,
            padding=0,
            dilation=2,
            dropout_value=dropout,
        )  # output_size = 13, RF = 21

        # Transition Block 2
        self.trans2 = TransitionBlock(
            in_channels=n_channels, out_channels=n_channels // 2, padding=0
        )  # output_size = 13, RF = 25

        # Convolution Block 3 (with Depthwise Separable Convolution)
        self.conv7 = ConvBlock(
            in_channels=n_channels // 2,
            out_channels=n_channels,
            norm=norm,
            padding=1,
            dropout_value=dropout,
            depthwise_seperable=True,
        )  # output_size = 13, RF = 33
        self.conv8 = ConvBlock(
            in_channels=n_channels,
            out_channels=n_channels,
            norm=norm,
            padding=1,
            dropout_value=dropout,
        )  # output_size = 13, RF = 41
        self.conv9 = ConvBlock(
            in_channels=n_channels,
            out_channels=n_channels,
            norm=norm,
            padding=0,
            dilation=4,
            dropout_value=dropout,
        )  # output_size = 9, RF = 49

        # Transition Block 3
        self.trans3 = TransitionBlock(
            in_channels=n_channels, out_channels=n_channels // 2, padding=0
        )  # output_size = 9, RF = 57

        # Convolution Block 4 (with Depthwise Separable Convolution)
        self.conv10 = ConvBlock(
            in_channels=n_channels // 2,
            out_channels=n_channels,
            norm=norm,
            padding=1,
            depthwise_seperable=True,
        )  # output_size = 9, RF = 77
        self.conv11 = ConvBlock(
            in_channels=n_channels, out_channels=n_channels, norm=norm, padding=1
        )  # output_size = 9, RF = 93
        self.conv12 = ConvBlock(
            in_channels=n_channels, out_channels=n_channels, norm=norm, padding=0
        )  # output_size = 7, RF = 109

        # Output Block
        self.gap = nn.AdaptiveAvgPool2d((1, 1))  # output_size = 1, RF = 125
        self.fc = None  # output_size = 1, RF = 125

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after passing through the model.
        """
        x = self.conv1(x)
        if self.config["debug"]:
            print("After Conv1: ", x.shape)
        x = x + self.conv2(x)
        if self.config["debug"]:
            print("After Conv2: ", x.shape)
        x3 = self.conv3(x)
        if self.config["debug"]:
            print("After Conv3: ", x3.shape)
        x = self.trans1(x3)
        if self.config["debug"]:
            print("After Transition1: ", x.shape)
        x = self.conv4(x)
        if self.config["debug"]:
            print("After Conv4: ", x.shape)
        x = x + self.conv5(x)
        if self.config["debug"]:
            print("After Conv5: ", x.shape)
        x6 = self.conv6(x)
        if self.config["debug"]:
            print("After Conv6: ", x6.shape)
        x = self.trans2(x6)
        if self.config["debug"]:
            print("After Transition2: ", x.shape)
        x = self.conv7(x)
        if self.config["debug"]:
            print("After Conv7: ", x.shape)
        x = x + self.conv8(x)
        if self.config["debug"]:
            print("After Conv8: ", x.shape)
        x9 = self.conv9(x)
        if self.config["debug"]:
            print("After Conv9: ", x9.shape)
        x = self.trans3(x9)
        if self.config["debug"]:
            print("After Transition3: ", x.shape)
        x = self.conv10(x)
        if self.config["debug"]:
            print("After Conv10: ", x.shape)
        x = x + self.conv11(x)
        if self.config["debug"]:
            print("After Conv11: ", x.shape)
        x = self.conv12(x)
        if self.config["debug"]:
            print("After Conv12: ", x.shape)
        x = self.gap(x)
        if self.config["debug"]:
            print("After GAP: ", x.shape)

        if self.fc is None:
            in_features = x.shape[1] * x.shape[2] * x.shape[3]
            self.fc = nn.Linear(in_features, 10).to(
                x.device
            )  # output_size = 1, RF = 125

        x = x.view(x.size(0), -1)
        if self.config["debug"]:
            print("After View: ", x.shape)
        x = self.fc(x)
        if self.config["debug"]:
            print("After FC: ", x.shape)
        return F.log_softmax(x, dim=-1)
