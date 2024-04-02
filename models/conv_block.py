import torch.nn as nn
import torch.nn.functional as F

# import sys
# # add parent directory to path
# sys.path.append('/Users/aakashvardhan/Library/CloudStorage/GoogleDrive-vardhan.aakash1@gmail.com/My Drive/ERA v2/s8-normalization/config.py')


GROUP_SIZE_GN = 2
GROUP_SIZE_LN = 1


class LayerNorm(nn.Module):
    def __init__(self, num_features):
        """
        Initializes a LayerNorm module.

        Args:
            num_features (int): Number of input features.
        """
        super().__init__()
        self.layer_norm = nn.GroupNorm(
            num_groups=GROUP_SIZE_LN, num_channels=num_features
        )

    def forward(self, x):
        """
        Forward pass of the LayerNorm module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying layer normalization.
        """
        return self.layer_norm(x)


class DepthwiseSeparableConv2d(nn.Module):
    """
    Depthwise Separable Convolutional Block.

    This class implements a depthwise separable convolutional block, which consists of a depthwise convolution
    followed by a pointwise convolution. It is commonly used in convolutional neural networks to reduce the
    number of parameters and improve computational efficiency.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (tuple, optional): Size of the convolutional kernel. Defaults to (3, 3).
        padding (int, optional): Amount of padding. Defaults to 1.
        bias (bool, optional): Whether to include a bias term. Defaults to False.
        **kwargs: Additional keyword arguments to be passed to the convolutional layers.

    Attributes:
        depthwise (nn.Conv2d): Depthwise convolutional layer.
        pointwise (nn.Conv2d): Pointwise convolutional layer.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=(3, 3),
        padding=1,
        bias=False,
        **kwargs
    ):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=in_channels,
            bias=bias,
            **kwargs
        )
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, kernel_size=(1, 1), bias=bias, **kwargs
        )

    def forward(self, x):
        """
        Forward pass of the convolutional block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after passing through the depthwise and pointwise convolutions.
        """
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class ConvBlock(nn.Module):
    """
    A convolutional block module that consists of a convolutional layer followed by normalization, activation, and dropout.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        norm (str): Type of normalization to be applied. Supported values are "bn" (BatchNorm2d), "gn" (GroupNorm), and "ln" (LayerNorm).
        kernel_size (tuple, optional): Size of the convolutional kernel. Defaults to (3, 3).
        dropout_value (float, optional): Dropout probability. Defaults to 0.
        dilation (int, optional): Dilation rate for the convolutional layer. Defaults to 1.
        depthwise_seperable (bool, optional): Whether to use depthwise separable convolution. Defaults to False.
        **kwargs: Additional keyword arguments to be passed to the convolutional layer.

    Attributes:
        norm (nn.Module): Normalization layer.
        conv (nn.Module): Convolutional layer.
        convblock (nn.Sequential): Sequential module consisting of the convolutional layer, normalization, ReLU activation, and dropout.

    Methods:
        forward(x): Forward pass of the ConvBlock module.

    """

    def __init__(
        self,
        in_channels,
        out_channels,
        norm,
        kernel_size=(3, 3),
        dropout_value=0,
        dilation=1,
        depthwise_seperable=False,
        **kwargs
    ):
        super().__init__()

        if norm == "bn":
            self.norm = lambda num_features: nn.BatchNorm2d(num_features)
        elif norm == "gn":
            self.norm = lambda num_features: nn.GroupNorm(GROUP_SIZE_GN, num_features)
        elif norm == "ln":
            self.norm = lambda num_features: LayerNorm(num_features)
        else:
            raise ValueError("Norm type {} not supported".format(norm))

        if depthwise_seperable:
            self.conv = DepthwiseSeparableConv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                bias=False,
                dilation=dilation,
                **kwargs
            )
        else:
            self.conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                dilation=dilation,
                bias=False,
                **kwargs
            )

        self.convblock = nn.Sequential(
            self.conv, self.norm(out_channels), nn.ReLU(), nn.Dropout(dropout_value)
        )

    def forward(self, x):
        """
        Forward pass of the ConvBlock module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after passing through the ConvBlock.
        """
        return self.convblock(x)
