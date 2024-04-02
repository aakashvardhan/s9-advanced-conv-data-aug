import torch
import torch.nn as nn
import torch.nn.functional as F


class TransitionBlock(nn.Module):
    """
    Transition block in a neural network architecture.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (tuple, optional): Size of the convolutional kernel. Defaults to (1, 1).
        **kwargs: Additional keyword arguments to be passed to the nn.Conv2d layer.
    """

    def __init__(self, in_channels, out_channels, kernel_size=(1, 1), **kwargs):
        super().__init__()

        self.conv1d = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                bias=False,
                **kwargs
            ),
        )

    def forward(self, x):
        """
        Forward pass of the TransitionBlock.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after passing through the convolutional layer.
        """
        x = self.conv1d(x)
        return x
