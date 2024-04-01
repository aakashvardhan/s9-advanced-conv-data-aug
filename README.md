# Advanced Convolution Techniques & Data Augmentation Strategies for Image Classification

## Table of Contents
1. [Introduction](#introduction)
2. [Advanced Convolution Techniques](#advanced-convolution-techniques)
3. [Albumentations](#albumentations)
4. [Model Summary](#model-summary)
5. [Results](#results)


## Introduction

This repository is a development of [s8-normalization](https://github.com/aakashvardhan/s8-normalization)

In this repository, we will be using the following techniques to improve the performance of the model:
1. **Advanced Convolution Techniques**:
    - Depthwise Separable Convolution
    - Dilated Convolution
2. **Data Augmentation Techniques**:
    - Albumentations

We will be using the CIFAR-10 dataset for this experiment and using Batch Normalization.

## Advanced Convolution Techniques

### Depthwise Separable Convolution

Depthwise Separable Convolution is a technique that factorizes a standard convolution into two separate layers:
1. Depthwise Convolution: This layer applies a single filter per input channel. It is used to learn spatial information.
2. Pointwise Convolution: This layer applies a 1x1 convolution to combine the output of Depthwise Convolution. It is used to learn the depthwise information.

```python
# In conv_block.py
import torch.nn as nn

class DepthwiseSeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3,3), padding=1, bias=False, **kwargs):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=padding, groups=in_channels, bias=bias, **kwargs)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=(1,1), bias=bias, **kwargs)
        
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x
```

### Dilated Convolution

Dilated Convolution is a technique that increases the receptive field of the network without increasing the number of parameters. It introduces gaps in the convolutional kernel to increase the receptive field.

```python
# In model.py

class Net(nn.Module):
    def __init__(self, n_channels=3, norm='bn', dropout=0.1):
        ...
        self.conv9 = ConvBlock(in_channels=n_channels, out_channels=n_channels, norm=norm, padding=0,dilation=4,dropout_value=dropout)
        ...
    def forward(self, x):
        ...
        x = self.conv9(x)
        ...
```

## Albumentations

Albumentations is a Python library for image augmentation. It is used to apply a variety of transformations to the input image to increase the diversity of the dataset.

```python
import albumentations as A

A.Compose([
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=15, p=0.3),
            A.HorizontalFlip(p=0.5),
            A.CoarseDropout(max_holes=1, max_height=16, max_width=16, min_holes=1, min_height=16, min_width=16, fill_value=(0.4914, 0.4822, 0.4465), mask_fill_value=None),
            A.RandomBrightnessContrast(p=0.2),
            A.CenterCrop(32, 32, always_apply=True),
            A.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.247, 0.243, 0.261)),
            ToTensorV2(),
        ])
```
- **ShiftScaleRotate**: Randomly apply affine transforms: translate, scale and rotate the input.
- **HorizontalFlip**: Randomly horizontally flip the input.
- **CoarseDropout**: Randomly fill the input image with holes.
- **RandomBrightnessContrast**: Randomly change the brightness and contrast of the input image.
- **CenterCrop**: Crop the central part of the input.
- **Normalize**: Normalize the input image.
- **ToTensorV2**: Convert the input image to a PyTorch tensor.

## Model Summary

```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 19, 32, 32]             513
            Conv2d-2           [-1, 19, 32, 32]             513
       BatchNorm2d-3           [-1, 19, 32, 32]              38
              ReLU-4           [-1, 19, 32, 32]               0
           Dropout-5           [-1, 19, 32, 32]               0
         ConvBlock-6           [-1, 19, 32, 32]               0
            Conv2d-7           [-1, 19, 32, 32]           3,249
            Conv2d-8           [-1, 19, 32, 32]           3,249
       BatchNorm2d-9           [-1, 19, 32, 32]              38
             ReLU-10           [-1, 19, 32, 32]               0
          Dropout-11           [-1, 19, 32, 32]               0
        ConvBlock-12           [-1, 19, 32, 32]               0
           Conv2d-13           [-1, 38, 15, 15]           6,498
           Conv2d-14           [-1, 38, 15, 15]           6,498
      BatchNorm2d-15           [-1, 38, 15, 15]              76
             ReLU-16           [-1, 38, 15, 15]               0
          Dropout-17           [-1, 38, 15, 15]               0
        ConvBlock-18           [-1, 38, 15, 15]               0
           Conv2d-19           [-1, 19, 15, 15]             722
  TransitionBlock-20           [-1, 19, 15, 15]               0
           Conv2d-21           [-1, 38, 15, 15]           6,498
           Conv2d-22           [-1, 38, 15, 15]           6,498
      BatchNorm2d-23           [-1, 38, 15, 15]              76
             ReLU-24           [-1, 38, 15, 15]               0
          Dropout-25           [-1, 38, 15, 15]               0
        ConvBlock-26           [-1, 38, 15, 15]               0
           Conv2d-27           [-1, 38, 15, 15]          12,996
           Conv2d-28           [-1, 38, 15, 15]          12,996
      BatchNorm2d-29           [-1, 38, 15, 15]              76
             ReLU-30           [-1, 38, 15, 15]               0
          Dropout-31           [-1, 38, 15, 15]               0
        ConvBlock-32           [-1, 38, 15, 15]               0
           Conv2d-33           [-1, 38, 11, 11]          12,996
           Conv2d-34           [-1, 38, 11, 11]          12,996
      BatchNorm2d-35           [-1, 38, 11, 11]              76
             ReLU-36           [-1, 38, 11, 11]               0
          Dropout-37           [-1, 38, 11, 11]               0
        ConvBlock-38           [-1, 38, 11, 11]               0
           Conv2d-39           [-1, 19, 11, 11]             722
  TransitionBlock-40           [-1, 19, 11, 11]               0
           Conv2d-41           [-1, 19, 11, 11]             171
           Conv2d-42           [-1, 19, 11, 11]             171
           Conv2d-43           [-1, 38, 11, 11]             722
           Conv2d-44           [-1, 38, 11, 11]             722
DepthwiseSeparableConv2d-45           [-1, 38, 11, 11]               0
DepthwiseSeparableConv2d-46           [-1, 38, 11, 11]               0
      BatchNorm2d-47           [-1, 38, 11, 11]              76
             ReLU-48           [-1, 38, 11, 11]               0
          Dropout-49           [-1, 38, 11, 11]               0
        ConvBlock-50           [-1, 38, 11, 11]               0
           Conv2d-51           [-1, 38, 11, 11]          12,996
           Conv2d-52           [-1, 38, 11, 11]          12,996
      BatchNorm2d-53           [-1, 38, 11, 11]              76
             ReLU-54           [-1, 38, 11, 11]               0
          Dropout-55           [-1, 38, 11, 11]               0
        ConvBlock-56           [-1, 38, 11, 11]               0
           Conv2d-57             [-1, 38, 3, 3]          12,996
           Conv2d-58             [-1, 38, 3, 3]          12,996
      BatchNorm2d-59             [-1, 38, 3, 3]              76
             ReLU-60             [-1, 38, 3, 3]               0
          Dropout-61             [-1, 38, 3, 3]               0
        ConvBlock-62             [-1, 38, 3, 3]               0
           Conv2d-63             [-1, 19, 3, 3]             722
  TransitionBlock-64             [-1, 19, 3, 3]               0
           Conv2d-65             [-1, 19, 3, 3]             171
           Conv2d-66             [-1, 19, 3, 3]             171
           Conv2d-67             [-1, 38, 3, 3]             722
           Conv2d-68             [-1, 38, 3, 3]             722
DepthwiseSeparableConv2d-69             [-1, 38, 3, 3]               0
DepthwiseSeparableConv2d-70             [-1, 38, 3, 3]               0
      BatchNorm2d-71             [-1, 38, 3, 3]              76
             ReLU-72             [-1, 38, 3, 3]               0
          Dropout-73             [-1, 38, 3, 3]               0
        ConvBlock-74             [-1, 38, 3, 3]               0
           Conv2d-75             [-1, 38, 3, 3]          12,996
           Conv2d-76             [-1, 38, 3, 3]          12,996
      BatchNorm2d-77             [-1, 38, 3, 3]              76
             ReLU-78             [-1, 38, 3, 3]               0
          Dropout-79             [-1, 38, 3, 3]               0
        ConvBlock-80             [-1, 38, 3, 3]               0
           Conv2d-81             [-1, 38, 1, 1]          12,996
           Conv2d-82             [-1, 38, 1, 1]          12,996
      BatchNorm2d-83             [-1, 38, 1, 1]              76
             ReLU-84             [-1, 38, 1, 1]               0
          Dropout-85             [-1, 38, 1, 1]               0
        ConvBlock-86             [-1, 38, 1, 1]               0
AdaptiveAvgPool2d-87             [-1, 38, 1, 1]               0
        Linear-88                   [-1, 10]             380
================================================================
Total params: 196,422
Trainable params: 196,422
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.01
Forward/backward pass size (MB): 3.85
Params size (MB): 0.75
Estimated Total Size (MB): 4.61
----------------------------------------------------------------
```

**Note**: The model summary is slightly different as outputted from the [notebook](https://github.com/aakashvardhan/s9-advanced-conv-data-aug/blob/main/notebooks/model_train.ipynb).

The fully connected layer is not included in the model summary as it was initialzed in the forward method of the model.

```python
# In model.py

def forward(self, x):
    ...
    if self.fc is None:
        in_features = x.shape[1] * x.shape[2] * x.shape[3]
        self.fc = nn.Linear(in_features, 10).to(x.device)
    ...
```