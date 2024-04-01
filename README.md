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

**Note**: The model summary is manually edited in **README.md** as outputted from the [notebook](https://github.com/aakashvardhan/s9-advanced-conv-data-aug/blob/main/notebooks/model_train.ipynb).

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

## Results

The model was trained for 100 epochs with the following hyperparameters:
- Batch Size: 128
- Learning Rate: 0.05
- StepLR with step_size=25 and gamma=0.5

```
EPOCH: 1
Loss=1.5628697872161865 Batch_id=390 Accuracy=33.97: 100%|██████████| 391/391 [00:16<00:00, 23.26it/s]

Test set: Average loss: 1.4640, Accuracy: 4539/10000 (45.39%)

Learning rate: 0.05
EPOCH: 2
Loss=1.3799045085906982 Batch_id=390 Accuracy=44.64: 100%|██████████| 391/391 [00:17<00:00, 22.83it/s]

Test set: Average loss: 1.2954, Accuracy: 5299/10000 (52.99%)

Learning rate: 0.05
EPOCH: 3
Loss=1.2930619716644287 Batch_id=390 Accuracy=49.62: 100%|██████████| 391/391 [00:16<00:00, 23.94it/s]

Test set: Average loss: 1.2055, Accuracy: 5630/10000 (56.30%)

Learning rate: 0.05
EPOCH: 4
Loss=1.2050474882125854 Batch_id=390 Accuracy=53.87: 100%|██████████| 391/391 [00:15<00:00, 24.81it/s]

Test set: Average loss: 1.1113, Accuracy: 5994/10000 (59.94%)

Learning rate: 0.05
EPOCH: 5
Loss=1.2289100885391235 Batch_id=390 Accuracy=57.76: 100%|██████████| 391/391 [00:16<00:00, 23.19it/s]

Test set: Average loss: 1.0228, Accuracy: 6334/10000 (63.34%)

Learning rate: 0.05

...
...
...

EPOCH: 25
Loss=0.785340428352356 Batch_id=390 Accuracy=76.56: 100%|██████████| 391/391 [00:15<00:00, 25.48it/s]

Test set: Average loss: 0.5449, Accuracy: 8131/10000 (81.31%)

Learning rate: 0.025
EPOCH: 26
Loss=0.8196824789047241 Batch_id=390 Accuracy=76.55: 100%|██████████| 391/391 [00:14<00:00, 27.33it/s]

Test set: Average loss: 0.5203, Accuracy: 8201/10000 (82.01%)

Learning rate: 0.025
EPOCH: 27
Loss=0.6206899881362915 Batch_id=390 Accuracy=76.63: 100%|██████████| 391/391 [00:16<00:00, 23.61it/s]

Test set: Average loss: 0.5259, Accuracy: 8199/10000 (81.99%)

Learning rate: 0.025
EPOCH: 28
Loss=0.6587123870849609 Batch_id=390 Accuracy=76.74: 100%|██████████| 391/391 [00:14<00:00, 26.38it/s]

Test set: Average loss: 0.5232, Accuracy: 8210/10000 (82.10%)

Learning rate: 0.025
EPOCH: 29
Loss=1.0209885835647583 Batch_id=390 Accuracy=76.82: 100%|██████████| 391/391 [00:14<00:00, 26.79it/s]

Test set: Average loss: 0.5219, Accuracy: 8196/10000 (81.96%)

Learning rate: 0.025
EPOCH: 30
Loss=0.6588395833969116 Batch_id=390 Accuracy=77.33: 100%|██████████| 391/391 [00:14<00:00, 26.47it/s]

Test set: Average loss: 0.5211, Accuracy: 8190/10000 (81.90%)

Learning rate: 0.025

...
...
...

EPOCH: 50
Loss=0.8329460024833679 Batch_id=390 Accuracy=79.73: 100%|██████████| 391/391 [00:16<00:00, 24.23it/s]

Test set: Average loss: 0.4875, Accuracy: 8344/10000 (83.44%)

Learning rate: 0.0125
EPOCH: 51
Loss=0.5698076486587524 Batch_id=390 Accuracy=80.00: 100%|██████████| 391/391 [00:16<00:00, 24.04it/s]

Test set: Average loss: 0.4757, Accuracy: 8330/10000 (83.30%)

Learning rate: 0.0125
EPOCH: 52
Loss=0.570016086101532 Batch_id=390 Accuracy=79.97: 100%|██████████| 391/391 [00:16<00:00, 24.42it/s]

Test set: Average loss: 0.4629, Accuracy: 8416/10000 (84.16%)

Learning rate: 0.0125
EPOCH: 53
Loss=0.3298833668231964 Batch_id=390 Accuracy=80.30: 100%|██████████| 391/391 [00:16<00:00, 23.80it/s]

Test set: Average loss: 0.4722, Accuracy: 8399/10000 (83.99%)

Learning rate: 0.0125
EPOCH: 54
Loss=0.5669925808906555 Batch_id=390 Accuracy=79.95: 100%|██████████| 391/391 [00:16<00:00, 23.38it/s]

Test set: Average loss: 0.4595, Accuracy: 8402/10000 (84.02%)

Learning rate: 0.0125
EPOCH: 55
Loss=0.6621028780937195 Batch_id=390 Accuracy=79.87: 100%|██████████| 391/391 [00:16<00:00, 24.09it/s]

Test set: Average loss: 0.4707, Accuracy: 8397/10000 (83.97%)

Learning rate: 0.0125

...
...
...

EPOCH: 75
Loss=0.49508872628211975 Batch_id=390 Accuracy=81.34: 100%|██████████| 391/391 [00:16<00:00, 23.48it/s]

Test set: Average loss: 0.4375, Accuracy: 8501/10000 (85.01%)

Learning rate: 0.00625
EPOCH: 76
Loss=0.6408173441886902 Batch_id=390 Accuracy=81.52: 100%|██████████| 391/391 [00:16<00:00, 23.07it/s]

Test set: Average loss: 0.4459, Accuracy: 8475/10000 (84.75%)

Learning rate: 0.00625
EPOCH: 77
Loss=0.5987493395805359 Batch_id=390 Accuracy=81.59: 100%|██████████| 391/391 [00:17<00:00, 22.03it/s]

Test set: Average loss: 0.4385, Accuracy: 8498/10000 (84.98%)

Learning rate: 0.00625
EPOCH: 78
Loss=0.32850271463394165 Batch_id=390 Accuracy=81.28: 100%|██████████| 391/391 [00:16<00:00, 23.73it/s]

Test set: Average loss: 0.4369, Accuracy: 8497/10000 (84.97%)

Learning rate: 0.00625
EPOCH: 79
Loss=0.46883076429367065 Batch_id=390 Accuracy=81.44: 100%|██████████| 391/391 [00:16<00:00, 23.75it/s]

Test set: Average loss: 0.4384, Accuracy: 8481/10000 (84.81%)

Learning rate: 0.00625
EPOCH: 80
Loss=0.40786638855934143 Batch_id=390 Accuracy=81.57: 100%|██████████| 391/391 [00:17<00:00, 22.31it/s]

Test set: Average loss: 0.4411, Accuracy: 8498/10000 (84.98%)

Learning rate: 0.003125

...
...
...

EPOCH: 95
Loss=0.5651302337646484 Batch_id=390 Accuracy=82.33: 100%|██████████| 391/391 [00:16<00:00, 23.91it/s]

Test set: Average loss: 0.4312, Accuracy: 8525/10000 (85.25%)

Learning rate: 0.003125
EPOCH: 96
Loss=0.5819368958473206 Batch_id=390 Accuracy=82.10: 100%|██████████| 391/391 [00:16<00:00, 23.97it/s]

Test set: Average loss: 0.4260, Accuracy: 8548/10000 (85.48%)

Learning rate: 0.003125
EPOCH: 97
Loss=0.5556617975234985 Batch_id=390 Accuracy=82.07: 100%|██████████| 391/391 [00:17<00:00, 22.08it/s]

Test set: Average loss: 0.4329, Accuracy: 8502/10000 (85.02%)

Learning rate: 0.003125
EPOCH: 98
Loss=0.38931137323379517 Batch_id=390 Accuracy=82.09: 100%|██████████| 391/391 [00:16<00:00, 23.23it/s]

Test set: Average loss: 0.4274, Accuracy: 8552/10000 (85.52%)

Learning rate: 0.003125
EPOCH: 99
Loss=0.6452785730361938 Batch_id=390 Accuracy=81.89: 100%|██████████| 391/391 [00:16<00:00, 23.55it/s]

Test set: Average loss: 0.4292, Accuracy: 8552/10000 (85.52%)

Learning rate: 0.003125
EPOCH: 100
Loss=0.46943503618240356 Batch_id=390 Accuracy=82.27: 100%|██████████| 391/391 [00:17<00:00, 21.92it/s]

Test set: Average loss: 0.4303, Accuracy: 8543/10000 (85.43%)

Learning rate: 0.0015625
```

The model achieved an accuracy of 85.43% on the test dataset after 100 epochs.

![Accuracy](https://github.com/aakashvardhan/s9-advanced-conv-data-aug/blob/main/asset/model-performance.png)