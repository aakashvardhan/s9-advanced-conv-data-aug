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

### Dilated Convolution

Dilated Convolution is a technique that increases the receptive field of the network without increasing the number of parameters. It introduces gaps in the convolutional kernel to increase the receptive field.

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
