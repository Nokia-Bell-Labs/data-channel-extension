# © 2024 Nokia
# Licensed under the BSD 3-Clause Clear License
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""
ImageNet Dataset (using PyTorch's ImageNet and ImageFolder classes)
"""
import os

import torchvision
from torch.utils.data import Dataset
from torchvision import transforms
import torch
import ai8x
import math
from utils.data_reshape import DataReshape, fractional_repeat, DownsamplePIL, DownsampleTensor, DownsampleTensorRepeat
from utils.data_augmentation import DataRotation
from utils.coordconv import AI8XCoordConv2D
from functools import partial

initial_image_size = 350
def imagenette_get_datasets(data, load_train=True, load_test=True,
                            input_size=224, target_size=64, target_channel=3, folder=True, augment_data=True):
    """
    Load the ImageNet 2012 Classification dataset.

    The original training dataset is split into training and validation sets.
    By default we use a 90:10 (45K:5K) training:validation split.

    The output of torchvision datasets are PIL Image images of range [0, 1].
    We transform them to Tensors of normalized range [-128/128, +127/128]

    Data augmentation: 4 pixels are padded on each side, and a 224x224 crop is randomly sampled
    from the padded image or its horizontal flip.
    """
    (data_dir, args) = data
    transform_list = []

    transform_list.append(transforms.Resize((input_size, input_size)))

    assert (args.data_rotate + args.data_repeat + args.coordconv + args.data_reshape) <= 1, "Only one or zero variable should be True"
    if args.data_rotate:
        transform_list.append(DownsamplePIL(target_size))
        transform_list.append(DataRotation(target_channel))
        transform_list.append(transforms.ToTensor())
        transform_list.append(transforms.Normalize(fractional_repeat((0.485, 0.456, 0.406), target_channel),
                                                   fractional_repeat((0.229, 0.224, 0.225), target_channel)))
    elif args.data_repeat:
        transform_list.append(transforms.ToTensor())
        transform_list.append(DownsampleTensorRepeat(target_size, target_channel))
        transform_list.append(transforms.Normalize(fractional_repeat((0.485, 0.456, 0.406), target_channel),
                                                   fractional_repeat((0.229, 0.224, 0.225), target_channel)))
    elif args.coordconv:
        transform_list.append(transforms.ToTensor())
        transform_list.append(DownsampleTensor(target_size))
        transform_list.append(transforms.Normalize((0.485, 0.456, 0.406),
                                                   (0.229, 0.224, 0.225)))
        transform_list.append(AI8XCoordConv2D(args.with_r))

    elif args.data_reshape:
        transform_list.append(transforms.ToTensor())
        transform_list.append(DataReshape(target_size, target_channel, args.data_reshape_method))
        transform_list.append(transforms.Normalize(fractional_repeat((0.485, 0.456, 0.406), target_channel),
                                                   fractional_repeat((0.229, 0.224, 0.225), target_channel)))
    else:  #simple downsampling
        transform_list.append(transforms.ToTensor())
        transform_list.append(DownsampleTensor(target_size))
        transform_list.append(transforms.Normalize((0.485, 0.456, 0.406),
                                                   (0.229, 0.224, 0.225)))
    transform_list.append(ai8x.normalize(args=args))

    if load_train:
        train_transform = transforms.Compose(transform_list)

        if not folder:
            train_dataset = torchvision.datasets.ImageNet(
                os.path.join(data_dir, 'Imagenette'),
                split='train',
                transform=train_transform,
            )
        else:
            train_dataset = torchvision.datasets.ImageFolder(
                os.path.join(data_dir, 'Imagenette', 'train'),
                transform=train_transform,
            )
    else:
        train_dataset = None

    if load_test:
        test_transform = transforms.Compose(transform_list)


        if not folder:
            test_dataset = torchvision.datasets.ImageNet(
                os.path.join(data_dir, 'Imagenette'),
                split='val',
                transform=test_transform,
            )
        else:
            test_dataset = torchvision.datasets.ImageFolder(
                os.path.join(data_dir, 'Imagenette', 'val'),
                transform=test_transform,
            )

        if args.truncate_testset:
            test_dataset.data = test_dataset.data[:1]
    else:
        test_dataset = None

    return train_dataset, test_dataset


datasets = []

for size in [32]:
    for channel in range(65):
        dic = {}
        dic['name'] = f'Imagenette_{channel}x{size}x{size}'
        dic['input'] = (channel, size, size)
        dic['output'] = list(map(str, range(10)))
        dic['loader'] = partial(imagenette_get_datasets, load_train=True, load_test=True, input_size=initial_image_size,
                                target_size=size, target_channel=channel)
        datasets.append(dic)
