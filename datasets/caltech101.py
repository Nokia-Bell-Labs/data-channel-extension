# © 2024 Nokia
# Licensed under the BSD 3-Clause Clear License
# SPDX-License-Identifier: BSD-3-Clause-Clear

import os
import os.path
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple, Union

from PIL import Image

from torchvision.datasets.utils import download_and_extract_archive, verify_str_arg
from torchvision.datasets.vision import VisionDataset
from utils.data_reshape import DataReshape, fractional_repeat, DownsamplePIL, DownsampleTensor, DownsampleTensorRepeat
from utils.data_augmentation import DataRotation
from utils.coordconv import AI8XCoordConv2D
from functools import partial

initial_image_size = 300
class Caltech101(VisionDataset):
    """`Caltech 101 <https://data.caltech.edu/records/20086>`_ Dataset.

    .. warning::

        This class needs `scipy <https://docs.scipy.org/doc/>`_ to load target files from `.mat` format.

    Args:
        root (str or ``pathlib.Path``): Root directory of dataset where directory
            ``caltech101`` exists or will be saved to if download is set to True.
        target_type (string or list, optional): Type of target to use, ``category`` or
            ``annotation``. Can also be a list to output a tuple with all specified
            target types.  ``category`` represents the target class, and
            ``annotation`` is a list of points from a hand-generated outline.
            Defaults to ``category``.
        transform (callable, optional): A function/transform that takes in a PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

            .. warning::

                To download the dataset `gdown <https://github.com/wkentaro/gdown>`_ is required.
    """

    def __init__(
            self,
            root: Union[str, Path],
            target_type: Union[List[str], str] = "category",
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
    ) -> None:
        super().__init__(os.path.join(root, "caltech101"), transform=transform, target_transform=target_transform)
        os.makedirs(self.root, exist_ok=True)
        if isinstance(target_type, str):
            target_type = [target_type]
        self.target_type = [verify_str_arg(t, "target_type", ("category", "annotation")) for t in target_type]

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted. You can use download=True to download it")

        self.categories = sorted(os.listdir(os.path.join(self.root, "101_ObjectCategories")))
        self.categories.remove("BACKGROUND_Google")  # this is not a real class

        # For some reason, the category names in "101_ObjectCategories" and
        # "Annotations" do not always match. This is a manual map between the
        # two. Defaults to using same name, since most names are fine.
        name_map = {
            "Faces": "Faces_2",
            "Faces_easy": "Faces_3",
            "Motorbikes": "Motorbikes_16",
            "airplanes": "Airplanes_Side_2",
        }
        self.annotation_categories = list(map(lambda x: name_map[x] if x in name_map else x, self.categories))

        self.index: List[int] = []
        self.y = []
        for (i, c) in enumerate(self.categories):
            n = len(os.listdir(os.path.join(self.root, "101_ObjectCategories", c)))
            self.index.extend(range(1, n + 1))
            self.y.extend(n * [i])

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where the type of target specified by target_type.
        """
        import scipy.io

        img = Image.open(
            os.path.join(
                self.root,
                "101_ObjectCategories",
                self.categories[self.y[index]],
                f"image_{self.index[index]:04d}.jpg",
            )
        )

        target: Any = []
        for t in self.target_type:
            if t == "category":
                target.append(self.y[index])
            elif t == "annotation":
                data = scipy.io.loadmat(
                    os.path.join(
                        self.root,
                        "Annotations",
                        self.annotation_categories[self.y[index]],
                        f"annotation_{self.index[index]:04d}.mat",
                    )
                )
                target.append(data["obj_contour"])
        target = tuple(target) if len(target) > 1 else target[0]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def _check_integrity(self) -> bool:
        # can be more robust and check hash of files
        return os.path.exists(os.path.join(self.root, "101_ObjectCategories"))

    def __len__(self) -> int:
        return len(self.index)

    def download(self) -> None:
        if self._check_integrity():
            print("Files already downloaded and verified")
            return

        download_and_extract_archive(
            "https://drive.google.com/file/d/137RyRjvTBkBiIfeYBNZBtViDHQ6_Ewsp",
            self.root,
            filename="101_ObjectCategories.tar.gz",
            md5="b224c7392d521a49829488ab0f1120d9",
        )
        download_and_extract_archive(
            "https://drive.google.com/file/d/175kQy3UsZ0wUEHZjqkUDdNVssr7bgh_m",
            self.root,
            filename="Annotations.tar",
            md5="6f83eeb1f24d99cab4eb377263132c91",
        )

    def extra_repr(self) -> str:
        return "Target type: {target_type}".format(**self.__dict__)


import os

import torchvision
from torchvision import transforms
import torch
import ai8x
import math
from datasets.custom_dataset import CustomSubsetDataset



def caltech101_get_datasets(data, load_train=True, load_test=True,
                            input_size=224, target_size=64, target_channel=3):
    (data_dir, args) = data

    full_train_dataset = Caltech101(
        os.path.join(data_dir, 'Caltech101'),
        download=True,
    )

    # Split the dataset into training (80%) and validation (20%)
    num_train = int(0.8 * len(full_train_dataset))
    num_val = len(full_train_dataset) - num_train
    train_dataset, val_dataset = torch.utils.data.random_split(full_train_dataset, [num_train, num_val])


    transform_list = []

    transform_list.append(transforms.Lambda(lambda x: x.convert("RGB")))  # Convert grayscale to RGB if needed
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

        train_dataset = CustomSubsetDataset(
            train_dataset, transform=train_transform)
    else:
        train_dataset = None

    if load_test:
        test_transform = transforms.Compose(transform_list)

        test_dataset = CustomSubsetDataset(
            val_dataset, transform=test_transform)

        if args.truncate_testset:
            test_dataset.data = test_dataset.data[:1]
    else:
        test_dataset = None

    return train_dataset, test_dataset

datasets = []

for size in [32]:
    for channel in range(65):
        dic = {}
        dic['name'] = f'Caltech101_{channel}x{size}x{size}'
        dic['input'] = (channel, size, size)
        dic['output'] = list(map(str, range(101)))
        dic['loader'] = partial(caltech101_get_datasets, load_train=True, load_test=True, input_size=initial_image_size,
                                target_size=size, target_channel=channel)
        datasets.append(dic)
