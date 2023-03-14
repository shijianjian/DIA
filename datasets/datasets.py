import os

import numpy as np
import torch
from torch.utils.data.dataset import Subset
from torchvision import datasets, transforms

from utils.utils import set_random_seed


BINARY_SUPERCLASS = list(range(2))
CIFAR10_SUPERCLASS = list(range(10))  # one class

CIFAR100_SUPERCLASS = [
    [4, 31, 55, 72, 95],
    [1, 33, 67, 73, 91],
    [54, 62, 70, 82, 92],
    [9, 10, 16, 29, 61],
    [0, 51, 53, 57, 83],
    [22, 25, 40, 86, 87],
    [5, 20, 26, 84, 94],
    [6, 7, 14, 18, 24],
    [3, 42, 43, 88, 97],
    [12, 17, 38, 68, 76],
    [23, 34, 49, 60, 71],
    [15, 19, 21, 32, 39],
    [35, 63, 64, 66, 75],
    [27, 45, 77, 79, 99],
    [2, 11, 36, 46, 98],
    [28, 30, 44, 78, 93],
    [37, 50, 65, 74, 80],
    [47, 52, 56, 59, 96],
    [8, 13, 48, 58, 90],
    [41, 69, 81, 85, 89],
]


class MultiDataTransform(object):
    def __init__(self, transform):
        self.transform1 = transform
        self.transform2 = transform

    def __call__(self, sample):
        x1 = self.transform1(sample)
        x2 = self.transform2(sample)
        return x1, x2


class MultiDataTransformList(object):
    def __init__(self, transform, clean_trasform, sample_num):
        self.transform = transform
        self.clean_transform = clean_trasform
        self.sample_num = sample_num

    def __call__(self, sample):
        set_random_seed(0)

        sample_list = []
        for i in range(self.sample_num):
            sample_list.append(self.transform(sample))

        return sample_list, self.clean_transform(sample)


def get_transform(image_size=None):
    # Note: data augmentation is implemented in the layers
    # Hence, we only define the identity transformation here
    if image_size:  # use pre-specified image size
        train_transform = transforms.Compose([
            transforms.Resize((image_size[0], image_size[1])),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        test_transform = transforms.Compose([
            transforms.Resize((image_size[0], image_size[1])),
            transforms.ToTensor(),
        ])
    else:  # use default image size
        train_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        test_transform = transforms.ToTensor()

    return train_transform, test_transform


def get_subset_with_len(dataset, length, shuffle=False):
    set_random_seed(0)
    dataset_size = len(dataset)

    index = np.arange(dataset_size)
    if shuffle:
        np.random.shuffle(index)

    index = torch.from_numpy(index[0:length])
    subset = Subset(dataset, index)

    assert len(subset) == length

    return subset


def get_transform_imagenet():

    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    train_transform = MultiDataTransform(train_transform)

    return train_transform, test_transform


def get_dataset(P, dataset, test_only=False, image_size=None, download=True, eval=False):

    train_transform, test_transform = get_transform(image_size=image_size)

    if dataset == 'pneumoniamnist'or dataset == 'breastmnist' or dataset == 'chestmnist' :
        import medmnist
        info = medmnist.INFO[dataset]
        image_size = (28, 28, 3)
        n_classes = len(info['label'])
        cls = getattr(medmnist, info['python_class'])
        train_set = cls(root=P.data_root, split='train', transform=train_transform, download=download, as_rgb=True)
        test_set = cls(root=P.data_root, split='test', transform=test_transform, download=download, as_rgb=True)
        train_set.labels = train_set.labels.squeeze()
        test_set.labels = test_set.labels.squeeze()
        if dataset == 'chestmnist':
            train_set.labels = train_set.labels.any(1).astype(int)
            test_set.labels = test_set.labels.any(1).astype(int)
        train_set.targets = train_set.labels
        test_set.targets = test_set.labels

    elif dataset == "covid":
        image_size = (224, 224, 3)
        n_classes = 2
        train_dir = os.path.join(P.data_root, 'covid', "train")
        test_dir = os.path.join(P.data_root, 'covid', "test")
        train_transform, test_transform = get_transform(image_size=image_size)
        train_set = datasets.ImageFolder(train_dir, transform=train_transform)
        test_set = datasets.ImageFolder(test_dir, transform=test_transform)

    elif dataset == "kvasir":
        image_size = (224, 224, 3)
        n_classes = 2
        train_dir = os.path.join(P.data_root, 'kvasir-anomaly-polyps', "train")
        test_dir = os.path.join(P.data_root, 'kvasir-anomaly-polyps', "test")
        train_transform, test_transform = get_transform(image_size=image_size)
        train_set = datasets.ImageFolder(train_dir, transform=train_transform)
        test_set = datasets.ImageFolder(test_dir, transform=test_transform)

    elif dataset == "chestxray":
        image_size = (224, 224, 3)
        n_classes = 2
        train_dir = os.path.join(P.data_root, "chest_xray", "train")
        test_dir = os.path.join(P.data_root, "chest_xray", "test")
        train_transform, test_transform = get_transform(image_size=image_size)
        train_set = datasets.ImageFolder(train_dir, transform=train_transform)
        test_set = datasets.ImageFolder(test_dir, transform=test_transform)

    elif dataset == "retina":
        def _target_transform(target):
            if target == 3:
                return 0
            else:
                return 1
        image_size = (224, 224, 3)
        n_classes = 2
        train_dir = os.path.join(P.data_root, "OCT2017", "train")
        test_dir = os.path.join(P.data_root, "OCT2017", "test")
        train_transform, test_transform = get_transform(image_size=image_size)
        train_set = datasets.ImageFolder(
            train_dir, transform=train_transform, target_transform=_target_transform)
        test_set = datasets.ImageFolder(
            test_dir, transform=test_transform, target_transform=_target_transform)

        normal_class = 3
        train_set.targets = list([int(i != normal_class) for i in train_set.targets])
        test_set.targets = list([int(i != normal_class) for i in test_set.targets])

    elif dataset == "aptos":
        def _target_transform(target):
            if target == 0:
                return 0
            else:
                return 1
        image_size = (224, 224, 3)
        n_classes = 4
        train_dir = os.path.join(P.data_root, "APTOS2019/split", "train")
        test_dir = os.path.join(P.data_root, "APTOS2019/split", "test")
        train_transform, test_transform = get_transform(image_size=image_size)
        train_set = datasets.ImageFolder(
            train_dir, transform=train_transform, target_transform=_target_transform)
        test_set = datasets.ImageFolder(
            test_dir, transform=test_transform, target_transform=_target_transform)

    else:
        raise NotImplementedError()

    if test_only:
        return test_set
    else:
        return train_set, test_set, image_size, n_classes


def get_superclass_list(dataset):
    if dataset == 'cifar10':
        return CIFAR10_SUPERCLASS
    elif dataset == 'cifar100':
        return CIFAR100_SUPERCLASS
    elif dataset == 'pneumoniamnist' or dataset == 'breastmnist' or dataset == 'chestmnist' or dataset == 'chestxray' \
        or dataset == 'covid' or dataset == 'kvasir' or dataset == 'breakhis' or dataset == 'retina' or dataset == 'aptos':
        return BINARY_SUPERCLASS
    else:
        raise NotImplementedError()


def get_subclass_dataset(dataset, classes):
    if not isinstance(classes, list):
        classes = [classes]

    indices = []
    for idx, tgt in enumerate(dataset.targets):
        if tgt in classes:
            indices.append(idx)

    dataset = Subset(dataset, indices)
    return dataset


def get_simclr_eval_transform_imagenet(sample_num, resize_factor, resize_fix):

    resize_scale = (resize_factor, 1.0)  # resize scaling factor
    if resize_fix:  # if resize_fix is True, use same scale
        resize_scale = (resize_factor, resize_factor)

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224, scale=resize_scale),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    clean_trasform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    transform = MultiDataTransformList(transform, clean_trasform, sample_num)

    return transform, transform
