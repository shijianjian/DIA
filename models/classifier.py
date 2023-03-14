import torch.nn as nn
import torchvision

from models.resnet import ResNet18, ResNet34, ResNet50
from models.resnet_imagenet import resnet18, resnet50
import models.transform_layers as TL


def get_simclr_augmentation(P, image_size):

    # parameter for resizecrop
    resize_scale = (P.resize_factor, 1.0) # resize scaling factor
    if P.resize_fix: # if resize_fix is True, use same scale
        resize_scale = (P.resize_factor, P.resize_factor)

    # Align augmentation
    color_jitter = TL.ColorJitterLayer(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=0.8)
    color_gray = TL.RandomColorGrayLayer(p=0.2)
    resize_crop = TL.RandomResizedCropLayer(scale=resize_scale, size=image_size)

    # Transform define #
    if P.dataset == 'imagenet': # Using RandomResizedCrop at PIL transform
        transform = nn.Sequential(
            color_jitter,
            color_gray,
        )
    else:
        transform = nn.Sequential(
            color_jitter,
            color_gray,
            resize_crop,
        )

    return transform


def get_shift_module(P, eval=False):

    if P.shift_trans_type == 'rotation':
        shift_transform = TL.Rotation()
        K_shift = 4
    elif P.shift_trans_type == 'cutperm':
        shift_transform = TL.CutPerm()
        K_shift = 4
    elif P.shift_trans_type == 'blurgaussian':
        if P.dataset.endswith("mnist"):
            assert P.diff_resolution == 32, "For Medical mnist dataset, the resolution has to be 32."
        # NOTE: using gaussian blur as 
        shift_transform = TL.Blur(
            size=P.kernel_size, gaussian=True, constant_size=P.diff_resolution, resize_to_constant=P.resize_to_constant, method="rotate")
        K_shift = 4
    elif P.shift_trans_type == 'blurmedian':
        if P.dataset.endswith("mnist"):
            assert P.diff_resolution == 32, "For Medical mnist dataset, the resolution has to be 32."
        # NOTE: using gaussian blur as
        shift_transform = TL.Blur(
            size=P.kernel_size, gaussian=False, constant_size=P.diff_resolution, resize_to_constant=P.resize_to_constant, method="rotate")
        K_shift = 4
    elif P.shift_trans_type == 'blurgaussian_cutperm':
        if P.dataset.endswith("mnist"):
            assert P.diff_resolution == 32, "For Medical mnist dataset, the resolution has to be 32."
        # NOTE: using gaussian blur as 
        shift_transform = TL.Blur(
            size=P.kernel_size, gaussian=True, constant_size=P.diff_resolution, resize_to_constant=P.resize_to_constant, method="cutperm")
        K_shift = 4
    elif P.shift_trans_type == 'blurmedian_cutperm':
        if P.dataset.endswith("mnist"):
            assert P.diff_resolution == 32, "For Medical mnist dataset, the resolution has to be 32."
        # NOTE: using gaussian blur as
        shift_transform = TL.Blur(
            size=P.kernel_size, gaussian=False, constant_size=P.diff_resolution, resize_to_constant=P.resize_to_constant, method="cutperm")
        K_shift = 4
    elif P.shift_trans_type == 'diffusion_rotation':
        if P.dataset.endswith("mnist"):
            assert P.diff_resolution == 32, "For Medical mnist dataset, the resolution has to be 32."
            pre_proc = torchvision.transforms.Pad(2)
            post_proc = torchvision.transforms.CenterCrop(28)
        elif (
            P.dataset == "covid" or P.dataset == "kvasir" or P.dataset == "breakhis" or P.dataset == "retina" or P.dataset == "chestxray"
            or P.dataset == "aptos"
        ):
            pre_proc = torchvision.transforms.Resize((P.diff_resolution, P.diff_resolution))
            post_proc = torchvision.transforms.Resize((224, 224))
        else:
            raise NotImplementedError
        shift_transform = TL.DiffusionCustomTrans(
            model_path=P.diffusion_model_path,
            max_range=4,
            pre_process=pre_proc,
            post_process=post_proc, # crop from 32 to 28 for medmnist dataset
            transform="rotation",
        )
        K_shift = 4  # define it as every 100 as a scale
    elif P.shift_trans_type == 'diffusion_cutperm':
        if P.dataset.endswith("mnist"):
            assert P.diff_resolution == 32, "For Medical mnist dataset, the resolution has to be 32."
            pre_proc = torchvision.transforms.Pad(2)
            post_proc = torchvision.transforms.CenterCrop(28)
        elif (
            P.dataset == "covid" or P.dataset == "kvasir" or P.dataset == "breakhis" or P.dataset == "retina" or P.dataset == "chestxray"
            or P.dataset == "aptos"
        ):
            pre_proc = torchvision.transforms.Resize((P.diff_resolution, P.diff_resolution))
            post_proc = torchvision.transforms.Resize((224, 224))
        else:
            raise NotImplementedError
        shift_transform = TL.DiffusionCustomTrans(
            model_path=P.diffusion_model_path,
            max_range=4,
            pre_process=pre_proc,
            post_process=post_proc, # crop from 32 to 28 for medmnist dataset
            transform="cutperm",
        )
        K_shift = 4  # define it as every 100 as a scale
    else:
        shift_transform = nn.Identity()
        K_shift = 1

    if not eval and not ('sup' in P.mode):
        assert P.batch_size == int(128 / K_shift)

    return shift_transform, K_shift


def get_shift_classifer(model, K_shift):

    model.shift_cls_layer = nn.Linear(model.last_dim, K_shift)

    return model


def get_classifier(mode, n_classes=10):
    if mode == 'resnet18':
        classifier = ResNet18(num_classes=n_classes)
    elif mode == 'resnet34':
        classifier = ResNet34(num_classes=n_classes)
    elif mode == 'resnet50':
        classifier = ResNet50(num_classes=n_classes)
    elif mode == 'resnet18_imagenet':
        classifier = resnet18(num_classes=n_classes)
    elif mode == 'resnet50_imagenet':
        classifier = resnet50(num_classes=n_classes)
    else:
        raise NotImplementedError()

    return classifier

