import argparse
from multiprocessing import Manager
import os
import datetime
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR100, DatasetFolder
from torchvision.datasets import ImageFolder
from torchvision.utils import save_image
import random
import numpy as np

from denoising_diffusion_pytorch import Trainer as _Trainer
from denoising_diffusion_pytorch.denoising_diffusion_pytorch import (
    Accelerator,
    has_int_squareroot,
    cycle,
    Adam,
    EMA,
    Path,
)
from train_eval_ddpm_unet import Unet, GaussianDiffusion

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')


def _seed_all(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)


def get_subset_sampler(dataset, class_names):
    indices = []
    for i in range(len(dataset.targets)):
        if dataset.targets[i] in class_names:
            indices.append(i)
    return torch.utils.data.sampler.SubsetRandomSampler(indices)


def remove_labels_collate_fn(data):
    a = []
    for i in range(len(data)):
        a.append(data[i][0])

    return torch.stack(a)


class CIFAR100Coarse(CIFAR100):
    """Create CIFAR100 dataset with coarse classes.

    Code refers to: https://github.com/ryanchankh/cifar100coarse/blob/master/cifar100coarse.py

    Args:
        root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """

    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        super(CIFAR100Coarse, self).__init__(
            root, train=train, transform=transform, target_transform=target_transform, download=download)

        # update labels
        coarse_labels = np.array([
            4, 1, 14, 8, 0, 6, 7, 7, 18, 3,
            3, 14, 9, 18, 7, 11, 3, 9, 7, 11,
            6, 11, 5, 10, 7, 6, 13, 15, 3, 15,
            0, 11, 1, 10, 12, 14, 16, 9, 11, 5,
            5, 19, 8, 8, 15, 13, 14, 17, 18, 10,
            16, 4, 17, 4, 2, 0, 17, 4, 18, 17,
            10, 3, 2, 12, 12, 16, 12, 1, 9, 19,
            2, 10, 0, 1, 16, 12, 9, 13, 15, 13,
            16, 19, 2, 4, 6, 19, 5, 5, 8, 19,
            18, 1, 2, 15, 6, 0, 17, 8, 14, 13
        ])
        self.targets = coarse_labels[self.targets]

        # update classes
        self.classes = [['beaver', 'dolphin', 'otter', 'seal', 'whale'],
                        ['aquarium_fish', 'flatfish', 'ray', 'shark', 'trout'],
                        ['orchid', 'poppy', 'rose', 'sunflower', 'tulip'],
                        ['bottle', 'bowl', 'can', 'cup', 'plate'],
                        ['apple', 'mushroom', 'orange', 'pear', 'sweet_pepper'],
                        ['clock', 'keyboard', 'lamp', 'telephone', 'television'],
                        ['bed', 'chair', 'couch', 'table', 'wardrobe'],
                        ['bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach'],
                        ['bear', 'leopard', 'lion', 'tiger', 'wolf'],
                        ['bridge', 'castle', 'house', 'road', 'skyscraper'],
                        ['cloud', 'forest', 'mountain', 'plain', 'sea'],
                        ['camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo'],
                        ['fox', 'porcupine', 'possum', 'raccoon', 'skunk'],
                        ['crab', 'lobster', 'snail', 'spider', 'worm'],
                        ['baby', 'boy', 'girl', 'man', 'woman'],
                        ['crocodile', 'dinosaur', 'lizard', 'snake', 'turtle'],
                        ['hamster', 'mouse', 'rabbit', 'shrew', 'squirrel'],
                        ['maple_tree', 'oak_tree', 'palm_tree', 'pine_tree', 'willow_tree'],
                        ['bicycle', 'bus', 'motorcycle', 'pickup_truck', 'train'],
                        ['lawn_mower', 'rocket', 'streetcar', 'tank', 'tractor']]

        self.class_to_idx = {
            "aquatic mammals": 0,
            "fish": 1,
            "flowers": 2,
            "food containers": 3,
            "fruit and vegetables": 4,
            "household electrical devices": 5,
            "household furniture": 6,
            "insects": 7,
            "large carnivores": 8,
            "large man-made outdoor things": 9,
            "large natural outdoor scenes": 10,
            "large omnivores and herbivores": 11,
            "medium-sized mammals": 12,
            "non-insect invertebrates": 13,
            "people": 14,
            "reptiles": 15,
            "small mammals": 16,
            "trees": 17,
            "vehicles 1": 18,
            "vehicles 2": 19,
        }


class ClassSpecificImageFolder(DatasetFolder):
    def __init__(
            self,
            root,
            dropped_classes=[],
            transform = None,
            target_transform = None,
            loader = torchvision.datasets.folder.default_loader,
            is_valid_file = None,
    ):
        self.dropped_classes = dropped_classes
        super(ClassSpecificImageFolder, self).__init__(root, loader, IMG_EXTENSIONS if is_valid_file is None else None,
                                                       transform=transform,
                                                       target_transform=target_transform,
                                                       is_valid_file=is_valid_file)
        self.imgs = self.samples

    def find_classes(self, directory):
        classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
        classes = [c for c in classes if c not in self.dropped_classes]
        if not classes:
            raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")

        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx


sample_dict = Manager().dict()

class MemoryFolder(ImageFolder):

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        if path in sample_dict:
            sample = sample_dict[path]
        else:
            sample = self.loader(path)
            sample_dict.update({path: sample})
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target



class Trainer(_Trainer):
    def __init__(
        self,
        diffusion_model,
        data_loader,
        *args,
        train_batch_size=16,
        gradient_accumulate_every=1,
        augment_horizontal_flip=True,
        train_lr=1e-4,
        train_num_steps=100000,
        ema_update_every=10,
        ema_decay=0.995,
        adam_betas=(0.9, 0.99),
        save_and_sample_every=1000,
        num_samples=25,
        results_folder="./results",
        amp=False,
        fp16=False,
        split_batches=True,
        convert_image_to=None
    ):
        # NOTE: Override the constructor. Do not call super().

        self.accelerator = Accelerator(
            split_batches=split_batches, mixed_precision="fp16" if fp16 else "no"
        )

        self.accelerator.native_amp = amp

        self.model = diffusion_model

        assert has_int_squareroot(
            num_samples
        ), "number of samples must have an integer square root"
        self.num_samples = num_samples
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every

        self.train_num_steps = train_num_steps
        self.image_size = diffusion_model.image_size

        dl = self.accelerator.prepare(data_loader)
        self.dl = cycle(dl)

        # optimizer

        self.opt = Adam(diffusion_model.parameters(), lr=train_lr, betas=adam_betas)

        # for logging results in a folder periodically

        if self.accelerator.is_main_process:
            self.ema = EMA(
                diffusion_model, beta=ema_decay, update_every=ema_update_every
            )

            self.results_folder = Path(results_folder)
            self.results_folder.mkdir(exist_ok=True)

        # step counter state

        self.step = 0

        # prepare model, dataloader, optimizer with accelerator

        self.model, self.opt = self.accelerator.prepare(self.model, self.opt)


def get_dataset(name, transform, image_size=None):

    def _make_ratio(train_set, ratio):
        if ratio == 1:
            return train_set

        if isinstance(train_set, (ImageFolder,)) and ratio != 1:
            index = np.arange(len(train_set.targets))
            np.random.shuffle(index)
            index = index[:int(len(train_set.targets) * ratio)]
            train_set.samples = [s for i, s in enumerate(train_set.samples) if i in index]
            train_set.imgs = [s for i, s in enumerate(train_set.imgs) if i in index]
            train_set.targets = [s for i, s in enumerate(train_set.targets) if i in index]
        else:
            raise NotImplementedError
        print(train_set.class_to_idx, len(train_set))
        return train_set

    if name == "cifar10":
        assert image_size is None
        image_size = 32
        train_set = torchvision.datasets.CIFAR10(
            root="./data",
            train=True,
            download=True,
            transform=transforms.Compose(transform),
        )
    elif name == "cifar100":
        assert image_size is None
        image_size = 32
        train_set = CIFAR100Coarse(
            root="./data",
            train=True,
            download=True,
            transform=transforms.Compose(transform),
        )
    elif name == "pneumoniamnist" or name == "breastmnist" or name == "chestmnist":
        import medmnist

        transform.append(transforms.Pad(2))
        assert image_size is None
        image_size = 32
        cls = getattr(medmnist, medmnist.INFO[name]["python_class"])
        train_set = cls(
            root="./data",
            split="train",
            transform=transforms.Compose(transform),
            download=True,
            as_rgb=True,
        )
        train_set.labels = train_set.labels.squeeze()
        if name == "chestmnist":  # convert multiclass to binary
            train_set.labels = train_set.labels.any(1).astype(int)
        # align with torchvision conventions
        train_set.targets = train_set.labels
        if args.ratio != 1:
            index = np.arange(len(train_set.targets))
            np.random.shuffle(index)
            index = index[:int(len(train_set.labels) * args.ratio)]
            train_set.imgs = train_set.imgs[index]
            train_set.labels = train_set.labels[index]
            train_set.targets = train_set.targets[index]
        print(len(train_set))
    elif name == "sars-covid":
        image_size = 32 if image_size is None else image_size
        transform.append(transforms.Resize((image_size, image_size)))
        train_set = ImageFolder("../data/SARS-COV-2", transform=transforms.Compose(transform))
        _make_ratio(train_set, ratio=args.ratio)
    elif name == "kvasir":
        image_size = 32 if image_size is None else image_size
        transform.append(transforms.Resize((image_size, image_size)))
        train_set = ImageFolder("../git/Diffusion_Anomaly/data/kvasir-anomaly-polyps/train", transform=transforms.Compose(transform))
        _make_ratio(train_set, ratio=args.ratio)
    elif name == "chestxray":
        image_size = 32 if image_size is None else image_size
        transform.append(transforms.Resize((image_size, image_size)))
        train_set = ImageFolder("../data/chest_xray/train", transform=transforms.Compose(transform))
        # train_set = ClassSpecificImageFolder(
        #     "../data/chest_xray/train", dropped_classes=["PNEUMONIA"], transform=transforms.Compose(transform))
    elif name == "breakhis":
        image_size = 32 if image_size is None else image_size
        transform.append(transforms.Resize((image_size, image_size)))
        train_set = ImageFolder("../git/Diffusion_Anomaly/data/BreaKHis400X/train", transform=transforms.Compose(transform))
        _make_ratio(train_set, ratio=args.ratio)
    elif name == "catsdogs":
        image_size = 32 if image_size is None else image_size
        transform.append(transforms.Resize((image_size, image_size)))
        train_set = ImageFolder("../data/CatsVSDogs/train/train", transform=transforms.Compose(transform))
        _make_ratio(train_set, ratio=args.ratio)
    elif name == "retina":
        image_size = 32 if image_size is None else image_size
        transform.append(transforms.Resize((image_size, image_size)))
        train_set = ImageFolder("../data/OCT2017/train", transform=transforms.Compose(transform))
        # train_set = ClassSpecificImageFolder(
        #     "../data/OCT2017/train", dropped_classes=["CNV", "DME", "DRUSEN"], transform=transforms.Compose(transform))
        # assert False, (len(train_set), train_set.class_to_idx)
    elif name == "aptos":
        image_size = 32 if image_size is None else image_size
        transform.append(transforms.Resize((image_size, image_size)))
        train_set = MemoryFolder("../data/APTOS2019/split/train", transform=transforms.Compose(transform))
        # train_set = ClassSpecificImageFolder(
        #     "../data/OCT2017/train", dropped_classes=["CNV", "DME", "DRUSEN"], transform=transforms.Compose(transform))
        # assert False, (len(train_set), train_set.class_to_idx)
    else:
        raise NotImplementedError(name)
    return train_set, image_size


def get_model(image_size, dilation: bool = False, local_randn: bool = False):
    if dilation:
        model = Unet(dim=64, dim_mults=(1, 2, 4, 8), dilation=image_size // 32).cuda()
    else:
        model = Unet(dim=64, dim_mults=(1, 2, 4, 8), dilation=1).cuda()

    diffusion = GaussianDiffusion(
        model,
        image_size=image_size,
        timesteps=1000,  # number of steps
        sampling_timesteps=250,  # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
        loss_type="l1",  # L1 or L2
        local_randn=local_randn,
    ).cuda()
    return diffusion


def train(dataset, cls_names, output_dir, args):

    augment_horizontal_flip = True
    _factor = 2  # scale the batch size for large GPU memories
    _factor = 16  # scale the batch size for large GPU memories
    train_batch_size = 16 * _factor

    transform = [transforms.ToTensor()]
    if augment_horizontal_flip:
        transform.append(transforms.RandomHorizontalFlip())

    ds, image_size = get_dataset(dataset, transform, args.image_size)
    # dataset and dataloader
    dl = torch.utils.data.DataLoader(
        ds,
        batch_size=train_batch_size,
        shuffle=False,
        pin_memory=False,
        num_workers=12,
        sampler=get_subset_sampler(ds, class_names=cls_names),
        collate_fn=remove_labels_collate_fn,
    )

    diffusion = get_model(image_size, args.dilation, args.local_randn)
    if args.state_dict_path is not None:
        diffusion.load_state_dict(torch.load(args.state_dict_path)["model"])
    _seed_all(0)
    os.makedirs(output_dir, exist_ok=True)
    trainer = Trainer(
        diffusion,
        data_loader=dl,
        train_batch_size=train_batch_size,
        train_lr=8e-5,
        train_num_steps=500000 * 10 // _factor,
        save_and_sample_every=500000 // _factor,
        gradient_accumulate_every=2,  # gradient accumulation steps
        ema_decay=0.995,  # exponential moving average decay
        amp=False,  # NOTE: MUST turn off mixed precision, otherwise it won't converge
        results_folder=output_dir,
    )

    trainer.train()


def eval(dataset, cls_names, state_dict_path, args):

    batch_size = 64

    transform = [transforms.ToTensor()]

    ds, image_size = get_dataset(dataset, transform, args.image_size)
    # dataset and dataloader
    dl = torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=24,
        sampler=get_subset_sampler(ds, class_names=cls_names),
        collate_fn=remove_labels_collate_fn,
    )

    diffusion = get_model(image_size, args.dilation, args.local_randn)
    diffusion.load_state_dict(torch.load(state_dict_path)["model"])

    steps = args.steps
    for i, x in enumerate(dl):
        l = diffusion.p_sample(x.cuda(), steps)
        save_image(l[0], str(f"ddmp_p_sample-{dataset}-step_{steps}-pred_img.png"), nrow=int(batch_size**0.5))
        save_image(l[1], str(f"ddmp_p_sample-{dataset}-step_{steps}-x_start.png"), nrow=int(batch_size**0.5))
        save_image(x, str(f"ddmp_p_sample-{dataset}-step_{steps}-input.png"), nrow=int(batch_size**0.5))
        break


def get_dataset_config(dataset, image_size=None):
    timestamp = str(datetime.datetime.now().strftime("-%Y%m%d-%H%M%S"))
    image_size = 32 if image_size is None else image_size
    if dataset == "cifar10" or dataset == "cifar100":
        return dataset, f"ddpm_models/cifar/{dataset}-{image_size}" + timestamp
    elif dataset == "pneumonia" or dataset == "breast" or dataset == "chest":
        return dataset + "mnist", f"ddpm_models/medmnist/{dataset}-{image_size}" + timestamp
    else:
        return dataset, f"ddpm_models/{dataset}-{image_size}" + timestamp


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "eval"])
    parser.add_argument("--dataset", choices=["pneumonia", "breast", "chest", "chestxray", "cifar10", "cifar100", "sars-covid", "kvasir", "breakhis", "catsdogs", "retina", "aptos", "imagenet"])
    parser.add_argument("--on_label", type=int, default=None)
    parser.add_argument("--image_size", type=int, default=None)
    parser.add_argument("--dilation", action="store_true", help="if to use dilation in unet.")
    parser.add_argument("--local_randn", action="store_true", help="if to use local randn in diffusion net.")
    parser.add_argument("--steps", type=int, default=100, help="evaluation arguments.")
    parser.add_argument("--seed", type=int, default=None, help="random seed")
    parser.add_argument("--state_dict_path", type=str, default=None, help="state dict")
    parser.add_argument("--ratio", type=float, default=1.)
    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)
        torch.random.manual_seed(args.seed)

    name, output_dir = get_dataset_config(args.dataset, args.image_size)

    if args.ratio != 1:
        output_dir = output_dir + f"_r{args.ratio}"

    if args.mode == "train" and (args.dataset == "cifar10" or args.dataset == "cifar100" or args.dataset == "catsdogs" or args.dataset == "imagenet"):
        if args.on_label is None and args.dataset == "cifar10":
            on_label = list(range(10))
        elif args.on_label is None and args.dataset == "cifar100":
            on_label = list(range(20))
        elif args.on_label is None and args.dataset == "catsdogs":
            on_label = list(range(2))
        elif args.on_label is None and args.dataset == "imagenet":
            on_label = list(range(1000))
        else:
            on_label = [args.on_label]
        train(name, on_label, output_dir + f"_{args.on_label}", args)
    elif args.mode == "train":
        assert args.on_label is None, "You shall point the corresponding like below."
        class_names = [0]
        if args.dataset == "sars-covid":
            class_names = [1]  # {'COVID': 0, 'non-COVID': 1}
        if args.dataset == "retina":
            class_names = [3]  # {'CNV': 0, 'DME': 1, 'DRUSEN': 2, 'NORMAL': 3}
        if args.dataset == "chestxray":
            class_names = [0]  # {'NORMAL': 0, 'PNEUMONIA': 1}
        if args.dataset == "aptos":
            class_names = [0]  # {'NORMAL': 0, 'PNEUMONIA': 1}
        print(output_dir, class_names)
        train(name, class_names, output_dir, args)
    else:
        eval(
            name,
            [0, 1],
            "./ddpm_models/kvasir-20221102-204753/model-5.pt",
            args
        )
