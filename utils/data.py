import torch
from itertools import islice
from typing import Tuple, Optional, Dict

from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


_MNIST_LIKE = {
    "mnist",
    "fashionmnist",
    "kmnist",
    "emnist",
    "qmnist",
}

_DATASET_CLASSES = {
    "mnist": datasets.MNIST,
    "fashionmnist": datasets.FashionMNIST,
    "kmnist": datasets.KMNIST,
    "emnist": datasets.EMNIST,
    "qmnist": datasets.QMNIST,
    "cifar10": datasets.CIFAR10,
    "cifar100": datasets.CIFAR100,
    "svhn": datasets.SVHN,
}


def _default_normalize_stats(dataset_key: str) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
    if dataset_key == "emnist":
        return (0.1722,), (0.3309,)
    if dataset_key in _MNIST_LIKE:
        return (0.1307,), (0.3081,)
    if dataset_key == "cifar10":
        return (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
    if dataset_key == "cifar100":
        return (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
    if dataset_key == "svhn":
        return (0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)
    return (0.1307,), (0.3081,)


def _build_transform(
        dataset_key: str,
        extra_transforms: Optional[list],
        normalize: bool,
) -> transforms.Compose:
    transform_list = [transforms.ToTensor()]

    if normalize:
        mean, std = _default_normalize_stats(dataset_key)
        transform_list.append(transforms.Normalize(mean, std))

    if extra_transforms:
        transform_list.extend(extra_transforms)

    return transforms.Compose(transform_list)


def get_dataloaders(
        data_dir: str,
        batch_size: int = 128,
        num_workers: int = 2,
        use_cuda: bool = True,
        extra_transforms: list = None,
        dataset: str = "MNIST",
        dataset_kwargs: Optional[Dict] = None,
        val_split: float = 0.1,
        train_size: Optional[int] = None,
        normalize: bool = True,
        download: bool = True,
):
    """
    Returns train_loader, val_loader, test_loader

    dataset: name of torchvision dataset (default: "MNIST").
        Supported: MNIST, FashionMNIST, KMNIST, EMNIST, QMNIST, CIFAR10, CIFAR100, SVHN
    dataset_kwargs: extra args passed to the dataset constructor (e.g., EMNIST split).
    val_split: validation split ratio used when train_size is None.
    train_size: fixed train split size for the training subset; overrides val_split.
    normalize: whether to apply default normalization stats for the dataset.
    """

    dataset_key = dataset.strip().lower()
    if dataset_key not in _DATASET_CLASSES:
        supported = ", ".join(sorted(_DATASET_CLASSES.keys()))
        raise ValueError(f"Unsupported dataset '{dataset}'. Supported: {supported}")

    dataset_kwargs = dict(dataset_kwargs or {})
    if dataset_key == "emnist" and "split" not in dataset_kwargs:
        dataset_kwargs["split"] = "balanced"

    transform = _build_transform(dataset_key, extra_transforms, normalize)

    # Download + load datasets
    if dataset_key == "svhn":
        full_train_dataset = datasets.SVHN(
            root=data_dir,
            split="train",
            download=download,
            transform=transform,
            **dataset_kwargs,
        )

        test_dataset = datasets.SVHN(
            root=data_dir,
            split="test",
            download=download,
            transform=transform,
            **dataset_kwargs,
        )
    else:
        dataset_cls = _DATASET_CLASSES[dataset_key]
        full_train_dataset = dataset_cls(
            root=data_dir,
            train=True,
            download=download,
            transform=transform,
            **dataset_kwargs,
        )

        test_dataset = dataset_cls(
            root=data_dir,
            train=False,
            download=download,
            transform=transform,
            **dataset_kwargs,
        )

    # Split the training dataset into training and validation sets
    if train_size is None:
        if dataset_key == "mnist":
            train_size = 50000
        else:
            train_size = int(len(full_train_dataset) * (1.0 - val_split))
            train_size = max(1, min(train_size, len(full_train_dataset) - 1))

    val_size = len(full_train_dataset) - train_size
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

    kwargs = {"num_workers": num_workers, "pin_memory": True} if use_cuda else {}

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        **kwargs,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        **kwargs,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        **kwargs,
    )

    return train_loader, val_loader, test_loader


def get_img_from_loader(loader: DataLoader, batch_idx: int = 0, img_idx: int = 0, device: str = 'cpu') \
        -> (torch.Tensor, int):
    """
    Get a specific image and label from a DataLoader
    :param loader:
    :param batch_idx:
    :param img_idx:
    :param device:
    :return:
    """
    img, label = next(islice(loader, batch_idx, batch_idx+1))
    img = img.to(device)
    return img[img_idx], label[img_idx]
