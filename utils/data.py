from functools import reduce, partial

import torch
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torchvision import transforms, datasets
from torchvision.datasets import VisionDataset, MNIST, EMNIST, CIFAR10, CIFAR100
from itertools import islice
from pathlib import Path
from typing import Any, Callable, Sequence

from config import load_callable
from utils import import_attr

def filter_classes(dataset: Dataset, selected_classes: list[int]) -> Subset:
    targets = getattr(dataset, "targets").detach().clone()
    combined = reduce(torch.logical_or, [targets == i for i in selected_classes])
    return Subset(dataset, indices=targets.where(combined, 0.).nonzero().squeeze())

def map_classes(y: Any, class_mapping: dict[Any, Any]) -> Any:
    return class_mapping[y]

def decrease_target_by_one(y: int) -> int:
    return y - 1

def _default_normalize_stats(dataset: Callable[..., VisionDataset]) -> tuple[tuple[float, ...], tuple[float, ...]]:
    if dataset is MNIST:
        return (0.1307,), (0.3081,)
    if dataset is EMNIST:
        return (0.1722,), (0.3309,)
    if dataset is CIFAR10:
        return (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
    if dataset is CIFAR100:
        return (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
    return (0.1307,), (0.3081,)

def get_datasets(
        root: str,
        dataset: Callable[..., VisionDataset] = MNIST,
        normalize: bool = True,
        download: bool = True,
        extra_transforms: list | None = None,
        dataset_kwargs: dict | None = None,
) -> tuple[Dataset, Dataset]:
    """
    Args:
        extra_transforms:
        root (str): directory of data
        dataset (callable): class of torchvision dataset (default: "MNIST").
        dataset_kwargs (dict, optional): extra args passed to the dataset constructor (e.g., EMNIST split).
        dataset_transform (callable, optional): Example::

            dataset_transform=partial(
                filter_classes,
                selected_classes=[1, 5, 13],  # Raw EMNIST labels: A, E, M
            )
        normalize (bool): whether to apply default normalization stats for the dataset.
        download (bool):
        extra_transforms (callable, optional): A function/transforms that takes in
            an image and a label and returns the transformed versions of both.
    Returns:
        tuple: (train_dataset, test_dataset)
    """
    """

    Args:
        data_dir (str): directory of data
        batch_size (int):
        num_workers (int):
        use_cuda (bool):
        dataset (callable): class of torchvision dataset (default: "MNIST").
        dataset_kwargs (dict, optional): extra args passed to the dataset constructor (e.g., EMNIST split).
        dataset_transform (callable, optional): Example::

            dataset_transform=partial(
                filter_classes,
                selected_classes=[1, 5, 13],  # Raw EMNIST labels: A, E, M
            )
        val_split (float): validation split ratio used when train_size is None.
        train_size (int, optional): fixed train split size for the training subset; overrides val_split.
        normalize (bool): whether to apply default normalization stats for the dataset.
        download (bool):

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """

    dataset_kwargs = dict(dataset_kwargs or {})

    if "target_transform" in dataset_kwargs:
        target_transform = dataset_kwargs.get("target_transform")
        func = load_callable(target_transform["target"])
        params = target_transform.get("params", {})
        dataset_kwargs["target_transform"] = lambda y: func(y, **params)

    transform_list: list[Any] = [transforms.ToTensor()]

    if normalize:
        mean, std = _default_normalize_stats(dataset)
        transform_list.append(transforms.Normalize(mean, std))

    if extra_transforms:
        transform_list.extend(extra_transforms)

    transform = transforms.Compose(transform_list)

    # Download + load datasets
    train_dataset = dataset(
        root=root,
        train=True,
        download=download,
        transform=transform,
        **dataset_kwargs,
    )

    test_dataset = dataset(
        root=root,
        train=False,
        download=download,
        transform=transform,
        **dataset_kwargs,
    )

    return train_dataset, test_dataset

def get_dataloaders(
        train_dataset: Dataset,
        test_dataset: Dataset,
        batch_size: int = 128,
        num_workers: int = 2,
        use_cuda: bool = True,
        dataset_transform: Callable | None = None,
        val_split: float = 0.1,
        train_size: int | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """

    Args:
        train_dataset:
        test_dataset:
        batch_size:
        num_workers:
        use_cuda:
        dataset_transform:
        val_split:
        train_size:

    Returns:

    """

    if dataset_transform is not None:
        train_dataset = dataset_transform(train_dataset)
        test_dataset = dataset_transform(test_dataset)

    # Split the training dataset into training and validation sets
    if train_size is None:
        if train_dataset is MNIST:
            train_size = 50000
        else:
            train_size = int(len(train_dataset) * (1.0 - val_split))
            train_size = max(1, min(train_size, len(train_dataset) - 1))

    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

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


def get_dataloaders_from_folder(
        data_dir: str,
        batch_size: int = 128,
        num_workers: int = 2,
        use_cuda: bool = True,
        transform_file: str | None = None,
        val_split: float = 0.1,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Load datasets from ImageFolder directory structure.

    Expected layout:
        data_dir/train/   (required)
        data_dir/test/    (required)
        data_dir/val/     (optional — otherwise val_split from train)

    Args:
        data_dir: Root data directory.
        batch_size: Batch size for all loaders.
        num_workers: DataLoader workers.
        use_cuda: Enable pin_memory.
        transform_file: Path to .py file with a `transform` variable.
                        If None, uses Compose([ToTensor()]).
        val_split: Fraction of train to use for validation if no val/ folder.

    Returns:
        (train_loader, val_loader, test_loader)
    """
    data_path = Path(data_dir)
    train_dir = data_path / "train"
    test_dir = data_path / "test"
    val_dir = data_path / "val"

    if not train_dir.is_dir():
        raise FileNotFoundError(f"Train directory not found: {train_dir}")
    if not test_dir.is_dir():
        raise FileNotFoundError(f"Test directory not found: {test_dir}")

    if transform_file is not None:
        transform = import_attr(transform_file, "transform")
    else:
        transform = transforms.Compose([transforms.ToTensor()])

    train_dataset = datasets.ImageFolder(root=str(train_dir), transform=transform)
    test_dataset = datasets.ImageFolder(root=str(test_dir), transform=transform)

    if val_dir.is_dir():
        val_dataset = datasets.ImageFolder(root=str(val_dir), transform=transform)
    else:
        train_size = int(len(train_dataset) * (1.0 - val_split))
        val_size = len(train_dataset) - train_size
        train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    kwargs = {"num_workers": num_workers, "pin_memory": True} if use_cuda else {}

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, **kwargs)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **kwargs)

    return train_loader, val_loader, test_loader


def get_img_from_loader(loader: DataLoader, batch_idx: int = 0, img_idx: int = 0, device: str = 'cpu') \
        -> tuple[torch.Tensor, int]:
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
