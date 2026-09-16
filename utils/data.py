import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, datasets
from torchvision.datasets import VisionDataset, MNIST, EMNIST, CIFAR10, CIFAR100
from itertools import islice
from pathlib import Path
from typing import Any, Callable, Sequence

from utils import import_attr

class ClassSubset(Dataset):
    """Dataset containing selected classes, remapped to 0..N-1."""

    def __init__(
        self,
        dataset: Dataset,
        targets: Sequence[int] | torch.Tensor,
        selected_classes: list[int],
    ) -> None:
        self.dataset = dataset
        self.targets = targets

        classes = sorted(selected_classes)
        self.label_map = {
            raw_label: new_label
            for new_label, raw_label in enumerate(classes)
        }
        self.indices = [
            index
            for index, target in enumerate(targets)
            if int(target) in self.label_map
        ]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        source_index = self.indices[index]
        image, _ = self.dataset[source_index]
        raw_label = int(self.targets[source_index])
        return image, self.label_map[raw_label]


def filter_classes(dataset: Dataset, selected_classes: list[int]) -> Dataset:
    targets = getattr(dataset, "targets")
    return ClassSubset(dataset, targets, selected_classes)


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


def get_dataloaders(
        data_dir: str,
        dataset: Callable[..., VisionDataset] = MNIST,
        batch_size: int = 128,
        num_workers: int = 2,
        use_cuda: bool = True,
        val_split: float = 0.1,
        normalize: bool = True,
        download: bool = True,
        extra_transforms: list | None = None,
        dataset_kwargs: dict | None = None,
        dataset_transform: Callable[[Dataset], Dataset] | None = None,
        train_size: int | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """

    Args:
        data_dir (str): directory of data
        batch_size (int):
        num_workers (int):
        use_cuda (bool):
        extra_transforms (callable, optional): A function/transforms that takes in
            an image and a label and returns the transformed versions of both.
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
    if dataset is EMNIST and "split" not in dataset_kwargs:
        dataset_kwargs["split"] = "balanced"

    if dataset is EMNIST and dataset_kwargs.get("split") == "letters":
        dataset_kwargs.setdefault("target_transform", lambda y: y - 1)

    transform_list: list[Any] = [transforms.ToTensor()]

    if normalize:
        mean, std = _default_normalize_stats(dataset)
        transform_list.append(transforms.Normalize(mean, std))

    if extra_transforms:
        transform_list.extend(extra_transforms)

    transform = transforms.Compose(transform_list)

    # Download + load datasets
    full_train_dataset = dataset(
        root=data_dir,
        train=True,
        download=download,
        transform=transform,
        **dataset_kwargs,
    )

    test_dataset = dataset(
        root=data_dir,
        train=False,
        download=download,
        transform=transform,
        **dataset_kwargs,
    )

    if dataset_transform is not None:
        full_train_dataset = dataset_transform(full_train_dataset)
        test_dataset = dataset_transform(test_dataset)

    # Split the training dataset into training and validation sets
    if train_size is None:
        if dataset is MNIST:
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
