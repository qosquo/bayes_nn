import torch
from itertools import islice
from typing import Tuple

from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


def get_dataloaders(
        data_dir: str,
        batch_size: int = 128,
        num_workers: int = 2,
        use_cuda: bool = True,
):
    """
    Returns train_loader, val_loader, test_loader
    """

    # Standard transforms for MNIST
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    # Download + load datasets
    full_train_dataset = datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transform,
    )

    test_dataset = datasets.MNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=transform,
    )

    # Split the training dataset into training and validation sets
    train_size = 50000
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

