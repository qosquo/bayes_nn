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
