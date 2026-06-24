import importlib.util
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms


def import_attr(path: str, attr: str) -> Any:
    """Load a named attribute from a Python file."""
    resolved = Path(path).resolve()
    spec = importlib.util.spec_from_file_location("_user_module", resolved)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module from '{path}'")
    module = importlib.util.module_from_spec(spec)
    parent = str(resolved.parent)
    inserted = parent not in sys.path
    if inserted:
        sys.path.insert(0, parent)
    try:
        spec.loader.exec_module(module)
    finally:
        if inserted and parent in sys.path:
            sys.path.remove(parent)
    if not hasattr(module, attr):
        available = [n for n in dir(module) if not n.startswith("_")]
        raise AttributeError(
            f"'{attr}' not found in '{path}'. Available: {available}"
        )
    return getattr(module, attr)


def compute_beta(batch_idx: int, num_batches: int, schedule: str = 'blundell',
                 warmup_factor: float = 1.0) -> float:
    """Compute KL weight (beta) for ELBO loss."""
    if schedule == 'uniform':
        return 1.0 / num_batches
    elif schedule == 'warmup':
        return warmup_factor / num_batches
    else:  # 'blundell'
        return (2 ** (num_batches - batch_idx - 1)) / (2 ** num_batches - 1)


def get_emnist_letters_loaders(
    data_dir: str,
    num_classes: int,
    batch_size: int = 128,
    num_workers: int = 2,
    use_cuda: bool = True,
    val_split: float = 0.1,
    extra_transforms: list | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Load EMNIST-Letters filtered to the first `num_classes` letters (A, B, ...).

    Raw EMNIST-Letters targets are 1-26; after target_transform (y-1) they become 0-25.
    Filtering to `targets <= num_classes` keeps raw labels 1-N, i.e. letters A through N
    with final labels 0 to num_classes-1.

    extra_transforms: additional transforms inserted before ToTensor (e.g. [GaussianBlur(ks)]).
    """
    transform = transforms.Compose([
        *(extra_transforms or []),
        transforms.ToTensor(),
        transforms.Normalize((0.1722,), (0.3309,)),
    ])
    target_transform = lambda y: y - 1  # noqa: E731

    train_full = datasets.EMNIST(
        root=data_dir, split="letters", train=True, download=True,
        transform=transform, target_transform=target_transform,
    )
    test_full = datasets.EMNIST(
        root=data_dir, split="letters", train=False, download=True,
        transform=transform, target_transform=target_transform,
    )

    # .targets holds raw labels (1-26); keep first num_classes letters
    train_idx = (train_full.targets <= num_classes).nonzero(as_tuple=True)[0].tolist()
    test_idx = (test_full.targets <= num_classes).nonzero(as_tuple=True)[0].tolist()

    train_filtered = Subset(train_full, train_idx)
    test_filtered = Subset(test_full, test_idx)

    train_size = int(len(train_filtered) * (1.0 - val_split))
    val_size = len(train_filtered) - train_size
    train_ds, val_ds = random_split(train_filtered, [train_size, val_size])

    kwargs: dict = {"num_workers": num_workers, "pin_memory": True} if use_cuda else {}
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, **kwargs),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False, **kwargs),
        DataLoader(test_filtered, batch_size=batch_size, shuffle=False, **kwargs),
    )


def load_state_dict(checkpoint_path: str) -> dict:
    """Load state_dict from checkpoint, handling common formats."""
    data = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(data, dict):
        for key in ("model_state", "model_state_dict", "state_dict", "model"):
            if key in data:
                return data[key]
        return data
    raise RuntimeError(
        f"Unexpected checkpoint format: {type(data).__name__}"
    )