import os

import torch
import torch.nn as nn
from torch.optim import Optimizer


def save_checkpoint(model: nn.Module, optimizer: Optimizer, epoch: int, path: str) -> None:
    """Saves model + optimizer state + epoch number."""
    os.makedirs(os.path.dirname(path), exist_ok=True)

    checkpoint = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "epoch": epoch,
    }

    torch.save(checkpoint, path)
    print(f"[checkpoint] Saved to {path}")


def load_checkpoint(
    model: nn.Module, optimizer: Optimizer | None, path: str, device: torch.device,
) -> int:
    """Loads checkpoint. Returns start_epoch (so you can resume training).

    If optimizer is None, only model weights are loaded.
    """
    if not os.path.isfile(path):
        print(f"[checkpoint] No checkpoint found at {path}. Starting fresh.")
        return 0

    checkpoint = torch.load(path, map_location=device)

    model.load_state_dict(checkpoint["model_state"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    start_epoch = checkpoint["epoch"] + 1

    print(f"[checkpoint] Loaded from {path}, starting at epoch {start_epoch}")
    return start_epoch