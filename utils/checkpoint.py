import os
import click
import torch
import torch.nn as nn
from torch.optim import Optimizer


def save_checkpoint(
        path: str,
        epoch: int,
        model: nn.Module,
        optimizer: Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        config: dict,
        best_val_loss: float,
        seed: int,
        **kwargs
) -> None:
    """Saves model + optimizer state + epoch number."""
    os.makedirs(os.path.dirname(path), exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),

        "best_val_loss": best_val_loss,

        "config": config,
        "seed": seed,
        **kwargs
    }

    torch.save(checkpoint, path)
    click.echo(f"Checkpoint saved to {path} at epoch {epoch}.")

def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    device: str | torch.device = "cpu"
) -> dict:
    checkpoint = torch.load(
        path,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    click.echo(f"Checkpoint loaded from {path} at epoch {checkpoint['epoch']}.")
    return checkpoint
