import torch
import os


def save_checkpoint(model, optimizer, epoch, path):
    """
    Saves model + optimizer state + epoch number.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    checkpoint = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "epoch": epoch,
    }

    torch.save(checkpoint, path)
    print(f"[checkpoint] Saved to {path}")


def load_checkpoint(model, optimizer, path, device):
    """
    Loads checkpoint.
    Returns start_epoch (so you can resume training).
    """
    if not os.path.isfile(path):
        print(f"[checkpoint] No checkpoint found at {path}. Starting fresh.")
        return 0

    checkpoint = torch.load(path, map_location=device)

    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    start_epoch = checkpoint["epoch"] + 1

    print(f"[checkpoint] Loaded from {path}, starting at epoch {start_epoch}")
    return start_epoch


def load_model(model, path, device):
    """
    Loads model.
    """
    if not os.path.isfile(path):
        print(f"[checkpoint] No model found at {path}. Starting fresh.")
        return 0

    checkpoint = torch.load(path, map_location=device)

    model.load_state_dict(checkpoint["model_state"])
    print(f"[model] Loaded from {path}")
    return None
