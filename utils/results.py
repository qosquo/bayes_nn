from pathlib import Path

import numpy as np
from torch import Tensor


def update_results(
        path: str | Path,
        key: str,
        kernel_size: int,
        mc_samples: int,
        evals: tuple[Tensor, Tensor, Tensor, Tensor],
) -> None:
    """Merges one dataset/kernel-size evaluation into an npz results archive."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    results: dict = {}
    if path.exists():
        with np.load(path, allow_pickle=True) as data:
            results = {k: data[k].item() if data[k].dtype == object else data[k].tolist()
                       for k in data.files}

    results["mc_samples"] = mc_samples
    results["kernel_sizes"] = sorted({*results.get("kernel_sizes", []), kernel_size})
    results.setdefault(f"{key.lower()}_evals", {})[kernel_size] = tuple(
        tensor.cpu().numpy() for tensor in evals
    )

    np.savez(path, **results)
