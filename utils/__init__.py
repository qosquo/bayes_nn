import importlib.util
import sys
from pathlib import Path
from typing import Any

import torch


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