from typing import Any

import yaml
import importlib
from functools import partial
from torchvision import datasets, transforms
from dataclasses import dataclass


DATASETS = {
    "MNIST": datasets.MNIST,
    "EMNIST": datasets.EMNIST,
    "FashionMNIST": datasets.FashionMNIST,
    "CIFAR10": datasets.CIFAR10,
}

@dataclass
class ModelConfig:
    name: str
    num_classes: int
    rho_init: float


@dataclass
class DataConfig:
    dataset: str
    batch_size: int
    num_workers: int
    kwargs: dict | None = None
    dataset_transform: dict | None = None


@dataclass
class TrainingConfig:
    epochs: int
    learning_rate: float
    gradient_clip_norm: float
    t_train: int
    mc_samples: int
    beta_schedule: str
    warmup_epochs: int


@dataclass
class PriorConfig:
    sigma1: float
    sigma2: float
    pi: float


@dataclass
class SchedulerConfig:
    type: str
    patience: int


@dataclass
class Config:
    model: ModelConfig
    data: DataConfig
    training: TrainingConfig
    prior: PriorConfig
    scheduler: SchedulerConfig


def load_callable(path: str):
    module_name, func_name = path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, func_name)

def build_transform(config: dict | None) -> partial[Any]:
    if config is None:
        raise ValueError("Transform config cannot be None")

    func = load_callable(config["target"])
    return partial(func, **config.get("params", {}))

def load_config(path: str) -> Config:
    with open(path) as f:
        raw = yaml.safe_load(f)

    return Config(
        model=ModelConfig(**raw["model"]),
        data=DataConfig(**raw["data"]),
        training=TrainingConfig(**raw["training"]),
        prior=PriorConfig(**raw["prior"]),
        scheduler=SchedulerConfig(**raw["scheduler"]),
    )

def get_dataset(name: str):
    try:
        return DATASETS[name]
    except KeyError:
        raise ValueError(f"Unknown dataset: {name}")