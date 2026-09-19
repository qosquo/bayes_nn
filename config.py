import yaml
from torchvision import datasets
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
    kwargs: dict | None
    batch_size: int
    num_workers: int


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