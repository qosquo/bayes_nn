import os
from pathlib import Path

import click
import torch
from torchvision.transforms import transforms

from models.lenet import Net
from config import build_transform, load_config, get_dataset
from evaluate import evaluate
from utils.checkpoint import load_checkpoint
from utils.data import get_datasets, get_dataloaders
from utils.results import update_results


MC_SAMPLES = 10
KERNEL_SIZES = (1, 3, 5, 7, 9)

DEFAULT_SEED = 42  # runs without a seed suffix in their directory name
SEEDS = (DEFAULT_SEED, 21, 63)

# [(key, config, run directory), ...]; checkpoints are "<run dir>_seed_<seed>/checkpoints/best.pt"
RUNS = (
    ("mnist", "configs/mnist/lenet_baseline.yaml", "runs/lenet_mnist_baseline_v4"),
    ("emnist", "configs/emnist-letters/lenet_baseline.yaml", "runs/lenet_emnist-letters_baseline_v2"),
    ("emnist-10", "configs/emnist-letters-10/lenet.yaml", "runs/lenet_emnist-letters-10_v2"),
    ("emnist-18", "configs/emnist-letters-18/lenet.yaml", "runs/lenet_emnist-letters-18_v2"),
)


def evaluate_checkpoint(key: str, config_path: str, checkpoint_path: str, output: str, seed: int) -> None:
    """Evaluates one checkpoint for every kernel size and merges the results into `output`."""
    config = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Net(
        prior_sigma1=config.prior.sigma1,
        prior_sigma2=config.prior.sigma2,
        prior_pi=config.prior.pi,
        num_classes=config.model.num_classes,
    ).to(device)
    load_checkpoint(checkpoint_path, model, device=device)

    dataset_transform = build_transform(config.data.dataset_transform) if config.data.dataset_transform else None

    for kernel_size in KERNEL_SIZES:
        train_dataset, test_dataset = get_datasets(
            root="./data",
            dataset=get_dataset(config.data.dataset),
            normalize=True,
            dataset_kwargs=config.data.kwargs,
            extra_transforms=[transforms.GaussianBlur(kernel_size=kernel_size)],
        )

        _, _, test_loader = get_dataloaders(
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            batch_size=config.data.batch_size,
            num_workers=config.data.num_workers,
            use_cuda=torch.cuda.is_available(),
            dataset_transform=dataset_transform,
        )

        click.echo(f">>> seed={seed} {key} ks={kernel_size}")
        preds, uncertainties = evaluate(model, test_loader, device, mc_samples=MC_SAMPLES)

        update_results(
            output,
            key=key,
            kernel_size=kernel_size,
            mc_samples=MC_SAMPLES,
            evals=(preds, *uncertainties),
        )


def main() -> None:
    # Dataset, checkpoint and output paths are relative to the repository root.
    os.chdir(Path(__file__).resolve().parent.parent)

    for seed in SEEDS:
        output = f"npz/mnist_vs_emnist-letters_comparison_seed_{seed}.npz"
        Path(output).unlink(missing_ok=True)  # clean start

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        suffix = "" if seed == DEFAULT_SEED else f"_seed_{seed}"
        for key, config_path, run_dir in RUNS:
            checkpoint_path = f"{run_dir}{suffix}/checkpoints/best.pt"
            evaluate_checkpoint(key, config_path, checkpoint_path, output, seed)


if __name__ == "__main__":
    main()
