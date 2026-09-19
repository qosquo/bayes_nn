from copy import deepcopy

import click
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch import Tensor
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets
from tqdm import tqdm
from datetime import datetime
from pathlib import Path

from config import Config, load_config, get_dataset
from models.lenet import Net
from utils import compute_beta
from utils.calibration import reliability_diagram
from utils.checkpoint import load_checkpoint, save_checkpoint
from utils.data import get_dataloaders

DATASETS = {
    "MNIST": datasets.MNIST,
    "EMNIST": datasets.EMNIST,
    "FashionMNIST": datasets.FashionMNIST,
    "CIFAR10": datasets.CIFAR10,
}


def elbo_loss(output: Tensor, y: Tensor, kl: Tensor | float, beta: float) -> Tensor:
    return F.nll_loss(output, y) + beta * kl


def train(model: nn.Module, optimizer: optim.Optimizer, train_loader: DataLoader,
          device: str | torch.device, epoch: int, grad_clip: float | None = None, mc_samples: int = 1,
          beta_schedule: str = 'blundell', warmup_factor: float = 1.0,
          writer: SummaryWriter | None = None) -> float:
    model.train()
    total_loss = 0
    accuracy = 0

    loop = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)
    M = len(train_loader)

    for batch_idx, (x, y) in enumerate(loop):
        x, y = x.detach().to(device), y.detach().to(device)

        optimizer.zero_grad()

        beta = compute_beta(batch_idx, M, beta_schedule, warmup_factor)

        output_ = []
        kl_ = []
        for _ in range(mc_samples):
            output_.append(F.log_softmax(model(x), dim=1))
            kl_.append(model.kl_divergence())
        output = torch.logsumexp(torch.stack(output_), dim=0) - math.log(mc_samples)
        kl = torch.mean(torch.stack(kl_), dim=0)
        loss = elbo_loss(output, y, kl, beta)
        loss.backward()

        if grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        # Accuracy calculation
        pred = output.argmax(dim=1)
        batch_acc = (pred == y).float().mean().item()
        accuracy += batch_acc

        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())

        # TensorBoard logging
        if writer:
            step = epoch * M + batch_idx
            writer.add_scalar("train/batch_accuracy", batch_acc, step)
            writer.add_scalar("train/loss", loss.item(), step)
            writer.add_scalar("train/nll", F.nll_loss(output, y).item(), step)
            writer.add_scalar("train/kl_divergence", kl.item(), step)

    if writer:
        writer.add_scalar("train/epoch_accuracy", accuracy / M, epoch)

    return total_loss / len(train_loader.dataset)


def test(model: nn.Module, test_loader: DataLoader, device: str | torch.device, epoch: int,
         mc_samples: int = 1, beta_schedule: str = 'blundell', warmup_factor: float = 1.0,
         writer: SummaryWriter | None = None) -> tuple[float, float]:
    model.train()
    test_loss = 0
    test_kl = 0
    correct = 0
    M = len(test_loader)

    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(test_loader):
            x, y = x.detach().to(device), y.detach().to(device)

            beta = compute_beta(batch_idx, M, beta_schedule, warmup_factor)

            output_ = []
            kl_ = []
            for _ in range(mc_samples):
                output_.append(F.log_softmax(model(x), dim=1))
                kl_.append(model.kl_divergence())
            output = torch.logsumexp(torch.stack(output_), dim=0) - math.log(mc_samples)
            kl = torch.mean(torch.stack(kl_), dim=0)
            test_loss += elbo_loss(output, y, kl, beta).item()
            test_kl += beta * kl.item()

            pred = output.argmax(dim=1)
            correct += (pred == y).sum().item()

    test_loss /= len(test_loader.dataset)
    test_kl /= len(test_loader.dataset)
    accuracy = correct / len(test_loader.dataset)

    if writer:
        writer.add_scalar("test/accuracy", accuracy, epoch)
        writer.add_scalar("test/loss", test_loss, epoch)
        writer.add_scalar("test/nll", (test_loss - test_kl), epoch)

    return test_loss, accuracy


@click.command()
@click.option("--config", "config_path", type=str, default=None, help="YAML file describing experiment.")
@click.option("--tensorboard", type=bool, is_flag=True, default=False,
              help="Use tensorboard for logging and visualization of training progress")
@click.option("--log-dir", type=str, default="./runs", show_default=True,
              help="Logging directory.")
@click.option("--data-dir", type=str, default="./data", show_default=True,
              help="Datasets directory.")
@click.option("--save-checkpoint", "save", type=bool, is_flag=True, default=False,
              help="Set this flag to True to save checkpoint every N epochs.")
@click.option("--save-interval", type=int, default=10)
@click.option("--resume", type=str, default=None, show_default=True,
              help="Path to checkpoint to resume from.")
@click.option("--seed", type=int, default=42, help="Random seed for reproducibility.")
def main(config_path: str | None, save: bool, save_interval: int,
         tensorboard: bool, log_dir: str, data_dir: str, resume: str, seed: int) -> None:
    """Train a Bayesian neural network with ELBO loss."""
    torch.manual_seed(seed)
    config: Config = load_config(config_path)
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = Path(log_dir)

    # TensorBoard
    writer = None
    if tensorboard:
        writer = SummaryWriter(
            log_dir=str(run_dir / "tensorboard")
        )

    # Data
    train_loader, val_loader, _ = get_dataloaders(
        data_dir=data_dir,
        batch_size=config.data.batch_size,
        num_workers=config.data.num_workers,
        use_cuda=torch.cuda.is_available(),
        dataset=get_dataset(config.data.dataset),
        dataset_kwargs=config.data.kwargs
    )

    # Model
    model = Net(
        prior_sigma1=config.prior.sigma1,
        prior_sigma2=config.prior.sigma2,
        prior_pi=config.prior.pi,
        num_classes=config.model.num_classes,
        rho_init=config.model.rho_init
    ).to(device)

    # Optimizer & scheduler
    optimizer = optim.Adam(model.parameters(), lr=config.training.learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, patience=config.scheduler.patience)
    best_val_loss = np.inf

    # Resume if checkpoint exists
    start_epoch = 0
    if resume:
        checkpoint = load_checkpoint(
            path=resume,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device
        )
        start_epoch = checkpoint.get("epoch", -1) + 1
        best_val_loss = checkpoint.get("best_val_loss", np.inf)
        if 'rng_state' in checkpoint:
            rng_state = checkpoint['rng_state']
            if device == torch.device("cuda"):
                rng_state = rng_state.cpu().to(torch.uint8)
            torch.set_rng_state(rng_state)

        if 'cuda_rng_state' in checkpoint and checkpoint['cuda_rng_state'] is not None and torch.cuda.is_available():
            cuda_rng_state = checkpoint['cuda_rng_state']
            for i, state in enumerate(cuda_rng_state):
                cuda_rng_state[i] = deepcopy(state.cpu().to(torch.uint8))
            torch.cuda.set_rng_state_all(cuda_rng_state)

    # Training loop
    for epoch in range(start_epoch, config.training.epochs):
        warmup_factor = min(1.0, epoch / config.training.warmup_epochs)
        train_loss = train(
            model=model,
            optimizer=optimizer,
            train_loader=train_loader,
            device=device,
            epoch=epoch,
            grad_clip=config.training.gradient_clip_norm,
            mc_samples=config.training.t_train,
            beta_schedule=config.training.beta_schedule,
            warmup_factor=warmup_factor,
            writer=writer
        )

        val_loss, val_acc = test(
            model=model,
            test_loader=val_loader,
            device=device,
            epoch=epoch,
            mc_samples=config.training.mc_samples,
            beta_schedule=config.training.beta_schedule,
            warmup_factor=warmup_factor,
            writer=writer
        )
        scheduler.step(val_loss)
        click.echo(f"""
Train: loss={train_loss:.6f}
Validation: loss={val_loss:.6f}
Validation Accuracy={val_acc * 100:.2f}%
""")

        # If save flag is not set, skip saving checkpoints
        if not save:
            continue

        # Save model if validation loss has decreased
        checkpoint_kwargs = {
            "epoch": epoch,
            "model": model,
            "optimizer": optimizer,
            "scheduler": scheduler,
            "config": config,
            "best_val_loss": best_val_loss,
            "seed": seed,
            "rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        }

        if val_loss < best_val_loss and epoch > config.training.warmup_epochs:
            click.echo(f"Validation loss decreased ({best_val_loss:.6f} --> {val_loss:.6f}).  Saving model ...")
            best_val_loss = val_loss
            save_checkpoint(
                path=str(run_dir / "checkpoints" / "best.pt"),
                **checkpoint_kwargs
            )

        save_checkpoint(
            path=str(run_dir / "checkpoints" / "last.pt"),
            **checkpoint_kwargs
        )

        if epoch % save_interval == 0:
            save_checkpoint(
                path=str(run_dir / "checkpoints" / f"epoch_{epoch:04d}.pt"),
                **checkpoint_kwargs
            )
            if writer:
                writer.add_figure(
                    'model/reliability_diagram',
                    reliability_diagram(
                        model=model,
                        loader=val_loader,
                        device=device,
                        mc_samples=config.training.mc_samples,
                        num_classes=config.model.num_classes,
                        n_bins=config.model.num_classes
                    ),
                    epoch
                )

    if writer:
        writer.flush()
        writer.close()


if __name__ == "__main__":
    main()
