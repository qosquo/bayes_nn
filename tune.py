# tune.py
import functools
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import click
import numpy as np

import optuna
import torch
from optuna.samplers import RandomSampler
from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import EMNIST

from config import Config, load_config, get_dataset
from train import train, test
from models.lenet import Net
from utils.data import get_dataloaders, filter_classes
from utils.calibration import expected_calibration_error
from utils.uncertainty import mc_predict


def objective(
        trial: optuna.trial.Trial,
        config: Config,
        data_dir: str,
        device: torch.device,
        epochs: int,
        writer_dir: Path | None
) -> float:
    writer = None
    if writer_dir is not None:
        writer = SummaryWriter(log_dir=str(writer_dir / f"trial_{trial.number:04d}"))

    config.prior.sigma1 = math.exp(trial.suggest_float('log_prior_sigma1', -2.0, 0.0))
    config.prior.sigma2 = math.exp(trial.suggest_float('log_prior_sigma2', -8.0, -6.0))
    config.prior.pi = trial.suggest_float('prior_pi', 0.2, 0.8)
    config.training.learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-1, log=True)

    train_loader, val_loader, _ = get_dataloaders(
        data_dir=data_dir,
        batch_size=config.data.batch_size,
        num_workers=config.data.num_workers,
        use_cuda=torch.cuda.is_available(),
        dataset=get_dataset(config.data.dataset),
        dataset_kwargs=config.data.kwargs,
    )

    model = Net(
        prior_sigma1=config.prior.sigma1,
        prior_sigma2=config.prior.sigma2,
        prior_pi=config.prior.pi,
        num_classes=config.model.num_classes,
        rho_init=config.model.rho_init
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=config.training.learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, patience=config.scheduler.patience)


    val_loss = 0.0
    for epoch in range(0, epochs):
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

        trial.report(val_loss, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

    # Secondary metrics: logged but not optimized
    mean_sigma = torch.mean(torch.stack([
        torch.log1p(torch.exp(p)).mean()
        for name, p in model.named_parameters() if 'rho' in name
    ])).item()

    all_preds = []
    all_targets = []
    for data, targets in val_loader:
        data, targets = data.to(device), targets.to(device)
        all_preds.append(mc_predict(model, data, config.training.mc_samples).mean(0))
        all_targets.append(targets)
    ece, _, _ = expected_calibration_error(
        torch.cat(all_preds),
        torch.cat(all_targets),
        num_classes=config.model.num_classes,
        num_bins=config.model.num_classes,
    )

    trial.set_user_attr('mean_sigma', mean_sigma)
    trial.set_user_attr('ece', ece)

    if writer is not None:
        writer.close()

    return val_loss


@click.command()
@click.option("--config", "config_path", type=str, default=None, help="Path to the configuration file.")
@click.option("--log-dir", type=str, default="./tunes", help="Path to the tuning directory.")
@click.option("--data-dir", type=str, default="./data", show_default=True,
              help="Datasets directory.")
@click.option("--study-name", type=str, default=None, help="Optuna study name.")
@click.option("--n-trials", type=int, default=30, show_default=True, help="Number of trials.")
@click.option("--n-startup-trials", type=int, default=5, show_default=True, help="Number of startup trials.")
@click.option("--n-warmup-steps", type=int, default=5, show_default=True, help="Number of warmup steps.")
@click.option("--epochs", type=int, default=30, show_default=True, help="Training epochs per trial.")
@click.option("--tensorboard", type=bool, is_flag=True, default=False,
              help="Use tensorboard for logging and visualization of training progress")
def main(
        config_path: str,
        log_dir: str,
        data_dir: str,
        study_name: str | None,
        n_trials: int,
        n_startup_trials: int,
        n_warmup_steps: int,
        epochs: int,
        tensorboard: bool = False,
) -> None:
    """Run Optuna hyperparameter search.
    """
    if study_name is None:
        raise click.BadParameter("Please provide a study name using --study-name.")

    config = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = Path(log_dir)
    Path.mkdir(run_dir / study_name, parents=True, exist_ok=True)

    study = optuna.create_study(
        study_name=study_name,
        storage=f"sqlite:///{run_dir / study_name / 'study.db'}",
        load_if_exists=True,
        direction='minimize',
        sampler=RandomSampler(seed=42),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=n_startup_trials,
            n_warmup_steps=n_warmup_steps,
        ),
    )
    study.optimize(
        functools.partial(objective,
                          config=config,
                          data_dir=data_dir,
                          device=device,
                          epochs=epochs,
                          writer_dir=run_dir / study_name if tensorboard else None),
        n_trials=n_trials,
    )

    click.echo(f"\nBest val_loss: {study.best_value:.6f}")
    click.echo(f"Best params: {study.best_params}")
    if 'log_prior_sigma1' in study.best_params:
        click.echo(f"  → prior_sigma1: {math.exp(study.best_params['log_prior_sigma1']):.4f}")
    if 'log_prior_sigma2' in study.best_params:
        click.echo(f"  → prior_sigma2: {math.exp(study.best_params['log_prior_sigma2']):.4f}")


if __name__ == '__main__':
    main()