# tune.py
import functools
import math
from datetime import datetime
from typing import Any

import click

import optuna
import torch
from torch.utils.tensorboard import SummaryWriter

from config import Config
from train import train
from models.lenet import Net
from utils.data import get_dataloaders
from utils.calibration import expected_calibration_error, mc_val_nll
from utils.uncertainty import mc_predict

# Hyperparameter search space: all parameters are searched by default;
# use --fix to pin any of them to a specific value
SEARCH_SPACE: dict[str, dict[str, Any]] = {
    'log_prior_sigma1': {'type': 'float', 'low': -2.0, 'high': 0.0},
    'log_prior_sigma2': {'type': 'float', 'low': -8.0, 'high': -6.0},
    'prior_pi':         {'type': 'float', 'low': 0.2, 'high': 0.8},
    'rho_init':         {'type': 'float', 'low': -7.0, 'high': -3.0},
    'T':                {'type': 'categorical', 'choices': [1, 2, 5, 10]},
    'lr':               {'type': 'float', 'low': 1e-5, 'high': 1e-2, 'log': True},
    'beta_schedule':    {'type': 'categorical', 'choices': ['blundell', 'uniform', 'warmup']},
    'grad_clip':        {'type': 'categorical', 'choices': [None, 0.5, 1.0, 5.0]},
    'batch_size':       {'type': 'categorical', 'choices': [64, 128, 256]},
}


def _parse_value(name: str, raw: str) -> float | int | str | None:
    """Parse a --fix string value to the appropriate type based on SEARCH_SPACE."""
    if raw.lower() == 'none':
        return None
    spec = SEARCH_SPACE[name]
    if spec['type'] == 'categorical':
        sample = next(c for c in spec['choices'] if c is not None)
        if isinstance(sample, int):
            return int(raw)
        elif isinstance(sample, float):
            return float(raw)
        return raw
    return float(raw)


def _parse_fix_options(fix_values: tuple[str, ...]) -> dict[str, Any]:
    """Parse a list of --fix key=value pairs into a dict of fixed parameters."""
    fixed: dict[str, Any] = {}
    for item in fix_values:
        if '=' not in item:
            raise click.BadParameter(f"Expected key=value, got '{item}'")
        key, raw = item.split('=', 1)
        if key not in SEARCH_SPACE:
            raise click.BadParameter(
                f"Unknown parameter '{key}'. Available: {', '.join(SEARCH_SPACE)}"
            )
        fixed[key] = _parse_value(key, raw)
    return fixed


def _suggest_or_fix(
    trial: optuna.trial.Trial,
    name: str,
    fixed: dict[str, Any],
) -> Any:
    """Return a fixed value or suggest from the search space."""
    if name in fixed:
        return fixed[name]
    spec = SEARCH_SPACE[name]
    if spec['type'] == 'float':
        return trial.suggest_float(name, spec['low'], spec['high'], log=spec.get('log', False))
    return trial.suggest_categorical(name, spec['choices'])


def objective(
    trial: optuna.trial.Trial,
    study_name: str,
    fixed: dict[str, Any],
    epochs: int,
) -> float:
    config = Config()
    device = config.device

    # Hyperparameters: searched or fixed via --fix
    log_sigma1 = _suggest_or_fix(trial, 'log_prior_sigma1', fixed)
    log_sigma2 = _suggest_or_fix(trial, 'log_prior_sigma2', fixed)
    sigma1 = math.exp(log_sigma1)
    sigma2 = math.exp(log_sigma2)
    pi = _suggest_or_fix(trial, 'prior_pi', fixed)
    rho_init = _suggest_or_fix(trial, 'rho_init', fixed)
    t_train = _suggest_or_fix(trial, 'T', fixed)
    lr = _suggest_or_fix(trial, 'lr', fixed)
    beta_schedule = _suggest_or_fix(trial, 'beta_schedule', fixed)
    grad_clip = _suggest_or_fix(trial, 'grad_clip', fixed)
    batch_size = _suggest_or_fix(trial, 'batch_size', fixed)

    config.batch_size = batch_size

    writer = SummaryWriter(log_dir="tunes/{}/{}_{}".format(
        study_name if study_name else 'optuna_study',
        f"{study_name if study_name else 'optuna_study'}_trial{trial.number}",
        datetime.now().strftime("%Y%m%d-%H%M%S")
    ))

    model = Net(
        prior_sigma1=sigma1,
        prior_sigma2=sigma2,
        prior_pi=pi,
        num_classes=config.num_classes,
        rho_init=rho_init,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_loader, val_loader, _ = get_dataloaders(
        data_dir="data",
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        use_cuda=torch.cuda.is_available(),
        dataset=config.dataset,
        dataset_kwargs={"split": "letters"},
    )

    for epoch in range(epochs):
        warmup_factor = min(1.0, 2.0 * epoch / epochs) if beta_schedule == 'warmup' else 1.0
        train(
            model, optimizer, train_loader, device, epoch,
            grad_clip=grad_clip, mc_samples=t_train,
            beta_schedule=beta_schedule, warmup_factor=warmup_factor,
            writer=writer,
        )

        if epoch % 3 == 0 or epoch == epochs - 1:
            # Interim NLL (T=5) for pruning decisions
            interim_nll = mc_val_nll(model, val_loader, device, n_samples=5)
            trial.report(interim_nll, epoch)
            if trial.should_prune():
                writer.close()
                raise optuna.TrialPruned()

    # Final evaluation: full MC NLL
    val_nll = mc_val_nll(model, val_loader, device, n_samples=10)

    # Secondary metrics: logged but not optimized
    mean_sigma = torch.mean(torch.stack([
        torch.log1p(torch.exp(p)).mean()
        for name, p in model.named_parameters() if 'rho' in name
    ])).item()
    all_preds = []
    all_targets = []
    for data, targets in val_loader:
        data, targets = data.to(device), targets.to(device)
        all_preds.append(mc_predict(model, data, t_train).mean(0))
        all_targets.append(targets)
    ece, _, _ = expected_calibration_error(
        torch.cat(all_preds), torch.cat(all_targets), num_classes=config.num_classes, num_bins=26,
    )

    trial.set_user_attr('mean_sigma', mean_sigma)
    trial.set_user_attr('ece', ece)

    writer.close()
    return val_nll


@click.command()
@click.option("--study-name", type=str, default=None, help="Optuna study name.")
@click.option("--storage", type=str, default=None, help="SQLite storage path.")
@click.option("--n-trials", type=int, default=30, show_default=True, help="Number of trials.")
@click.option("--epochs", type=int, default=30, show_default=True, help="Training epochs per trial.")
@click.option(
    "--fix", "fix_values", type=str, multiple=True,
    help="Fix a hyperparameter: --fix key=value (repeatable). "
         f"Available: {', '.join(SEARCH_SPACE)}.",
)
def main(
    study_name: str | None,
    storage: str | None,
    n_trials: int,
    epochs: int,
    fix_values: tuple[str, ...],
) -> None:
    """Run Optuna hyperparameter search.

    By default all hyperparameters are searched. Use --fix to pin specific ones.

    \b
    Examples:
        python tune.py --fix lr=0.001 --fix T=5
        python tune.py --fix grad_clip=none --epochs 50
        python tune.py --n-trials 50 --storage study.db
    """
    fixed = _parse_fix_options(fix_values)
    if fixed:
        click.echo(f"Fixed parameters: {fixed}")

    study = optuna.create_study(
        study_name=study_name,
        storage=f'sqlite://{storage}' if storage else None,
        load_if_exists=True,
        direction='minimize',
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=3,
        ),
    )
    study.optimize(
        functools.partial(objective, study_name=study_name, fixed=fixed, epochs=epochs),
        n_trials=n_trials,
    )

    click.echo(f"\nBest val_nll: {study.best_value:.6f}")
    click.echo(f"Best params: {study.best_params}")
    if 'log_prior_sigma1' in study.best_params:
        click.echo(f"  → prior_sigma1: {math.exp(study.best_params['log_prior_sigma1']):.4f}")
    if 'log_prior_sigma2' in study.best_params:
        click.echo(f"  → prior_sigma2: {math.exp(study.best_params['log_prior_sigma2']):.4f}")


if __name__ == '__main__':
    main()