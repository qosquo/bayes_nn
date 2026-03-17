# tune.py
import argparse
import functools
import math
from datetime import datetime

import optuna
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from config import Config
from train import train
from models.lenet import Net
from utils.data import get_dataloaders
from utils.calibration import expected_calibration_error

FIXED_EPOCHS = 30


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--study_name', type=str, default=None)
    parser.add_argument('--storage', type=str, default=None)
    parser.add_argument('--n_trials', type=int, default=30)
    return parser.parse_args()


def mc_val_nll(model, val_loader, device, n_samples=10):
    """Predictive NLL via MC-averaging: -1/N Σ log(1/T Σ p(y|x,w_t))"""
    model.train()  # keep stochastic weight sampling
    total_nll = 0.0
    total_samples = 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device) - 1
            log_probs = torch.stack([
                F.log_softmax(model(x), dim=1) for _ in range(n_samples)
            ])  # [n_samples, batch, classes]
            log_mixture = torch.logsumexp(log_probs, dim=0) - math.log(n_samples)
            total_nll += F.nll_loss(log_mixture, y, reduction='sum').item()
            total_samples += y.size(0)
    return total_nll / total_samples


def objective(trial: optuna.trial.Trial, study_name: str) -> float:
    config = Config()
    device = config.device

    # Suggest hyperparameters
    # Part 1. These are the ones that will be used for pruning decisions, so they should be cheap to evaluate (e.g., interim NLL after a few epochs)
    log_sigma1 = trial.suggest_float('log_prior_sigma1', -2, 0)
    log_sigma2 = trial.suggest_float('log_prior_sigma2', -8, -6)
    sigma1 = math.exp(log_sigma1)
    sigma2 = math.exp(log_sigma2)
    pi = trial.suggest_float('prior_pi', 0.2, 0.8)
    rho_init = trial.suggest_float('rho_init', -7, -3)
    # Part 2. Will be ignored for pruning decisions to keep them cheap, but can be used for final evaluation of promising trials
    t_train = trial.suggest_categorical('T', [1, 2, 5, 10]) if False else 1
    lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True) if False else config.learning_rate
    beta_schedule = trial.suggest_categorical('beta_schedule', ['blundell', 'uniform', 'warmup']) if False else 'warmup'
    grad_clip = trial.suggest_categorical('grad_clip', [None, 0.5, 1.0, 5.0]) if False else 1.0
    num_batches = trial.suggest_categorical('num_batches', [64, 128, 256]) if False else config.batch_size

    config.batch_size = num_batches

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

    for epoch in range(FIXED_EPOCHS):
        warmup_factor = min(1.0, 2.0 * epoch / FIXED_EPOCHS) if beta_schedule == 'warmup' else 1.0
        train(
            model, optimizer, train_loader, device, epoch,
            grad_clip=grad_clip, T=t_train,
            beta_schedule=beta_schedule, warmup_factor=warmup_factor,
            writer=writer,
        )

        if epoch % 3 == 0 or epoch == FIXED_EPOCHS - 1:
            # Cheap interim NLL (T=5) for pruning decisions
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
    ece, _, _ = expected_calibration_error(model, val_loader, device, T=t_train, num_classes=config.num_classes, num_bins=26)

    trial.set_user_attr('mean_sigma', mean_sigma)
    trial.set_user_attr('ece', ece)

    writer.close()
    return val_nll


if __name__ == '__main__':
    args = parse_args()
    study = optuna.create_study(
        study_name=args.study_name,
        storage=f'sqlite://{args.storage}' if args.storage else None,
        load_if_exists=True,
        direction='minimize',
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5,   # первые 5 trial'ов — полностью
            n_warmup_steps=3,     # pruning после 3-й эпохи
        ),
    )
    study.optimize(
        functools.partial(objective, study_name=args.study_name),
        n_trials=args.n_trials,
    )

    print(f"Best params: {study.best_params}")
    print(f"Best prior sigma1: {math.exp(study.best_params['log_prior_sigma1']):.4f}")
    print(f"Best prior sigma2: {math.exp(study.best_params['log_prior_sigma2']):.4f}")
    print(f"Best val_nll: {study.best_value:.6f}")