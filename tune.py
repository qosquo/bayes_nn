# tune.py
import argparse
import functools
import math
from datetime import datetime

import optuna
from torch.utils.tensorboard import SummaryWriter

from config import Config
from train import train, test
from models.lenet import Net
from utils.data import get_dataloaders
import torch

PATIENCE = 10


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--study_name', type=str, default=None)
    parser.add_argument('--storage', type=str, default=None)
    parser.add_argument('--n_trials', type=int, default=30)
    return parser.parse_args()


def objective(trial: optuna.trial.Trial, study_name: str) -> float:
    config = Config()
    device = config.device

    # Suggest hyperparameters
    lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    log_sigma1 = trial.suggest_float('log_prior_sigma1', -2, 0)
    log_sigma2 = trial.suggest_float('log_prior_sigma2', -8, -6)
    sigma1 = math.exp(log_sigma1)
    sigma2 = math.exp(log_sigma2)
    pi = trial.suggest_float('prior_pi', 0.2, 0.8)
    t_train = trial.suggest_categorical('T', [1, 2, 5, 10])
    rho_init = trial.suggest_float('rho_init', -7, -3)
    beta_schedule = trial.suggest_categorical('beta_schedule', ['blundell', 'uniform', 'warmup'])
    grad_clip = trial.suggest_categorical('grad_clip', [None, 0.5, 1.0, 5.0])
    num_batches = trial.suggest_categorical('num_batches', [64, 128, 256])

    config.batch_size = num_batches
    config.n_epochs = 50
    best_val_loss = float('inf')
    no_improve = 0

    writer = SummaryWriter(log_dir="tunes/{}_{}".format(
        f"{study_name if study_name else 'optuna_study'}_trial{trial.number}",
        datetime.now().strftime("%Y%m%d-%H%M%S")
    ))

    # Create model
    model = Net(
        prior_sigma1=sigma1,
        prior_sigma2=sigma2,
        prior_pi=pi,
        num_classes=10,
        rho_init=rho_init,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Get data
    train_loader, val_loader, test_loader = get_dataloaders(
        data_dir="data",
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        use_cuda=torch.cuda.is_available(),
    )

    for epoch in range(config.n_epochs):
        train(
            model, optimizer, train_loader, device, epoch,
            grad_clip=grad_clip, T=t_train,
            beta_schedule=beta_schedule, n_epochs=config.n_epochs,
            writer=writer,
        )
        val_loss, _ = test(model, val_loader, device, epoch, T=config.mc_samples, writer=writer)

        trial.report(val_loss, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                break

    writer.close()
    return best_val_loss


if __name__ == '__main__':
    args = parse_args()
    study = optuna.create_study(
        study_name=args.study_name,
        storage=f'sqlite://{args.storage}' if args.storage else None,
        load_if_exists=True,
        direction='minimize',
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=10),
    )
    study.optimize(
        functools.partial(objective, study_name=args.study_name),
        n_trials=args.n_trials,
    )

    print(f"Best params: {study.best_params}")
    print(f"Best prior sigma1: {math.exp(study.best_params['log_prior_sigma1']):.4f}")
    print(f"Best prior sigma2: {math.exp(study.best_params['log_prior_sigma2']):.4f}")
    print(f"Best val_loss: {study.best_value:.6f}")