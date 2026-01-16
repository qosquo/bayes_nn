# tune.py
import argparse
import math

import optuna

from config import Config
from train import train, test
from models.mlp import Net
from utils.data import get_dataloaders
import torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--study_name', type=str, default=None)
    parser.add_argument('--storage', type=str, default=None)
    parser.add_argument('--n_trials', type=int, default=30)
    return parser.parse_args()


def objective(trial: optuna.trial.Trial) -> float:
    config = Config()
    device = config.device

    # Suggest hyperparameters
    lr = trial.suggest_float('lr', 1e-5, 1e-2)
    log_sigma1 = trial.suggest_float('log_prior_sigma1', -2, 0)
    log_sigma2 = trial.suggest_float('log_prior_sigma2', -8, -6)
    sigma1 = math.exp(log_sigma1)
    sigma2 = math.exp(log_sigma2)
    pi = trial.suggest_float('prior_pi', 0.2, 0.8)
    T = trial.suggest_categorical('T', [1, 2, 5, 10])

    # Create model
    model = Net(prior_sigma1=sigma1, prior_sigma2=sigma2,
                prior_pi=pi, num_classes=10).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Get data
    train_loader, val_loader, test_loader = get_dataloaders(
        data_dir="data",
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        use_cuda=torch.cuda.is_available(),
    )

    # Train for N epochs
    best_val_acc = 0
    patience = 5
    no_improve = 0

    for epoch in range(config.n_epochs):
        train(model, optimizer, train_loader, device, epoch, config.log_interval)
        val_loss, val_acc = test(model, val_loader, device, epoch, T=T)

        # Report intermediate value for pruning
        trial.report(val_acc, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    return best_val_acc


if __name__ == '__main__':
    args = parse_args()
    study = optuna.create_study(
        study_name=args.study_name,
        storage=f'sqlite://{args.storage}' if args.storage else None,
        load_if_exists=True,
        direction='maximize'
    )
    study.optimize(objective, n_trials=args.n_trials)

    print(f"Best params: {study.best_params}")
    print(f"Best val_acc: {study.best_value}")