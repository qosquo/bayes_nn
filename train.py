import argparse

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from datetime import datetime

from config import Config
from models.lenet import Net
# from models.mlp import Net
from utils.calibration import reliability_diagram
from utils.data import get_dataloaders
from utils.checkpoint import save_checkpoint, load_checkpoint

# Optional Weights & Biases
USE_WANDB = False
if USE_WANDB:
    try:
        import wandb
        wandb.init(project="bayesian-nn")
    except ImportError:
        print("wandb module not found. Please install it if you want to use Weights & Biases.")
        USE_WANDB = False


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--learning_rate', type=float, default=None)
    parser.add_argument('--n_epochs', type=int, default=None)
    parser.add_argument('--prior_sigma1', type=float, default=None)
    parser.add_argument('--prior_sigma2', type=float, default=None)
    parser.add_argument('--prior_pi', type=float, default=None)
    parser.add_argument('--checkpoint_epoch', type=int, default=0)
    return parser.parse_args()


def elbo_loss(model, x, y, num_batches, beta):
    output = model(x)
    nll = num_batches * F.cross_entropy(output, y, reduction='mean')  # -log P(D|w)
    kl = model.kl_divergence()  # KL[q(w|θ) || P(w)]

    return nll + beta * kl, nll, kl


def train(model, optimizer, train_loader, device, epoch, log_interval, grad_clip=None, writer=None):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    loop = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)

    for batch_idx, (x, y) in enumerate(loop):
        x, y = torch.tensor(x).to(device), torch.tensor(y).to(device)

        optimizer.zero_grad()

        # beta = 1 / len(train_loader)
        beta = (2 ** (len(train_loader) - batch_idx - 1)) / (2 ** (len(train_loader)) - 1)
        loss, nll, kl = elbo_loss(model, x, y, len(train_loader), beta)
        loss.backward()

        if grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        # Accuracy calculation
        with torch.no_grad():
            output = model(x)
            pred = output.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)

        batch_acc = (pred == y).float().mean().item()
        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())

        # TensorBoard logging
        if writer:
            step = epoch * len(train_loader) + batch_idx
            writer.add_scalar("train/batch_loss", loss.item(), step)
            writer.add_scalar("train/batch_accuracy", batch_acc, step)
            writer.add_scalar("train/nll", nll.item(), step)
            writer.add_scalar("train/kl_divergence", kl.item(), step)

        # Epoch accuracy
        epoch_acc = correct / total
        if writer:
            writer.add_scalar("train/epoch_accuracy", epoch_acc, epoch)

        # WandB
        if USE_WANDB:
            wandb.log({"train_batch_loss": loss.item()})

    return total_loss / len(train_loader)


def test(model, test_loader, device, epoch, writer=None, T=1):
    model.eval()
    test_loss = 0
    test_nll = 0
    correct = 0

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)

            beta = 1 / len(test_loader)
            loss, nll, _ = elbo_loss(model, x, y, len(test_loader), beta)
            test_loss += loss.item()
            test_nll += nll.item()

            from utils.uncertainty import mc_predict
            mc_preds = mc_predict(model, x, T).mean(0)

            pred = mc_preds.argmax(dim=1)
            correct += (pred == y).sum().item()

    test_loss /= len(test_loader.dataset)
    test_nll /= len(test_loader.dataset)
    accuracy = correct / len(test_loader.dataset)

    print(f"Test: loss={test_loss:.6f}, nll={test_nll:.6f}, acc={accuracy * 100:.2f}%")

    if writer:
        writer.add_scalar("test/loss", test_loss, epoch)
        writer.add_scalar("test/nll", test_nll, epoch)
        writer.add_scalar("test/accuracy", accuracy, epoch)

    if USE_WANDB:
        wandb.log({"test_loss": test_loss, "test_nll": test_nll, "test_accuracy": accuracy})

    return test_loss, accuracy


def main():
    config = Config()
    device = config.device
    args = parse_args()

    for key, value in vars(args).items():
        if value is not None:
            setattr(config, key, value)

    # TensorBoard
    writer = SummaryWriter(log_dir="runs/{}_{}".format(
        config.model_name,
        datetime.now().strftime("%Y%m%d-%H%M%S")
    ))

    # Data
    train_loader, val_loader, test_loader = get_dataloaders(
        data_dir="data",
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        use_cuda=torch.cuda.is_available(),
    )

    # Model
    model = Net(
        prior_sigma1=config.prior_sigma1,
        prior_sigma2=config.prior_sigma2,
        prior_pi=config.prior_pi,
        num_classes=config.num_classes,
    ).to(device)

    # Optimizer & scheduler
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = StepLR(optimizer, step_size=config.scheduler_step_size, gamma=config.gamma)

    # Resume if checkpoint exists
    date = datetime.now().strftime("%Y%m%d")
    start_epoch = load_checkpoint(model,
                                  optimizer,
                                  f'{config.checkpoint_path}/{config.get_checkpoint_name(args.checkpoint_epoch, date)}',
                                  device)

    # Training loop
    for epoch in range(start_epoch, config.n_epochs):
        train_loss = train(
            model=model,
            optimizer=optimizer,
            train_loader=train_loader,
            device=device,
            epoch=epoch,
            log_interval=config.log_interval,
            grad_clip=config.gradient_clip_norm,
            writer=writer
        )

        # Validation step
        val_loss, val_acc = test(model, val_loader, device, epoch, writer, T=config.mc_samples)
        print(f"Validation: loss={val_loss:.6f}, acc={val_acc * 100:.2f}%")

        scheduler.step()

        # Save checkpoint
        if config.save_model and epoch % config.save_interval == 0:
            save_checkpoint(
                model,
                optimizer,
                epoch,
                f"{config.checkpoint_dir}/{config.get_checkpoint_name(epoch, date)}"
            )
            if writer:
                writer.add_figure('model/reliability_diagram', reliability_diagram(model, val_loader, device), epoch)

    # Save final model
    if config.save_model:
        save_checkpoint(model, optimizer, config.n_epochs - 1, config.checkpoint_path)

    writer.close()


if __name__ == "__main__":
    main()
