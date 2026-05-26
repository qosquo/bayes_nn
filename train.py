import click
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch import Tensor
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from datetime import datetime

from config import Config
from models.lenet import Net
from utils import compute_beta
from utils.calibration import reliability_diagram, mc_val_nll
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


def elbo_loss(output: Tensor, y: Tensor, kl: Tensor | float, beta: float) -> Tensor:
    return F.cross_entropy(output, y, reduction='sum') + beta * kl


def train(model: nn.Module, optimizer: optim.Optimizer, train_loader: DataLoader,
          device: torch.device, epoch: int, grad_clip: float | None = None, mc_samples: int = 1,
          beta_schedule: str = 'blundell', warmup_factor: float = 1.0,
          writer: SummaryWriter | None = None) -> float:
    model.train()
    total_loss = 0
    accuracy = 0

    loop = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)
    M = len(train_loader)

    for batch_idx, (x, y) in enumerate(loop):
        x, y = torch.tensor(x).to(device), torch.tensor(y).to(device)

        optimizer.zero_grad()

        beta = compute_beta(batch_idx, M, beta_schedule, warmup_factor)

        if mc_samples > 1:
            losses = []
            for _ in range(mc_samples):
                out = model(x)
                kl = model.kl_divergence()
                losses.append(elbo_loss(out, y, kl, beta))
            loss = torch.stack(losses).mean()
            output = out
        else:
            output = model(x)
            kl = model.kl_divergence()
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
            step = epoch * len(train_loader) + batch_idx
            writer.add_scalar("train/batch_loss", loss.item(), step)
            writer.add_scalar("train/batch_accuracy", batch_acc, step)
            writer.add_scalar("train/kl_divergence", kl.item(), step)

        # WandB
        if USE_WANDB:
            wandb.log({"train_batch_loss": loss.item()})

    if writer:
        writer.add_scalar("train/epoch_accuracy", accuracy / len(train_loader), epoch)

    return total_loss / len(train_loader)


def test(model: nn.Module, test_loader: DataLoader, device: torch.device, epoch: int,
         mc_samples: int = 1, beta_schedule: str = 'blundell', warmup_factor: float = 1.0,
         writer: SummaryWriter | None = None) -> tuple[float, float]:
    model.train()
    test_loss = 0
    correct = 0
    M = len(test_loader)

    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(test_loader):
            x, y = x.to(device), y.to(device)

            beta = compute_beta(batch_idx, M, beta_schedule, warmup_factor)

            if mc_samples > 1:
                outputs = torch.stack([model(x) for _ in range(mc_samples)])
                output = outputs.mean(0)
                kl = model.kl_divergence()
            else:
                output = model(x)
                kl = model.kl_divergence()

            loss = elbo_loss(output, y, kl, beta)
            test_loss += loss.item()

            pred = output.argmax(dim=1)
            correct += (pred == y).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = correct / len(test_loader.dataset)

    if writer:
        writer.add_scalar("test/loss", test_loss, epoch)
        writer.add_scalar("test/accuracy", accuracy, epoch)

    if USE_WANDB:
        wandb.log({"test_loss": test_loss, "test_accuracy": accuracy})

    return test_loss, accuracy


@click.command()
@click.option("--model-name", type=str, default=None, help="Model name for logging and checkpoints.")
@click.option("--batch-size", type=int, default=None, help="Training batch size.")
@click.option("--learning-rate", type=float, default=None, help="Learning rate.")
@click.option("--n-epochs", type=int, default=None, help="Number of training epochs.")
@click.option("--prior-sigma1", type=float, default=None, help="Prior sigma1.")
@click.option("--prior-sigma2", type=float, default=None, help="Prior sigma2.")
@click.option("--prior-pi", type=float, default=None, help="Prior mixture weight pi.")
@click.option("--checkpoint-epoch", type=int, default=0, show_default=True,
              help="Epoch to resume from.")
def main(model_name: str | None, batch_size: int | None, learning_rate: float | None,
         n_epochs: int | None, prior_sigma1: float | None, prior_sigma2: float | None,
         prior_pi: float | None, checkpoint_epoch: int) -> None:
    """Train a Bayesian neural network with ELBO loss."""
    config = Config()
    device = config.device

    # Override config with CLI options
    params = {
        'model_name': model_name, 'batch_size': batch_size,
        'learning_rate': learning_rate, 'n_epochs': n_epochs,
        'prior_sigma1': prior_sigma1, 'prior_sigma2': prior_sigma2,
        'prior_pi': prior_pi,
    }
    for key, value in params.items():
        if value is not None:
            setattr(config, key, value)

    # TensorBoard
    writer = SummaryWriter(log_dir="runs/{}/{}_{}".format(
        config.model_name,
        config.model_name,
        datetime.now().strftime("%Y%m%d-%H%M%S")
    ))

    # Data
    train_loader, val_loader, test_loader = get_dataloaders(
        data_dir="data",
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        use_cuda=torch.cuda.is_available(),
        dataset=config.dataset,
        dataset_kwargs={"split": "letters"},
    )

    # Model
    model = Net(
        prior_sigma1=config.prior_sigma1,
        prior_sigma2=config.prior_sigma2,
        prior_pi=config.prior_pi,
        num_classes=config.num_classes,
        rho_init=config.rho_init
    ).to(device)

    # Optimizer & scheduler
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, patience=6)

    # Resume if checkpoint exists
    date = datetime.now().strftime("%Y%m%d")
    start_epoch = load_checkpoint(model,
                                  optimizer,
                                  f'{config.checkpoint_path}/{config.get_checkpoint_name(checkpoint_epoch, date)}',
                                  device)

    # Training loop
    for epoch in range(start_epoch, config.n_epochs):
        warmup_factor = min(1.0, epoch / 20)  # warmup over 20 epochs
        train_loss = train(
            model=model,
            optimizer=optimizer,
            train_loader=train_loader,
            device=device,
            epoch=epoch,
            grad_clip=config.gradient_clip_norm,
            mc_samples=config.t_train,
            beta_schedule=config.beta_schedule,
            warmup_factor=warmup_factor,
            writer=writer
        )

        # Validation step
        val_loss, val_acc = test(
            model,
            val_loader,
            device,
            epoch,
            mc_samples=config.t_train,
            beta_schedule=config.beta_schedule,
            warmup_factor=warmup_factor,
            writer=writer
        )
        val_nll = mc_val_nll(model, val_loader, device, n_samples=config.mc_samples)
        scheduler.step(val_nll)
        click.echo(f"Validation: nll={val_nll:.6f}, loss={val_loss:.6f}, acc={val_acc * 100:.2f}%")

        if writer:
            writer.add_scalar("test/mc_nll", val_nll, epoch)
            writer.add_scalar("train/warmup_factor", warmup_factor, epoch)

        # Save checkpoint
        if config.save_model and epoch % config.save_interval == 0:
            save_checkpoint(
                model,
                optimizer,
                epoch,
                f"{config.checkpoint_dir}/{config.model_name}/{config.get_checkpoint_name(epoch, date)}"
            )
            if writer:
                writer.add_figure(
                    'model/reliability_diagram',
                    reliability_diagram(
                        model,
                        val_loader,
                        device,
                        mc_samples=config.t_train,
                        num_classes=config.num_classes,
                        n_bins=config.num_classes
                    ),
                    epoch
                )

    # Save final model
    if config.save_model:
        save_checkpoint(model, optimizer, config.n_epochs - 1, f"{config.checkpoint_dir}/{config.model_name}")

    writer.close()


if __name__ == "__main__":
    main()
