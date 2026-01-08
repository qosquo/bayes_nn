import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from datetime import datetime

from config import Config
# from models.lenet import Net
from models.mlp import Net
from utils.data import get_dataloaders
from utils.checkpoint import save_checkpoint, load_checkpoint

# Optional Weights & Biases
USE_WANDB = False
if USE_WANDB:
    import wandb
    wandb.init(project="bayesian-nn")


def elbo_loss(model, x, y, num_batches):
    output = model(x)
    nll = F.cross_entropy(output, y, reduction='sum')  # -log P(D|w)
    kl = model.kl_divergence()  # KL[q(w|θ) || P(w)]

    # Scale KL by minibatch weight (1/M from paper)
    return (kl / num_batches + nll) / x.size(0)


def train(model, optimizer, train_loader, device, epoch, log_interval, grad_clip=None, writer=None):
    model.train()
    total_loss = 0

    loop = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)

    for batch_idx, (x, y) in enumerate(loop):
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()

        loss = elbo_loss(model, x, y, len(train_loader))
        loss.backward()

        if grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())

        # TensorBoard logging
        if writer:
            writer.add_scalar("train/batch_loss", loss.item(), epoch * len(train_loader) + batch_idx)

        # WandB
        if USE_WANDB:
            wandb.log({"train_batch_loss": loss.item()})

    return total_loss / len(train_loader)


def test(model, test_loader, device, epoch, writer=None, T=1):
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)

            loss = elbo_loss(model, x, y, len(test_loader))
            test_loss += loss.item()

            from utils.uncertainty import mc_predict
            mc_preds = mc_predict(model, x, T).mean(0)

            pred = mc_preds.argmax(dim=1)
            correct += (pred == y).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = correct / len(test_loader.dataset)

    print(f"Test: loss={test_loss:.6f}, acc={accuracy * 100:.2f}%")

    if writer:
        writer.add_scalar("test/loss", test_loss, epoch)
        writer.add_scalar("test/accuracy", accuracy, epoch)

    if USE_WANDB:
        wandb.log({"test_loss": test_loss, "test_accuracy": accuracy})

    return test_loss, accuracy


def main():
    config = Config()
    device = config.device

    # TensorBoard
    writer = SummaryWriter(log_dir="runs/{}_{}".format(
        config.model_name,
        datetime.now().strftime("%Y%m%d-%H%M%S")
    ))

    # Data
    train_loader, test_loader = get_dataloaders(
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
    start_epoch = load_checkpoint(model, optimizer, config.checkpoint_path, device)

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

        test_loss, acc = test(model, test_loader, device, epoch, writer)

        scheduler.step()

        # Save checkpoint
        if config.save_model and epoch % config.save_interval == 0:
            save_checkpoint(
                model,
                optimizer,
                epoch,
                f"{config.checkpoint_dir}/{config.get_checkpoint_name(epoch)}"
            )

    # Save final model
    if config.save_model:
        save_checkpoint(model, optimizer, config.n_epochs - 1, config.checkpoint_path)

    writer.close()


if __name__ == "__main__":
    main()
