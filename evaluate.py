import click
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import Config, get_dataset, load_config
from models.lenet import Net
from utils.data import get_dataloaders, get_datasets
from utils.checkpoint import load_checkpoint
from utils.uncertainty import mc_predict, quantify_uncertainties


def evaluate(model: nn.Module, test_loader: DataLoader, device: torch.device, *,
             mc_samples: int = 1) -> tuple[Tensor, tuple[Tensor, Tensor, Tensor]]:
    """
    Runs uncertainty over the *entire* test loader.
    """
    model.eval()

    all_preds: list[Tensor] = []
    all_uncertainties: list[list[Tensor]] = [[], [], []]

    for x, y in tqdm(test_loader, desc="Evaluating", leave=False):
        x = x.to(device)

        mc_preds = mc_predict(model, x, mc_samples=mc_samples)

        all_preds.append(mc_preds.mean(dim=0))
        for batch, uncertainty in zip(all_uncertainties, quantify_uncertainties(mc_preds)):
            batch.append(uncertainty)

    return torch.cat(all_preds), tuple(torch.cat(batch) for batch in all_uncertainties)

@click.command()
@click.option("--config", "config_path", type=str, default=None, help="YAML file describing experiment.")
@click.option("--checkpoint", "checkpoint_path", type=str, default=None, help="Checkpoint file to load.")
@click.option("--data-dir", type=str, default="./data", show_default=True,
              help="Datasets directory.")
@click.option("-T", "--mc-samples", type=int, default=128,
              help="Number of Monte Carlo samples for uncertainty evaluation.")
@click.option("--batch-size", type=int, default=128, help="Batch size for evaluation.")
@click.option("--seed", type=int, default=42, help="Random seed for reproducibility.")
def main(config_path: str, checkpoint_path: str, data_dir: str, mc_samples: int, batch_size: int, seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    config = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Prepare data
    train_dataset, test_dataset = get_datasets(
        root=data_dir,
        dataset=get_dataset(config.data.dataset),
        normalize=True,
        dataset_kwargs=config.data.kwargs,
    )

    _, _, test_loader = get_dataloaders(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        batch_size=batch_size,
        num_workers=config.data.num_workers,
        use_cuda=torch.cuda.is_available(),
        dataset_transform=config.data.dataset_transform
    )

    # Model
    model = Net(
        prior_sigma1=config.prior.sigma1,
        prior_sigma2=config.prior.sigma2,
        prior_pi=config.prior.pi,
        num_classes=config.model.num_classes,
    ).to(device)

    # Load weights
    load_checkpoint(checkpoint_path, model, device=device)

    # Standard evaluation
    preds, _ = evaluate(model, test_loader, device, mc_samples=mc_samples)
    labels = torch.cat([y for _, y in test_loader]).to(device)
    accuracy = (preds.argmax(dim=1) == labels).float().mean().item()
    click.echo(f"Test accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    main()