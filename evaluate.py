import torch
import torch.nn.functional as F

from config import Config
from models.lenet import Net
from utils.data import get_dataloaders
from utils.checkpoint import load_checkpoint
from utils.uncertainty import mc_predict, quantify_uncertainties


def evaluate(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    acc = 100.0 * correct / total
    print(f"Accuracy: {acc:.2f}%")
    return acc


@torch.no_grad()
def evaluate_with_uncertainty(model, test_loader, device, mc_samples):
    """
    Runs uncertainty over the *entire* test loader.
    """
    model.eval()

    all_preds = []
    all_aleatoric = []
    all_epistemic = []

    for x, y in test_loader:
        x = x.to(device)

        mc_preds = mc_predict(model, x, mc_samples=mc_samples)
        preds, uncertainties = quantify_uncertainties(mc_preds)

        # uncertainties:
        # [0] predictive
        # [1] aleatoric
        # [2] epistemic
        aleatoric = uncertainties[1].diagonal(dim1=1, dim2=2).sum(-1)
        epistemic = uncertainties[2].diagonal(dim1=1, dim2=2).sum(-1)

        all_preds.append(preds.cpu())
        all_aleatoric.append(aleatoric.cpu())
        all_epistemic.append(epistemic.cpu())

    all_preds = torch.cat(all_preds)
    all_aleatoric = torch.cat(all_aleatoric)
    all_epistemic = torch.cat(all_epistemic)
    all_total = all_aleatoric + all_epistemic

    print(f"Total Uncertainty (mean): {all_total.mean().item()}")
    print(f"Aleatoric (mean): {all_aleatoric.mean().item()}")
    print(f"Epistemic (mean): {all_epistemic.mean().item()}")

    return all_preds, (all_total, all_aleatoric, all_epistemic)


if __name__ == "__main__":
    config = Config()
    device = config.device

    # Prepare data
    _, _, test_loader = get_dataloaders(
        data_dir="data",
        batch_size=config.test_batch_size,
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

    # Optimizer (needed to load checkpoint)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    # Load weights
    load_checkpoint(model, optimizer, config.checkpoint_path, device)

    # Standard evaluation
    evaluate(model, test_loader, device)

    # Uncertainty
    evaluate_with_uncertainty(
        model,
        test_loader,
        device,
        mc_samples=config.mc_samples,
    )
