import torch
import torch.nn.functional as F

from config import Config
from models.lenet import Net
from utils.data import get_dataloaders
from utils.checkpoint import load_checkpoint
from utils.uncertainty import quantify_uncertainties


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


def evaluate_with_uncertainty(model, test_loader, device, mc_samples):
    """
    Runs uncertainty on the *first* batch only (обычно достаточно).
    """
    model.eval()

    x, y = next(iter(test_loader))
    x, y = x.to(device), y.to(device)

    preds, uncertainties = quantify_uncertainties(model, x, T=mc_samples)

    print("Predictions:", preds.tolist())
    aleatoric = uncertainties[1].diagonal(dim1=1, dim2=2).sum(-1).mean().item()
    epistemic = uncertainties[2].diagonal(dim1=1, dim2=2).sum(-1).mean().item()
    print(f"Aleatoric: {aleatoric}")
    print(f"Epistemic: {epistemic}")

    return preds, uncertainties


if __name__ == "__main__":
    config = Config()
    device = config.device

    # Prepare data
    _, test_loader = get_dataloaders(
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
