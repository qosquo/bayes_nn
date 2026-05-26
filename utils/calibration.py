import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader
from torchmetrics.classification import MulticlassCalibrationError, BinaryCalibrationError
import matplotlib.pyplot as plt
from utils.uncertainty import mc_predict


@torch.no_grad()
def mc_val_nll(model: nn.Module, val_loader: DataLoader, device: torch.device, n_samples: int = 10) -> float:
    """Predictive NLL via MC-averaging: -1/N Σ log(1/T Σ p(y|x,w_t))"""
    model.train()  # keep stochastic weight sampling
    total_nll = 0.0
    total_samples = 0

    for x, y in val_loader:
        x, y = x.to(device), y.to(device)
        log_probs = torch.stack([
            F.log_softmax(model(x), dim=1) for _ in range(n_samples)
        ])  # [n_samples, batch, classes]
        log_mixture = torch.logsumexp(log_probs, dim=0) - math.log(n_samples)
        total_nll += F.nll_loss(log_mixture, y, reduction='sum').item()
        total_samples += y.size(0)
    return total_nll / total_samples


def expected_calibration_error(
    preds: Tensor,
    targets: Tensor,
    num_bins: int = 10,
    num_classes: int = 10,
) -> tuple[float, list[float], list[float]]:
    if not isinstance(preds, Tensor) or not isinstance(targets, Tensor):
        raise TypeError("preds and targets must be pre-computed Tensors")

    device = preds.device
    ece_metric = MulticlassCalibrationError(num_classes=num_classes, n_bins=num_bins, norm='l1').to(device)
    ece_value = ece_metric(preds, targets).item()

    # Bin-wise stats for reliability diagram
    confidences, predictions = preds.max(1)
    accuracies = (predictions == targets).float()

    bin_boundaries = torch.linspace(0, 1, num_bins + 1).to(device)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    bin_conf: list[float] = []
    bin_acc: list[float] = []

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        if in_bin.float().mean().item() > 0:
            bin_acc.append(accuracies[in_bin].mean().item())
            bin_conf.append(confidences[in_bin].mean().item())

    print(f"\nExpected Calibration Error: {ece_value:.4f}")
    return ece_value, bin_conf, bin_acc


def static_calibration_error(
    preds: Tensor,
    targets: Tensor,
    n_bins: int = 15,
    num_classes: int = 10,
) -> float:
    """SCE: per-class binary calibration error averaged over all classes."""
    if not isinstance(preds, Tensor) or not isinstance(targets, Tensor):
        raise TypeError("preds and targets must be pre-computed Tensors")

    device = preds.device
    bce_metric = BinaryCalibrationError(n_bins=n_bins, norm='l1').to(device)
    sce = 0.0
    for k in range(num_classes):
        bce_metric.reset()
        pk = preds[:, k]
        yk = (targets == k).long()
        sce += bce_metric(pk, yk).item()

    sce /= num_classes
    print(f"\nStatic Calibration Error: {sce:.6f}")
    return sce


@torch.no_grad()
def reliability_diagram(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    mc_samples: int = 10,
    n_bins: int = 10,
    num_classes: int = 10,
) -> plt.Figure:
    all_preds: list[Tensor] = []
    all_targets: list[Tensor] = []
    for data, targets in loader:
        data, targets = data.to(device), targets.to(device)
        all_preds.append(mc_predict(model, data, mc_samples).mean(0))
        all_targets.append(targets)

    ece, bin_conf, bin_acc = expected_calibration_error(
        torch.cat(all_preds), torch.cat(all_targets), n_bins, num_classes,
    )

    fig = plt.figure(figsize=(5, 5))
    plt.plot([0, 1], [0, 1], "--", color="gray", label="Perfect Calibration")
    plt.bar(bin_conf, bin_acc, width=0.05, alpha=0.3, edgecolor="black", label="Model")
    plt.plot(bin_conf, bin_acc, marker="o", color="blue")

    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.title(f"Reliability Diagram (ECE={ece:.4f})")
    plt.legend()
    plt.grid(True)
    return fig