import torch
from torchmetrics.classification import MulticlassCalibrationError
import matplotlib.pyplot as plt
from utils.uncertainty import mc_predict

@torch.no_grad()
def expected_calibration_error(model, test_loader, device, T=20, num_bins=10, num_classes=10):
    # Initialize torchmetric
    ece_metric = MulticlassCalibrationError(num_classes=num_classes, n_bins=num_bins, norm='l1').to(device)

    all_preds = []
    all_targets = []

    for data, targets in test_loader:
        data, targets = data.to(device), targets.to(device)
        # mc_predict returns (T, Batch, Classes) -> mean(0) gives (Batch, Classes)
        mc_preds = mc_predict(model, data, T).mean(0)

        all_preds.append(mc_preds)
        all_targets.append(targets)

    # Concatenate all batches
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)

    # Calculate ECE
    ece_value = ece_metric(all_preds, all_targets).item()

    # Optional: For the Reliability Diagram, we still need bin-wise stats
    # Torchmetrics doesn't expose bin-level acc/conf easily in the base class,
    # so we manually compute them for the plot using a simple mask
    confidences, predictions = all_preds.max(1)
    accuracies = (predictions == all_targets).float()

    bin_boundaries = torch.linspace(0, 1, num_bins + 1).to(device)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    bin_conf = []
    bin_acc = []

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.float().mean()
        if prop_in_bin.item() > 0:
            bin_acc.append(accuracies[in_bin].mean().item())
            bin_conf.append(confidences[in_bin].mean().item())

    print(f"\nExpected Calibration Error: {ece_value:.4f}")
    return ece_value, bin_conf, bin_acc

@torch.no_grad()
def reliability_diagram(model, loader, device, T=20, n_bins=10, num_classes=10):
    ece, bin_conf, bin_acc = expected_calibration_error(model, loader, device, T, n_bins, num_classes)

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
