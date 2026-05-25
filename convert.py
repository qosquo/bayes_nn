"""
CLI tool for converting deterministic CNNs to Bayesian NNs via MOPED.

Usage:
    python convert.py convert   --arch model.py --class-name MyNet --weights pretrained.pth
    python convert.py finetune  --arch model.py --class-name MyNet --weights bayesian.pth --folder ./data
    python convert.py evaluate  --arch model.py --class-name MyNet --weights finetuned.pth --folder ./data
"""

import json

import click
import torch
import torch.optim as optim

from converter import convert_to_bayesian, BayesianModelWrapper
from models.bayesian_layers import BayesianLinear, BayesianConv2d
from train import elbo_loss, train, test
from evaluate import evaluate_with_uncertainty
from utils import import_attr, load_state_dict
from utils.data import get_dataloaders_from_folder
from utils.calibration import expected_calibration_error, static_calibration_error


def _resolve_device(device_str: str) -> torch.device:
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def _load_model(arch: str, class_name: str, weights: str, device: torch.device,
                model_args: str | None = None):
    """Load user model, optionally wrap in BayesianModelWrapper, load weights."""
    model_cls = import_attr(arch, class_name)

    extra_kwargs = {}
    if model_args is not None:
        try:
            extra_kwargs = json.loads(model_args)
        except json.JSONDecodeError as e:
            raise click.ClickException(f"Invalid --model-args JSON: {e}")

    model = model_cls(**extra_kwargs)

    is_already_bayesian = any(isinstance(m, (BayesianLinear, BayesianConv2d))
                              for m in model.modules())

    if is_already_bayesian:
        model = BayesianModelWrapper(model)

    state_dict = load_state_dict(weights)
    model.load_state_dict(state_dict)
    model.to(device)
    click.echo(f"Loaded model from {weights}")

    return model, is_already_bayesian


# ── CLI group ──

@click.group()
def cli():
    """MOPED: Convert deterministic CNN to Bayesian NN."""


# ── convert ──

@cli.command()
@click.option("--arch", "-a", required=True, type=click.Path(exists=True),
              help="Python file containing the model class.")
@click.option("--class-name", "-c", required=True,
              help="Name of the model class in the arch file.")
@click.option("--weights", "-w", required=True, type=click.Path(exists=True),
              help="Path to pretrained weights (.pth).")
@click.option("--delta", "-d", type=float, default=0.1, show_default=True,
              help="MOPED scale factor.")
@click.option("--output", "-o", default="bayesian_model.pth", show_default=True,
              help="Output path.")
@click.option("--device", default="auto", show_default=True,
              help="Device: cpu, cuda, or auto.")
@click.option("--model-args", default=None,
              help='JSON dict of extra constructor kwargs, e.g. \'{"num_classes": 26}\'')
def convert(arch, class_name, weights, delta, output, device, model_args):
    """Convert a deterministic model to Bayesian using MOPED initialization."""
    dev = _resolve_device(device)

    model_cls = import_attr(arch, class_name)

    extra_kwargs = {}
    if model_args is not None:
        extra_kwargs = json.loads(model_args)

    model = model_cls(**extra_kwargs)

    state_dict = load_state_dict(weights)
    model.load_state_dict(state_dict)
    click.echo(f"Loaded deterministic model from {weights}")

    bayesian_model = convert_to_bayesian(model, delta=delta)
    bayesian_model = bayesian_model.to(dev)

    n_bayesian = sum(1 for m in bayesian_model.modules()
                     if isinstance(m, (BayesianLinear, BayesianConv2d)))
    click.echo(f"Converted {n_bayesian} layers to Bayesian (delta={delta})")

    torch.save({"model_state": bayesian_model.state_dict()}, output)
    click.echo(f"Saved Bayesian model to {output}")


# ── finetune ──

@cli.command()
@click.option("--arch", "-a", required=True, type=click.Path(exists=True),
              help="Python file containing the model class.")
@click.option("--class-name", "-c", required=True,
              help="Name of the model class in the arch file.")
@click.option("--weights", "-w", required=True, type=click.Path(exists=True),
              help="Path to Bayesian model weights (.pth).")
@click.option("--folder", "-f", required=True,
              type=click.Path(exists=True, file_okay=False),
              help="Data root (with train/ and test/ subdirs).")
@click.option("--transforms", "-t", "transform_file", default=None,
              type=click.Path(exists=True, dir_okay=False),
              help="Python file defining a `transform` variable.")
@click.option("--epochs", "-e", type=int, default=10, show_default=True)
@click.option("--lr", type=float, default=1e-4, show_default=True)
@click.option("--batch-size", "-b", type=int, default=64, show_default=True)
@click.option("--beta-schedule", type=click.Choice(["warmup", "uniform", "blundell"]),
              default="warmup", show_default=True)
@click.option("--grad-clip", type=float, default=1.0, show_default=True)
@click.option("--output", "-o", default="finetuned_model.pth", show_default=True)
@click.option("--device", default="auto", show_default=True)
@click.option("--model-args", default=None,
              help='JSON dict of extra constructor kwargs.')
def finetune(arch, class_name, weights, folder, transform_file,
             epochs, lr, batch_size, beta_schedule, grad_clip, output, device, model_args):
    """Finetune a Bayesian model with ELBO loss."""
    dev = _resolve_device(device)

    model, is_already_bayesian = _load_model(arch, class_name, weights, dev, model_args)
    if not is_already_bayesian:
        model = convert_to_bayesian(model.model if isinstance(model, BayesianModelWrapper) else model,
                                    delta=0.1)
        model = model.to(dev)

    train_loader, val_loader, _ = get_dataloaders_from_folder(
        data_dir=folder,
        batch_size=batch_size,
        use_cuda=dev.type == "cuda",
        transform_file=transform_file,
    )

    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        warmup_factor = min(1.0, epoch / 20)
        train_loss = train(
            model=model,
            optimizer=optimizer,
            train_loader=train_loader,
            device=dev,
            epoch=epoch,
            grad_clip=grad_clip,
            beta_schedule=beta_schedule,
            warmup_factor=warmup_factor,
        )

        val_loss, val_acc = test(
            model, val_loader, dev, epoch,
            beta_schedule=beta_schedule,
            warmup_factor=warmup_factor,
        )
        click.echo(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.6f}, val_acc={val_acc * 100:.2f}%")

    torch.save({"model_state": model.state_dict()}, output)
    click.echo(f"Saved finetuned model to {output}")


# ── evaluate ──

@cli.command()
@click.option("--arch", "-a", required=True, type=click.Path(exists=True),
              help="Python file containing the model class.")
@click.option("--class-name", "-c", required=True,
              help="Name of the model class in the arch file.")
@click.option("--weights", "-w", required=True, type=click.Path(exists=True),
              help="Path to model weights (.pth).")
@click.option("--folder", "-f", required=True,
              type=click.Path(exists=True, file_okay=False),
              help="Data root (with train/ and test/ subdirs).")
@click.option("--transforms", "-t", "transform_file", default=None,
              type=click.Path(exists=True, dir_okay=False),
              help="Python file defining a `transform` variable.")
@click.option("--mc-samples", "-T", type=int, default=20, show_default=True)
@click.option("--batch-size", "-b", type=int, default=64, show_default=True)
@click.option("--output", "-o", default=None,
              help="Output JSON path (optional).")
@click.option("--device", default="auto", show_default=True)
@click.option("--model-args", default=None,
              help='JSON dict of extra constructor kwargs.')
def evaluate(arch, class_name, weights, folder, transform_file,
             mc_samples, batch_size, output, device, model_args):
    """Evaluate uncertainty of a Bayesian model on test data."""
    dev = _resolve_device(device)

    model, is_already_bayesian = _load_model(arch, class_name, weights, dev, model_args)
    if not is_already_bayesian:
        model = convert_to_bayesian(model.model if isinstance(model, BayesianModelWrapper) else model,
                                    delta=0.1)
        model = model.to(dev)

    _, _, test_loader = get_dataloaders_from_folder(
        data_dir=folder,
        batch_size=batch_size,
        use_cuda=dev.type == "cuda",
        transform_file=transform_file,
    )

    # Determine num_classes
    x_sample, _ = next(iter(test_loader))
    with torch.no_grad():
        model.train()
        out_sample = model(x_sample.to(dev))
    num_classes = out_sample.shape[1]

    # Standard accuracy
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(dev), y.to(dev)
            pred = model(x).argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    accuracy = 100.0 * correct / total

    # ECE & SCE (reuse existing functions)
    click.echo(f"Running MC evaluation (T={mc_samples})...")
    ece, _, _ = expected_calibration_error(
        model, test_loader, dev, T=mc_samples, num_classes=num_classes,
    )
    sce = static_calibration_error(
        model, test_loader, dev, T=mc_samples, num_classes=num_classes,
    )

    # Uncertainty (reuse evaluate.py)
    all_preds, (total_unc, all_aleatoric, all_epistemic) = evaluate_with_uncertainty(
        model, test_loader, dev, mc_samples=mc_samples,
    )

    results = {
        "accuracy": round(accuracy, 4),
        "ece": round(ece, 6),
        "sce": round(sce, 6),
        "mc_samples": mc_samples,
        "num_classes": num_classes,
        "num_test_samples": total,
        "uncertainty": {
            "aleatoric_mean": round(all_aleatoric.mean().item(), 6),
            "aleatoric_std": round(all_aleatoric.std().item(), 6),
            "epistemic_mean": round(all_epistemic.mean().item(), 6),
            "epistemic_std": round(all_epistemic.std().item(), 6),
        },
    }

    if output:
        with open(output, "w") as f:
            json.dump(results, f, indent=2)
        click.echo(f"Results saved to {output}")

    click.echo(json.dumps(results, indent=2))


def main():
    cli()


if __name__ == "__main__":
    main()