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
from train import train, test
from utils import import_attr, load_state_dict
from utils.data import get_dataloaders_from_folder
from utils.calibration import expected_calibration_error, static_calibration_error
from utils.uncertainty import mc_predict, quantify_uncertainties


def _resolve_device(device_str: str) -> torch.device:
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def _load_model(
    arch: str,
    class_name: str,
    weights: str,
    device: torch.device,
    model_args: str | None = None,
    ensure_bayesian: bool = False,
    delta: float = 0.1,
) -> BayesianModelWrapper | torch.nn.Module:
    """Load user model, load weights, optionally convert to Bayesian."""
    model_cls = import_attr(arch, class_name)

    extra_kwargs: dict = {}
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

    if ensure_bayesian and not is_already_bayesian:
        model = convert_to_bayesian(model, delta=delta)
        model.to(device)

    return model


@click.group()
def cli():
    """MOPED: Convert deterministic CNN to Bayesian NN."""


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
def convert(arch: str, class_name: str, weights: str, delta: float,
            output: str, device: str, model_args: str | None) -> None:
    """Convert a deterministic model to Bayesian using MOPED initialization."""
    dev = _resolve_device(device)

    model = _load_model(arch, class_name, weights, dev, model_args)
    bayesian_model = convert_to_bayesian(model, delta=delta)
    bayesian_model.to(dev)

    n_bayesian = sum(1 for m in bayesian_model.modules()
                     if isinstance(m, (BayesianLinear, BayesianConv2d)))
    click.echo(f"Converted {n_bayesian} layers to Bayesian (delta={delta})")

    torch.save({"model_state": bayesian_model.state_dict()}, output)
    click.echo(f"Saved Bayesian model to {output}")


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
def finetune(arch: str, class_name: str, weights: str, folder: str,
             transform_file: str | None, epochs: int, lr: float, batch_size: int,
             beta_schedule: str, grad_clip: float, output: str, device: str,
             model_args: str | None) -> None:
    """Finetune a Bayesian model with ELBO loss."""
    dev = _resolve_device(device)

    model = _load_model(arch, class_name, weights, dev, model_args, ensure_bayesian=True)

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
def evaluate(arch: str, class_name: str, weights: str, folder: str,
             transform_file: str | None, mc_samples: int, batch_size: int,
             output: str | None, device: str, model_args: str | None) -> None:
    """Evaluate uncertainty of a Bayesian model on test data."""
    dev = _resolve_device(device)

    model = _load_model(arch, class_name, weights, dev, model_args, ensure_bayesian=True)

    _, _, test_loader = get_dataloaders_from_folder(
        data_dir=folder,
        batch_size=batch_size,
        use_cuda=dev.type == "cuda",
        transform_file=transform_file,
    )

    click.echo(f"Running MC evaluation (mc_samples={mc_samples})...")

    all_mean_probs: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []
    all_aleatoric: list[torch.Tensor] = []
    all_epistemic: list[torch.Tensor] = []

    for x, y in test_loader:
        x = x.to(dev)
        mc_out = mc_predict(model, x, mc_samples)
        _, uncertainties = quantify_uncertainties(mc_out)
        all_mean_probs.append(mc_out.mean(0).cpu())
        all_targets.append(y)
        all_aleatoric.append(uncertainties[1].diagonal(dim1=1, dim2=2).sum(-1).cpu())
        all_epistemic.append(uncertainties[2].diagonal(dim1=1, dim2=2).sum(-1).cpu())

    mean_probs = torch.cat(all_mean_probs)
    targets = torch.cat(all_targets)
    cat_aleatoric = torch.cat(all_aleatoric)
    cat_epistemic = torch.cat(all_epistemic)

    num_classes = mean_probs.shape[1]
    total = targets.shape[0]
    accuracy = 100.0 * (mean_probs.argmax(1) == targets).float().mean().item()

    ece, _, _ = expected_calibration_error(mean_probs, targets, num_classes=num_classes)
    sce = static_calibration_error(mean_probs, targets, num_classes=num_classes)

    results = {
        "accuracy": round(accuracy, 4),
        "ece": round(ece, 6),
        "sce": round(sce, 6),
        "mc_samples": mc_samples,
        "num_classes": num_classes,
        "num_test_samples": total,
        "uncertainty": {
            "aleatoric_mean": round(cat_aleatoric.mean().item(), 6),
            "aleatoric_std": round(cat_aleatoric.std().item(), 6),
            "epistemic_mean": round(cat_epistemic.mean().item(), 6),
            "epistemic_std": round(cat_epistemic.std().item(), 6),
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