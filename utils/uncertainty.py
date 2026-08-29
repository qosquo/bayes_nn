import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@torch.no_grad()
def mc_predict(model: nn.Module, x: Tensor, mc_samples: int = 10) -> Tensor:
    """
    Runs T stochastic forward passes.

    :param model: The model to evaluate.
    :param x: Input tensor of shape [batch, ...].
    :param mc_samples: Number of Monte Carlo samples (forward passes).
    :return: Tensor shape: [T, batch, num_classes]
    """
    model.train()
    preds = []

    for _ in range(mc_samples):
        logits = model(x)
        probs = F.softmax(logits, dim=1)
        preds.append(probs.unsqueeze(0))

    return torch.cat(preds)


@torch.no_grad()
def quantify_uncertainties(mc_preds: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """
    Quantifies aleatoric and epistemic uncertainties from MC predictions.
    :param mc_preds: Class probabilities from `T` Monte Carlo forward passes.
            Shape: `[T, B, C]`, where:
                `T` = number of MC samples,
                `B` = batch size,
                `C` = number of classes.
            Each mc_preds[t, b] is a probability vector over C classes.
    :return: Tuple of (total, aleatoric, epistemic)
            uncertainty matrices, each of shape [B, C, C].
    """
    T = mc_preds.shape[0]
    # Средние вероятности по T проходам: [batch_size, num_classes]
    mean_probs = torch.mean(mc_preds, dim=0)
    # Aleatoric: E[diag(p) - p⊗p]
    aleatoric = (
            torch.diag_embed(mc_preds).mean(dim=0) -
            torch.einsum('tbi,tbj->bij', mc_preds, mc_preds) / T
    )

    # Epistemic: E[(p - p̄)⊗(p - p̄)]
    deviation = mc_preds - mean_probs
    epistemic = torch.einsum('tbi,tbj->bij', deviation, deviation) / T

    return aleatoric + epistemic, aleatoric, epistemic
