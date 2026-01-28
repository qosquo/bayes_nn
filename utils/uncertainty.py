import torch
import torch.nn.functional as F


@torch.no_grad()
def mc_predict(model, x, mc_samples=20):
    """
    Runs T stochastic forward passes.
    Returns tensor shape: [T, batch, num_classes]
    """
    model.train()
    preds = []

    for _ in range(mc_samples):
        logits = model(x)
        probs = F.softmax(logits, dim=1)
        preds.append(probs.unsqueeze(0))

    return torch.cat(preds)


@torch.no_grad()
def quantify_uncertainties(mc_preds: torch.Tensor):
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

    return mean_probs.argmax(dim=1), (aleatoric + epistemic, aleatoric, epistemic)
