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


def quantify_uncertainties(model, x_star, T=5):
    mc_preds = mc_predict(model, x_star, T)

    # Средние вероятности по T проходам: [batch_size, num_classes]
    mean_probs = torch.mean(mc_preds, dim=0)
    _, predicted = torch.max(mean_probs.data, 1)

    # ---------- Алеаторная неопределенность ----------
    diag_each = torch.diag_embed(mc_preds)
    outer_each = torch.matmul(
        mc_preds.unsqueeze(-1),
        mc_preds.unsqueeze(-2)
    )
    # [batch_size, num_classes, num_classes]
    aleatoric = torch.mean(diag_each - outer_each, dim=0)

    # ---------- Эпистемическая неопределенность ----------
    deviation = mc_preds - mean_probs.unsqueeze(0)  # [T, batch_size, num_classes]
    # [T, batch_size, num_classes, num_classes]
    epistemic_each = torch.matmul(
        deviation.unsqueeze(-1),      # [T, batch_size, num_classes, 1]
        deviation.unsqueeze(-2)       # [T, batch_size, 1, num_classes]
    )

    # [batch_size, num_classes, num_classes]
    epistemic = torch.mean(epistemic_each, dim=0)

    # ---------- Общая неопределенность ----------
    # [batch_size, num_classes, num_classes]
    total = aleatoric + epistemic

    return predicted, (total, aleatoric, epistemic)
