import random
from collections.abc import Callable
from itertools import islice

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from utils.uncertainty import quantify_uncertainties, mc_predict


def gaussian_blur(img: Tensor, kernel_size: int) -> Tensor:
    from torchvision.transforms.functional import gaussian_blur
    return gaussian_blur(img, kernel_size)


def plot_corrupted_images_with_uncertainty_and_probability(model: nn.Module,
                        corrupted_imgs: list[Tensor],
                        label: int | None = None,
                        mc_samples: int = 10) -> None:
    """Проверка изображения на разных типах искажений"""

    corrupted_mc_preds = [mc_predict(model, corrupted_img.unsqueeze(0), mc_samples) for corrupted_img in corrupted_imgs]
    corrupted_uncertainties = [quantify_uncertainties(mc_preds) for mc_preds in corrupted_mc_preds]

    fig, axes = plt.subplots(
        3, len(corrupted_imgs),
        figsize=(16, 9),
        gridspec_kw={
            'height_ratios': [1.5, 1, 1],
        },
        sharey='row'
    )

    for i, (corrupted_img, mc_preds) in enumerate(zip(corrupted_imgs, corrupted_mc_preds)):
        axes[0, i].imshow(corrupted_img.squeeze().cpu(), cmap='gray')
        axes[0, i].set_title(f"""
    Prediction: {mc_preds.mean(0).squeeze().argmax()},  True: {label}
        """)
        axes[0, i].axis("off")

    for i, (_, alea, epis) in enumerate(corrupted_uncertainties):
        uncertainties = {
            "AU": alea.squeeze().diag().cpu().numpy(),
            "EU": epis.squeeze().diag().cpu().numpy(),
        }

        bottom = np.zeros_like(next(iter(uncertainties.values())))
        for u_type, values in uncertainties.items():
            axes[1, i].bar(range(len(bottom)), values, bottom=bottom, label=u_type)
            bottom += values

        axes[1, i].set_xticks(range(len(bottom)))
        axes[1, i].set_xticklabels(range(len(bottom)))
        axes[1, 0].set_ylabel('Uncertainty')
        axes[1, i].legend(loc="upper right")

    for i, mc_preds in enumerate(corrupted_mc_preds):
        mean_probs = mc_preds.mean(0).squeeze()
        axes[2, i].set_ylim(0, 1)
        axes[2, i].bar(range(len(mean_probs)), mean_probs.cpu().numpy())
        axes[2, i].set_xticks(range(len(mean_probs)))
        axes[2, i].set_xticklabels(range(len(mean_probs)))
        axes[2, i].set_xlabel('Class')
        axes[2, 0].set_ylabel('Probability')

    plt.tight_layout()
    plt.show()


def corruptions_uncertainty(model: nn.Module, img: Tensor, label: int | None = None,
                            corruptions: dict[str, Callable[[Tensor], Tensor]] | None = None,
                            num_classes: int = 10, mc_samples: int = 10) -> None:
    """Проверка изображения на разных типах искажений"""

    assert corruptions is not None

    FIXED_MAX = 0.3
    fig, axes = plt.subplots(
        3,
        len(corruptions.keys()),
        figsize=(15, 9),
        gridspec_kw={
            'height_ratios': [1.5, 1, 1],
        },
        sharey='row'
    )

    for col, (name, corrupt_fn) in enumerate(corruptions.items()):
        corrupted = corrupt_fn(img).unsqueeze(0)
        mc_preds = mc_predict(model, corrupted, mc_samples)
        pred, (total, alea, epis) = quantify_uncertainties(mc_preds)


        total = total.diagonal(dim1=1, dim2=2).sum(-1)
        aleatoric = alea.diagonal(dim1=1, dim2=2).sum(-1)
        epistemic = epis.diagonal(dim1=1, dim2=2).sum(-1)

        # Изображение
        axes[0, col].imshow(corrupted.cpu().squeeze(), cmap='gray')
        axes[0, col].set_title(f"""
{name}
Prediction: {pred.item()}, True: {label}
        """
        )
        axes[0, col].axis('off')

        uncertainties = {
            "AU": [alea[0, label, label].item() for label in range(num_classes)],
            "EU": [epis[0, label, label].item() for label in range(num_classes)],
        }

        bottom = np.zeros(num_classes)
        for u_type, values in uncertainties.items():
            axes[1, col].bar(range(num_classes), values, bottom=bottom, label=u_type)
            bottom += values

        axes[1, col].set_ylim(0, FIXED_MAX)
        even_ticks = [i for i in range(num_classes) if i % 2 == 0]
        even_labels = [str(i) for i in even_ticks]
        axes[1, col].set_xticks(even_ticks)
        axes[1, col].set_xticklabels(even_labels)
        axes[1, col].set_xlabel('Class')
        axes[1, col].legend(loc="upper right")

        # MC-предсказания
        mean_probs = mc_preds.mean(0)[0]
        axes[2, col].set_ylim(0, 1)
        axes[2, col].bar(range(num_classes), mean_probs.cpu().numpy())
        axes[2, col].set_xticks(even_ticks)
        axes[2, col].set_xticklabels(even_labels)
        axes[2, col].set_xlabel('Class')

    axes[1, 0].set_ylabel('Uncertainty')
    axes[2, 0].set_ylabel('Probability')

    plt.tight_layout()
    plt.show()

