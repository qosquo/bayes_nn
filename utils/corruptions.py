import random
from itertools import islice
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

from utils.uncertainty import quantify_uncertainties, mc_predict


def gaussian_blur(img, kernel_size):
    from torchvision.transforms.functional import gaussian_blur
    return gaussian_blur(img, kernel_size)


def test_on_corruptions(model, img, corruptions: dict = None, classes: tuple = None, T=5):
    """Проверка изображения на разных типах искажений"""

    assert classes is not None
    assert corruptions is not None

    FIXED_MAX = 1.0
    fig, axes = plt.subplots(len(classes) + 2, len(corruptions.keys()), figsize=(15, 10))

    for col, (name, corrupt_fn) in enumerate(corruptions.items()):
        corrupted = corrupt_fn(img).unsqueeze(0)
        mc_preds = mc_predict(model, corrupted, mc_samples=T)
        mean_probs = mc_preds.mean(0)[0]
        pred, (total, alea, epis) = quantify_uncertainties(mc_preds)

        # Изображение
        axes[0, col].imshow(corrupted.cpu().squeeze(), cmap='gray')
        axes[0, col].set_title(f'{name}\nPred: {pred.item()}')
        axes[0, col].axis('off')

        for row, label in enumerate(classes, start=1):
            # Uncertainty для класса
            total_unc = total[0, label, label].item()
            alea_unc = alea[0, label, label].item()
            epis_unc = epis[0, label, label].item()

            # Class probability
            class_prob = mean_probs[label].item()

            # Построение графика
            ax = axes[row, col]
            ax.bar(
                ['T', 'A', 'E'],
                [total_unc / class_prob, alea_unc / class_prob, epis_unc / class_prob],
                color=['#3498db', '#e74c3c', '#2ecc71'],
                alpha=0.8
            )
            ax.set_ylabel('Uncertainty ({})'.format(label))
            ax.set_ylim(bottom=0)
            ax.set_ylim(0, FIXED_MAX)

        # MC-предсказания
        axes[len(classes)+1, col].bar(range(10), mean_probs.cpu().numpy())
        axes[len(classes)+1, col].set_xlabel('Class')
        axes[len(classes)+1, col].set_ylabel('Probability')
        axes[len(classes)+1, col].set_ylim([0, 1])

    plt.tight_layout()
    plt.show()


def corruptions_uncertainty(model, img, corruptions: dict = None, T=5):
    """Проверка изображения на разных типах искажений"""

    assert corruptions is not None

    FIXED_MAX = 0.3
    fig, axes = plt.subplots(2, len(corruptions.keys()), figsize=(15, 8))

    for col, (name, corrupt_fn) in enumerate(corruptions.items()):
        corrupted = corrupt_fn(img).unsqueeze(0)
        mc_preds = mc_predict(model, corrupted, mc_samples=T)
        mean_probs = mc_preds.mean(0)[0]
        pred, (total, alea, epis) = quantify_uncertainties(mc_preds)

        # Изображение
        axes[0, col].imshow(corrupted.cpu().squeeze(), cmap='gray')
        axes[0, col].set_title(f'{name}\nPred: {pred.item()}')
        axes[0, col].axis('off')

        uncertainties = {
            "AU": [alea[0, label, label].item() for label in range(10)],
            "EU": [epis[0, label, label].item() for label in range(10)],
        }

        bottom = np.zeros(10)
        for u_type, values in uncertainties.items():
            axes[1, col].bar(range(10), values, bottom=bottom, label=u_type)
            bottom += values

        axes[1, col].set_ylim(0, FIXED_MAX)
        axes[1, col].set_xticks(range(10))
        axes[1, col].set_xticklabels(range(10))
        axes[1, col].set_title(f'{name} Uncertainties')
        axes[1, col].set_xlabel('Class')
        axes[1, col].set_ylabel('Uncertainty')
        axes[1, col].legend(loc="upper right")


    plt.tight_layout()
    plt.show()

