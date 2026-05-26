"""
Core conversion logic: deterministic CNN -> Bayesian NN via MOPED.

Recursively replaces nn.Linear -> BayesianLinear, nn.Conv2d -> BayesianConv2d,
preserving all other layers (BatchNorm, ReLU, Pool, etc.) unchanged.
"""

import torch
import torch.nn as nn
from torch import Tensor

from models.bayesian_layers import BayesianLinear, BayesianConv2d, BayesianModel
from moped import init_moped_params


def _convert_module(module: nn.Module, delta: float) -> nn.Module:
    """Recursively replace Linear/Conv2d with Bayesian counterparts."""
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            mu, rho, mu_bias, rho_bias = init_moped_params(
                child.weight.data, child.bias.data, delta
            )
            bayesian = BayesianLinear(
                child.in_features,
                child.out_features,
                init_mu=mu,
                init_rho=rho,
                init_mu_bias=mu_bias,
                init_rho_bias=rho_bias,
            )
            setattr(module, name, bayesian)
        elif isinstance(child, nn.Conv2d):
            bias = child.bias
            if bias is None:
                bias_data = torch.zeros(child.out_channels)
            else:
                bias_data = bias.data

            mu, rho, mu_bias, rho_bias = init_moped_params(
                child.weight.data, bias_data, delta
            )
            bayesian = BayesianConv2d(
                child.in_channels,
                child.out_channels,
                child.kernel_size,
                stride=child.stride,
                padding=child.padding,
                init_mu=mu,
                init_rho=rho,
                init_mu_bias=mu_bias,
                init_rho_bias=rho_bias,
            )
            setattr(module, name, bayesian)
        else:
            _convert_module(child, delta)

    return module


class BayesianModelWrapper(BayesianModel):
    """Wraps an arbitrary model with converted Bayesian layers."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)


def convert_to_bayesian(model: nn.Module, delta: float = 0.1) -> BayesianModelWrapper:
    """Convert a deterministic model to Bayesian using MOPED initialization.

    Args:
        model: Pretrained deterministic nn.Module (e.g. ResNet, LeNet).
        delta: MOPED scale factor controlling initial posterior width.
               Smaller delta = narrower posterior = closer to deterministic behavior.

    Returns:
        BayesianModelWrapper with kl_divergence() method.
    """
    _convert_module(model, delta)
    return BayesianModelWrapper(model)