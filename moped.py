"""
MOPED: Multi-task Optimal Posterior for Estimation of Distributions
(Krishnan et al., 2019 — arXiv:1906.05323)

Initializes Bayesian layer parameters from pretrained deterministic weights.
mu = pretrained weight, rho = softplus_inverse(delta * |mu|)
"""

import torch
from torch import Tensor


def compute_moped_rho(mu: Tensor, delta: float) -> Tensor:
    """Compute rho from mu using MOPED formula: rho = log(exp(delta * |mu|) - 1).

    This is the inverse softplus so that softplus(rho) = delta * |mu|,
    i.e. the initial standard deviation is proportional to the weight magnitude.
    """
    sigma = delta * mu.abs()
    # Inverse softplus: rho = log(exp(sigma) - 1)
    # For numerical stability, when sigma is large: rho ≈ sigma
    rho = torch.where(
        sigma > 20.0,
        sigma,
        torch.log(torch.expm1(sigma)),
    )
    # Clamp to avoid -inf for zero weights
    rho = rho.clamp(min=-7.0)
    return rho


def init_moped_params(weight: Tensor, bias: Tensor, delta: float):
    """Convert deterministic weight/bias to MOPED-initialized Bayesian parameters.

    Returns:
        (mu, rho, mu_bias, rho_bias)
    """
    mu = weight.clone()
    rho = compute_moped_rho(mu, delta)
    mu_bias = bias.clone()
    rho_bias = compute_moped_rho(mu_bias, delta)
    return mu, rho, mu_bias, rho_bias