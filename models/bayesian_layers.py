import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def log_mix_gauss(w: Tensor, sigma1: float, sigma2: float, pi: float) -> Tensor:
    """Log probability under scale mixture prior: pi * N(0, sigma1) + (1-pi) * N(0, sigma2)."""
    g1 = torch.distributions.Normal(0, sigma1).log_prob(w)
    g2 = torch.distributions.Normal(0, sigma2).log_prob(w)
    return torch.log(pi * torch.exp(g1) + (1 - pi) * torch.exp(g2))


class _BayesianLayerBase(nn.Module):
    """Base for Bayesian layers: shared weight sampling and KL computation."""

    mu: nn.Parameter
    rho: nn.Parameter
    mu_bias: nn.Parameter
    rho_bias: nn.Parameter

    def __init__(self, prior_sigma1: float, prior_sigma2: float, pi: float) -> None:
        super().__init__()
        self.sigma1 = prior_sigma1
        self.sigma2 = prior_sigma2
        self.pi = pi
        self.log_prior: float | Tensor = 0
        self.log_var_posterior: float | Tensor = 0

    def _sample_weights(self) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        eps_w = torch.randn_like(self.mu)
        eps_b = torch.randn_like(self.mu_bias)
        sigma_w = torch.log1p(torch.exp(self.rho))
        sigma_b = torch.log1p(torch.exp(self.rho_bias))
        w = self.mu + sigma_w * eps_w
        b = self.mu_bias + sigma_b * eps_b
        return w, b, sigma_w, sigma_b, eps_w, eps_b

    def _update_kl(self, w: Tensor, b: Tensor, sigma_w: Tensor, sigma_b: Tensor,
                   eps_w: Tensor, eps_b: Tensor) -> None:
        """Compute and store log q(w|theta) and log P(w) for KL divergence."""
        log_post_w = (-(eps_w**2) / 2 - torch.log(sigma_w) - math.log(math.sqrt(2 * math.pi))).sum()
        log_post_b = (-(eps_b**2) / 2 - torch.log(sigma_b) - math.log(math.sqrt(2 * math.pi))).sum()
        self.log_var_posterior = log_post_w + log_post_b
        self.log_prior = (log_mix_gauss(w, self.sigma1, self.sigma2, self.pi).sum()
                          + log_mix_gauss(b, self.sigma1, self.sigma2, self.pi).sum())


class BayesianLinear(_BayesianLayerBase):
    def __init__(self, in_features: int, out_features: int,
                 prior_sigma1: float = 1.5, prior_sigma2: float = 0.5,
                 pi: float = 0.5, rho_init: float = -4.5,
                 init_mu: Tensor | None = None, init_rho: Tensor | None = None,
                 init_mu_bias: Tensor | None = None, init_rho_bias: Tensor | None = None) -> None:
        super().__init__(prior_sigma1, prior_sigma2, pi)
        self.in_features = in_features
        self.out_features = out_features

        # variational parameters
        self.mu = nn.Parameter(init_mu if init_mu is not None
                               else torch.Tensor(out_features, in_features).normal_(0, 0.1))
        self.rho = nn.Parameter(init_rho if init_rho is not None
                                else torch.Tensor(out_features, in_features).uniform_(rho_init - 0.5, rho_init + 0.5))

        self.mu_bias = nn.Parameter(init_mu_bias if init_mu_bias is not None
                                    else torch.Tensor(out_features).normal_(0, 0.1))
        self.rho_bias = nn.Parameter(init_rho_bias if init_rho_bias is not None
                                     else torch.Tensor(out_features).uniform_(rho_init - 0.5, rho_init + 0.5))

    def forward(self, x: Tensor) -> Tensor:
        w, b, sigma_w, sigma_b, eps_w, eps_b = self._sample_weights()
        self._update_kl(w, b, sigma_w, sigma_b, eps_w, eps_b)
        return F.linear(x, w, b)


class BayesianConv2d(_BayesianLayerBase):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int | tuple[int, int],
                 stride: int = 1, padding: int = 0,
                 prior_sigma1: float = 1.5, prior_sigma2: float = 0.5,
                 pi: float = 0.5, rho_init: float = -4.5,
                 init_mu: Tensor | None = None, init_rho: Tensor | None = None,
                 init_mu_bias: Tensor | None = None, init_rho_bias: Tensor | None = None) -> None:
        super().__init__(prior_sigma1, prior_sigma2, pi)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding

        # Variational parameters for weights
        self.mu = nn.Parameter(init_mu if init_mu is not None
                               else torch.Tensor(out_channels, in_channels, *self.kernel_size).normal_(0, 0.1))
        self.rho = nn.Parameter(init_rho if init_rho is not None
                                else torch.Tensor(out_channels, in_channels, *self.kernel_size).uniform_(rho_init - 0.5, rho_init + 0.5))

        # Variational parameters for bias
        self.mu_bias = nn.Parameter(init_mu_bias if init_mu_bias is not None
                                    else torch.Tensor(out_channels).normal_(0, 0.1))
        self.rho_bias = nn.Parameter(init_rho_bias if init_rho_bias is not None
                                     else torch.Tensor(out_channels).uniform_(rho_init - 0.5, rho_init + 0.5))

    def forward(self, x: Tensor) -> Tensor:
        w, b, sigma_w, sigma_b, eps_w, eps_b = self._sample_weights()
        self._update_kl(w, b, sigma_w, sigma_b, eps_w, eps_b)
        return F.conv2d(x, w, b, self.stride, self.padding)


class BayesianModel(nn.Module):
    """Base class for Bayesian neural networks, providing kl_divergence()."""

    def kl_divergence(self) -> Tensor | float:
        """Returns KL[q(w|theta) || P(w)] summed over all Bayesian layers."""
        kl: Tensor | float = 0
        for module in self.modules():
            if isinstance(module, (BayesianLinear, BayesianConv2d)):
                kl += (module.log_var_posterior - module.log_prior)
        return kl