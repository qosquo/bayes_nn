import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features, prior_sigma1=1.5, prior_sigma2=0.5, pi=0.5, rho_init=-4.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # variational parameters
        self.mu = nn.Parameter(torch.Tensor(out_features, in_features).normal_(0, 0.1))
        self.rho = nn.Parameter(torch.Tensor(out_features, in_features).uniform_(rho_init - 0.5, rho_init + 0.5))

        self.mu_bias = nn.Parameter(torch.Tensor(out_features).normal_(0, 0.1))
        self.rho_bias = nn.Parameter(torch.Tensor(out_features).uniform_(rho_init - 0.5, rho_init + 0.5))

        # scale mixture prior
        self.sigma1 = prior_sigma1
        self.sigma2 = prior_sigma2
        self.pi = pi

        self.log_prior = 0
        self.log_var_posterior = 0

    def _sample_weights(self):
        eps_w = torch.randn(self.mu.size(), device=self.mu.device)
        eps_b = torch.randn(self.mu_bias.size(), device=self.mu_bias.device)

        sigma_w = torch.log1p(torch.exp(self.rho))
        sigma_b = torch.log1p(torch.exp(self.rho_bias))

        w = self.mu + sigma_w * eps_w
        b = self.mu_bias + sigma_b * eps_b

        return w, b, sigma_w, sigma_b, eps_w, eps_b

    def forward(self, x):
        w, b, sigma_w, sigma_b, eps_w, eps_b = self._sample_weights()

        log_posterior_w = (- (eps_w**2) / 2 - torch.log(sigma_w) - math.log(math.sqrt(2*math.pi))).sum()
        log_posterior_b = (- (eps_b**2) / 2 - torch.log(sigma_b) - math.log(math.sqrt(2*math.pi))).sum()
        self.log_var_posterior = log_posterior_w + log_posterior_b

        def log_mix_gauss(w, sigma1, sigma2, pi):
            g1 = torch.distributions.Normal(0, sigma1).log_prob(w)
            g2 = torch.distributions.Normal(0, sigma2).log_prob(w)
            return torch.log(pi*torch.exp(g1) + (1-pi)*torch.exp(g2))

        self.log_prior = log_mix_gauss(w, self.sigma1, self.sigma2, self.pi).sum() \
                         + log_mix_gauss(b, self.sigma1, self.sigma2, self.pi).sum()

        return F.linear(x, w, b)

class BayesianConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 prior_sigma1=1.5, prior_sigma2=0.5, pi=0.5, rho_init=-4.5):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding

        # Variational parameters for weights
        self.mu = nn.Parameter(torch.Tensor(out_channels, in_channels, *self.kernel_size).normal_(0, 0.1))
        self.rho = nn.Parameter(torch.Tensor(out_channels, in_channels, *self.kernel_size).uniform_(rho_init - 0.5, rho_init + 0.5))

        # Variational parameters for bias
        self.mu_bias = nn.Parameter(torch.Tensor(out_channels).normal_(0, 0.1))
        self.rho_bias = nn.Parameter(torch.Tensor(out_channels).uniform_(rho_init - 0.5, rho_init + 0.5))

        # Scale mixture prior
        self.sigma1 = prior_sigma1
        self.sigma2 = prior_sigma2
        self.pi = pi

        self.log_prior = 0
        self.log_var_posterior = 0

    def _sample_weights(self):
        eps_w = torch.randn_like(self.mu)
        eps_b = torch.randn_like(self.mu_bias)

        sigma_w = torch.log1p(torch.exp(self.rho))
        sigma_b = torch.log1p(torch.exp(self.rho_bias))

        w = self.mu + sigma_w * eps_w
        b = self.mu_bias + sigma_b * eps_b

        return w, b, sigma_w, sigma_b, eps_w, eps_b

    def forward(self, x):
        w, b, sigma_w, sigma_b, eps_w, eps_b = self._sample_weights()

        # Log variational posterior q(w|θ)
        log_posterior_w = (- (eps_w**2) / 2 - torch.log(sigma_w) - math.log(math.sqrt(2*math.pi))).sum()
        log_posterior_b = (- (eps_b**2) / 2 - torch.log(sigma_b) - math.log(math.sqrt(2*math.pi))).sum()
        self.log_var_posterior = log_posterior_w + log_posterior_b

        # Log scale mixture prior P(w)
        def log_mix_gauss(w, sigma1, sigma2, pi):
            g1 = torch.distributions.Normal(0, sigma1).log_prob(w)
            g2 = torch.distributions.Normal(0, sigma2).log_prob(w)
            return torch.log(pi*torch.exp(g1) + (1-pi)*torch.exp(g2))

        self.log_prior = log_mix_gauss(w, self.sigma1, self.sigma2, self.pi).sum() \
                         + log_mix_gauss(b, self.sigma1, self.sigma2, self.pi).sum()

        return F.conv2d(x, w, b, self.stride, self.padding)
