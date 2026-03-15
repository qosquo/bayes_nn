from torch import nn
import torch.nn.functional as F

from models.bayesian_layers import BayesianConv2d, BayesianLinear


class Net(nn.Module):
    def __init__(self, prior_sigma1, prior_sigma2, prior_pi, num_classes=10, rho_init=-4.5):
        super(Net, self).__init__()
        self.conv1 = BayesianConv2d(
            1,
            6,
            kernel_size=5,
            stride=1,
            padding=2,
            prior_sigma1=prior_sigma1,
            prior_sigma2=prior_sigma2,
            pi=prior_pi,
            rho_init=rho_init,
        )  # 28x28 -> 28x28
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)  # 28x28 -> 14x14
        self.conv2 = BayesianConv2d(
            6,
            16,
            kernel_size=5,
            stride=1,
            prior_sigma1=prior_sigma1,
            prior_sigma2=prior_sigma2,
            pi=prior_pi,
            rho_init=rho_init,
        )  # 14x14 -> 10x10
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)  # 10x10 -> 5x5

        # Fully connected layers
        self.fc1 = BayesianLinear(
            16 * 5 * 5,
            120,
            prior_sigma1=prior_sigma1,
            prior_sigma2=prior_sigma2,
            pi=prior_pi,
            rho_init=rho_init,
        )
        self.fc2 = BayesianLinear(
            120,
            84,
            prior_sigma1=prior_sigma1,
            prior_sigma2=prior_sigma2,
            pi=prior_pi,
            rho_init=rho_init,
        )
        self.fc3 = BayesianLinear(
            84,
            num_classes,
            prior_sigma1=prior_sigma1,
            prior_sigma2=prior_sigma2,
            pi=prior_pi,
            rho_init=rho_init,
        )

    def forward(self, x):
        x = F.tanh(self.conv1(x))
        x = self.pool1(x)
        x = F.tanh(self.conv2(x))
        x = self.pool2(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        x = self.fc3(x)
        return x


    def kl_divergence(self):
        """Returns KL[q(w|θ) || P(w)]"""
        kl = 0
        for module in self.modules():
            if isinstance(module, (BayesianLinear, BayesianConv2d)):
                kl += (module.log_var_posterior - module.log_prior)
        return kl
