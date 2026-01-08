from torch import nn

from models.bayesian_layers import BayesianLinear


class Net(nn.Module):
    def __init__(self, prior_sigma1, prior_sigma2, prior_pi, num_classes=10):
        super(Net, self).__init__()
        self.fc1 = BayesianLinear(
            784,
            256,
            prior_sigma1=prior_sigma1,
            prior_sigma2=prior_sigma2,
            pi=prior_pi
        )
        self.fc2 = BayesianLinear(
            256,
            256,
            prior_sigma1=prior_sigma1,
            prior_sigma2=prior_sigma2,
            pi=prior_pi
        )
        self.fc3 = BayesianLinear(
            256,
            num_classes,
            prior_sigma1=prior_sigma1,
            prior_sigma2=prior_sigma2,
            pi=prior_pi
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def kl_divergence(self):
        """Returns KL[q(w|θ) || P(w)]"""
        kl = 0
        for module in self.modules():
            if isinstance(module, BayesianLinear):
                kl += (module.log_var_posterior - module.log_prior)
        return kl
