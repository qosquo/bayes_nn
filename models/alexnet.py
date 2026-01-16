from torch import nn
import torch.nn.functional as F
from models.bayesian_layers import BayesianConv2d, BayesianLinear


class Net(nn.Module):
    def __init__(self, prior_sigma1=1.5, prior_sigma2=0.5, prior_pi=0.5, num_classes=10):
        super().__init__()
        # Feature extractor
        self.conv1 = BayesianConv2d(1, 32, kernel_size=3, padding=1,
                                    prior_sigma1=prior_sigma1, prior_sigma2=prior_sigma2, pi=prior_pi)
        self.conv2 = BayesianConv2d(32, 64, kernel_size=3, padding=1,
                                    prior_sigma1=prior_sigma1, prior_sigma2=prior_sigma2, pi=prior_pi)
        self.conv3 = BayesianConv2d(64, 128, kernel_size=3, padding=1,
                                    prior_sigma1=prior_sigma1, prior_sigma2=prior_sigma2, pi=prior_pi)

        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 28x28 → 14x14
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 14x14 → 7x7
        self.pool3 = nn.AdaptiveAvgPool2d((2, 2))  # 7x7 → 2x2

        # Classifier
        self.fc1 = BayesianLinear(512, 512,
                                  prior_sigma1=prior_sigma1, prior_sigma2=prior_sigma2, pi=prior_pi)
        self.fc2 = BayesianLinear(512, 256,
                                  prior_sigma1=prior_sigma1, prior_sigma2=prior_sigma2, pi=prior_pi)
        self.fc3 = BayesianLinear(256, num_classes,
                                  prior_sigma1=prior_sigma1, prior_sigma2=prior_sigma2, pi=prior_pi)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def kl_divergence(self):
        kl = 0
        for module in self.modules():
            if isinstance(module, (BayesianLinear, BayesianConv2d)):
                kl += (module.log_var_posterior - module.log_prior)
        return kl