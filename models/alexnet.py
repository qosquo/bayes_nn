from torch import Tensor, nn
import torch.nn.functional as F
from models.bayesian_layers import BayesianConv2d, BayesianLinear, BayesianModel


class Net(BayesianModel):
    def __init__(self, prior_sigma1: float = 1.5, prior_sigma2: float = 0.5,
                 prior_pi: float = 0.5, num_classes: int = 10) -> None:
        super().__init__()
        # Feature extractor
        self.conv1 = BayesianConv2d(1, 32, kernel_size=3, padding=1,
                                    prior_sigma1=prior_sigma1, prior_sigma2=prior_sigma2, pi=prior_pi)
        self.conv2 = BayesianConv2d(32, 64, kernel_size=3, padding=1,
                                    prior_sigma1=prior_sigma1, prior_sigma2=prior_sigma2, pi=prior_pi)
        self.conv3 = BayesianConv2d(64, 128, kernel_size=3, padding=1,
                                    prior_sigma1=prior_sigma1, prior_sigma2=prior_sigma2, pi=prior_pi)

        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 28x28 -> 14x14
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 14x14 -> 7x7
        self.pool3 = nn.AdaptiveAvgPool2d((2, 2))  # 7x7 -> 2x2

        # Classifier
        self.fc1 = BayesianLinear(512, 512,
                                  prior_sigma1=prior_sigma1, prior_sigma2=prior_sigma2, pi=prior_pi)
        self.fc2 = BayesianLinear(512, 256,
                                  prior_sigma1=prior_sigma1, prior_sigma2=prior_sigma2, pi=prior_pi)
        self.fc3 = BayesianLinear(256, num_classes,
                                  prior_sigma1=prior_sigma1, prior_sigma2=prior_sigma2, pi=prior_pi)

    def forward(self, x: Tensor) -> Tensor:
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