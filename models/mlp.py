from torch import Tensor, nn

from models.bayesian_layers import BayesianLinear, BayesianModel


class Net(BayesianModel):
    def __init__(self, prior_sigma1: float, prior_sigma2: float, prior_pi: float,
                 num_classes: int = 10) -> None:
        super(Net, self).__init__()
        self.fc1 = BayesianLinear(
            784,
            400,
            prior_sigma1=prior_sigma1,
            prior_sigma2=prior_sigma2,
            pi=prior_pi
        )
        self.fc2 = BayesianLinear(
            400,
            400,
            prior_sigma1=prior_sigma1,
            prior_sigma2=prior_sigma2,
            pi=prior_pi
        )
        self.fc3 = BayesianLinear(
            400,
            400,
            prior_sigma1=prior_sigma1,
            prior_sigma2=prior_sigma2,
            pi=prior_pi
        )
        self.fc4 = BayesianLinear(
            400,
            num_classes,
            prior_sigma1=prior_sigma1,
            prior_sigma2=prior_sigma2,
            pi=prior_pi
        )
        self.relu = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        x = x.view(x.size(0), -1)  # Flatten
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x