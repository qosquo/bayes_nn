import math

import torch
from datetime import datetime

class Config:
    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Random seed
    seed = 42

    # Data
    dataset = 'EMNIST'
    data_mean = 0.1307
    data_std = 0.3081
    num_classes = 10
    # 10 letters drawn with: random.seed(42); sorted(random.sample(range(1, 27), 10))
    # Raw EMNIST labels (1-26) → A=1 C=3 D=4 E=5 H=8 I=9 R=18 U=21 V=22 Y=25
    selected_classes: list[int] = [1, 3, 4, 5, 8, 9, 18, 21, 22, 25]

    # Training hyperparameters
    t_train = 1
    batch_size = 128
    test_batch_size = 14
    n_epochs = 100
    learning_rate = 0.0012612947431859787
    gamma = 0.95  # LR decay
    gradient_clip_norm = 1.0
    beta_schedule = 'uniform'

    # Model architecture (BNN priors) — tuned on EMNIST-Letters 10 classes (phase 1)
    prior_sigma1 = math.exp(-1.1274774168136914)  # 0.3238
    prior_sigma2 = math.exp(-6.823377519611586)   # 0.001088
    prior_pi = 0.21995714027396807
    rho_init = -5.53116734457342

    # Training settings
    log_interval = 100
    scheduler_step_size = 50

    # Checkpoint settings
    save_model = True
    save_interval = 10  # Save every N epochs
    checkpoint_dir = 'checkpoints'
    model_name = 'lenet_emnist_nc10_lr1p2612947431859787_logprior1m1p127_logprior2m6p823_priorpip219957_rhoinit_m5p53116734457342'

    # Google Drive (for Colab)
    use_drive = False  # Set True when running on Colab
    drive_path = '/content/drive/MyDrive/Colab Notebooks/mnist_bnn'

    # CUDA settings
    num_workers = 1 if torch.cuda.is_available() else 0
    pin_memory = True if torch.cuda.is_available() else False

    # Uncertainty quantification
    mc_samples = 10

    @property
    def checkpoint_path(self) -> str:
        try:
            from IPython import get_ipython
            # Check if in Jupyter
            if get_ipython() and get_ipython().__class__.__name__ == 'ZMQInteractiveShell':
                base = '..'  # Go up one level in Jupyter
            else:
                base = self.drive_path if self.use_drive else '.'
        except:
            base = self.drive_path if self.use_drive else '.'

        return f'{base}/{self.checkpoint_dir}'

    def get_checkpoint_name(self, epoch: int, date: str | None) -> str:
        if date is None:
            date = datetime.now().strftime('%Y%m%d')
        return f'{self.model_name}_epoch_{epoch}_{date}.pth'