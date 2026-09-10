import math

import torch
from datetime import datetime

class Config:
    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Random seed
    seed = 42

    # Data
    dataset = 'MNIST'
    num_classes = 10

    # Training hyperparameters
    t_train = 1
    batch_size = 128
    test_batch_size = 14
    n_epochs = 100
    learning_rate = 1e-3
    gamma = 0.95  # LR decay
    gradient_clip_norm = 1.0
    beta_schedule = 'uniform'

    prior_sigma1 = math.exp(-1)
    prior_sigma2 = math.exp(-6)
    prior_pi = 0.5
    rho_init = -3

    # Training settings
    log_interval = 100
    scheduler_step_size = 50

    # Checkpoint settings
    save_model = True
    save_interval = 10  # Save every N epochs
    checkpoint_dir = 'checkpoints'
    model_name = 'lenet_mnist_lr1em3_logprior1m1_logprior2m6_priorpip5_rhoinit_m3'

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