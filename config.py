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
    num_classes = 26

    # Training hyperparameters
    batch_size = 256
    test_batch_size = 14
    n_epochs = 200
    learning_rate = 5e-5
    gamma = 0.95  # LR decay

    # Model architecture (BNN priors)
    prior_sigma1 = math.exp(0.25)
    prior_sigma2 = math.exp(-6)
    prior_pi = 0.25

    # Training settings
    log_interval = 100
    gradient_clip_norm = 1.0
    scheduler_step_size = 50

    # Checkpoint settings
    save_model = True
    save_interval = 10  # Save every N epochs
    checkpoint_dir = 'checkpoints'
    model_name = 'lenet_mnist_lrp5em05_logprior1p25_logprior2m06_priorpip25_batch_256_v1'

    # Google Drive (for Colab)
    use_drive = False  # Set True when running on Colab
    drive_path = '/content/drive/MyDrive/Colab Notebooks/mnist_bnn'

    # CUDA settings
    num_workers = 1 if torch.cuda.is_available() else 0
    pin_memory = True if torch.cuda.is_available() else False

    # Uncertainty quantification
    mc_samples = 10

    @property
    def checkpoint_path(self):
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

    def get_checkpoint_name(self, epoch, date: str):
        if date is None:
            date = datetime.now().strftime('%Y%m%d')
        return f'{self.model_name}_epoch_{epoch}_{date}.pth'