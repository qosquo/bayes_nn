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
    t_train = 1
    batch_size = 64
    test_batch_size = 14
    n_epochs = 100
    learning_rate = 0.0009426338014280636
    gamma = 0.95  # LR decay
    gradient_clip_norm = 1.0
    beta_schedule = 'warmup'

    # Model architecture (BNN priors)
    prior_sigma1 = math.exp(-0.8470609173270909)
    prior_sigma2 = math.exp(-7.293222293379696)
    prior_pi = 0.44622172885322486
    rho_init = -5.724956071835678

    # Training settings
    log_interval = 100
    scheduler_step_size = 50

    # Checkpoint settings
    save_model = True
    save_interval = 10  # Save every N epochs
    checkpoint_dir = 'checkpoints'
    model_name = 'lenet_emnist_lrp9p426em04_logprior1mp847_logprior2m7p293_priorpip446_rhoinit_m5p724_batch_64_v1'

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