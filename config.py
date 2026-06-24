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
    num_classes = 18

    # Training hyperparameters
    t_train = 1
    batch_size = 128
    test_batch_size = 14
    n_epochs = 100
    learning_rate = 0.0007825211241526634
    gamma = 0.95  # LR decay
    gradient_clip_norm = 1.0
    beta_schedule = 'warmup'

    # Model architecture (BNN priors) — tuned on EMNIST-Letters 18 classes
    prior_sigma1 = math.exp(-1.3636347242568807)  # 0.2557
    prior_sigma2 = math.exp(-6.836981395452928)   # 0.001073
    prior_pi = 0.6918513223148149
    rho_init = -5.997174678029607

    # Training settings
    log_interval = 100
    scheduler_step_size = 50

    # Checkpoint settings
    save_model = True
    save_interval = 10  # Save every N epochs
    checkpoint_dir = 'checkpoints'
    model_name = 'lenet_emnist_nc18_lr7p825em04_logprior1mp364_logprior2m6p837_priorpip692_rhoinit_m5p997_batch_128_v1'

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