import torch
from datetime import datetime

class Config:
    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Random seed
    seed = 42

    # Data
    data_mean = 0.1307
    data_std = 0.3081
    num_classes = 10

    # Training hyperparameters
    batch_size = 128
    test_batch_size = 14
    n_epochs = 100
    learning_rate = 1e-3
    gamma = 0.95  # LR decay

    # Model architecture (BNN priors)
    prior_sigma1 = 1.5
    prior_sigma2 = 0.5
    prior_pi = 0.5

    # Training settings
    log_interval = 100
    gradient_clip_norm = 1.0
    scheduler_step_size = 50

    # Checkpoint settings
    save_model = True
    save_interval = 25  # Save every N epochs
    checkpoint_dir = 'checkpoints'
    model_name = 'mnist_bnn_mlp'

    # Google Drive (for Colab)
    use_drive = False  # Set True when running on Colab
    drive_path = '/content/drive/MyDrive/Colab Notebooks/mnist_bnn'

    # CUDA settings
    num_workers = 1 if torch.cuda.is_available() else 0
    pin_memory = True if torch.cuda.is_available() else False

    # Uncertainty quantification
    mc_samples = 5

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
            
        return f'{base}/{self.checkpoint_dir}/{self.model_name}.pth'

    def get_checkpoint_name(self, epoch):
        today = datetime.now().strftime('%Y%m%d')
        return f'{self.model_name}_epoch_{epoch}_{today}.pth'