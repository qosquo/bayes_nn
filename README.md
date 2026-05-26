# Bayesian Neural Networks

A PyTorch toolkit for image classification with Bayesian neural networks. Includes training from scratch, hyperparameter tuning with Optuna, and a CLI tool for converting pre-trained deterministic CNNs to Bayesian NNs via the [MOPED](https://arxiv.org/abs/1906.05323) initialization method.

## Features

- **Bayesian layers** (`BayesianLinear`, `BayesianConv2d`) with scale-mixture Gaussian priors and variational inference via the reparameterization trick
- **MOPED conversion** -- turn any deterministic CNN into a Bayesian NN by initializing posterior means from pre-trained weights
- **Uncertainty quantification** -- decompose predictive uncertainty into epistemic (parameter) and aleatoric (data) components via Monte Carlo sampling
- **Calibration metrics** -- Expected Calibration Error (ECE), Static Calibration Error (SCE), MC-averaged NLL
- **Hyperparameter tuning** -- automated search over prior parameters, learning rate, KL schedule, and more with Optuna
- **Built-in architectures** -- LeNet, MLP, AlexNet (all Bayesian)

## Installation

Requires Python >= 3.10.

```bash
pip install -e .
```

**Core dependencies:** `torch>=2.0`, `torchvision`, `torchmetrics`, `click`, `tqdm`, `matplotlib`

**Optional:** `optuna` (tuning), `tensorboard` (logging), `wandb` (experiment tracking)

## Project Structure

```
bayes_nn/
├── train.py              # Training loop + CLI
├── evaluate.py           # Evaluation with uncertainty decomposition
├── tune.py               # Optuna hyperparameter search
├── convert.py            # MOPED conversion CLI (convert / finetune / evaluate)
├── converter.py          # Core conversion logic (det -> Bayesian)
├── moped.py              # MOPED initialization math
├── config.py             # Default hyperparameters
├── models/
│   ├── bayesian_layers.py  # BayesianLinear, BayesianConv2d, BayesianModel
│   ├── lenet.py            # Bayesian LeNet
│   ├── mlp.py              # Bayesian MLP
│   └── alexnet.py          # Bayesian AlexNet
├── utils/
│   ├── data.py             # Dataset loaders (MNIST, EMNIST, CIFAR, ImageFolder)
│   ├── uncertainty.py      # mc_predict(), quantify_uncertainties()
│   ├── calibration.py      # ECE, SCE, NLL
│   ├── checkpoint.py       # Save/load checkpoints
│   ├── corruptions.py      # Image corruption functions
│   └── __init__.py         # Shared helpers (import_attr, compute_beta, load_state_dict)
├── examples/               # Jupyter notebooks with experiments
└── pyproject.toml
```

## Usage

### Training a Bayesian NN

Train a Bayesian neural network on the configured dataset (default: EMNIST):

```bash
python train.py
```

Override defaults via CLI options:

```bash
python train.py \
  --model-name lenet_experiment \
  --batch-size 128 \
  --learning-rate 0.001 \
  --n-epochs 50 \
  --prior-sigma1 1.5 \
  --prior-sigma2 0.5 \
  --prior-pi 0.5
```

Training logs are written to `runs/` (TensorBoard) and checkpoints to `checkpoints/`.

### Hyperparameter Tuning

Search over Bayesian hyperparameters with Optuna:

```bash
python tune.py --n-trials 50 --epochs 30
```

Pin specific hyperparameters with `--fix`:

```bash
python tune.py --fix lr=0.001 --fix T=5 --fix grad_clip=none
```

Search space: `log_prior_sigma1`, `log_prior_sigma2`, `prior_pi`, `rho_init`, `T` (MC samples), `lr`, `beta_schedule`, `grad_clip`, `batch_size`.

### MOPED Conversion (Deterministic -> Bayesian)

Convert any pre-trained deterministic CNN to a Bayesian NN, fine-tune it with ELBO, and evaluate uncertainty via `python convert.py`.

#### 1. Convert

Replace `nn.Linear` and `nn.Conv2d` with their Bayesian counterparts, initializing posteriors from pre-trained weights:

```bash
python convert.py convert \
  --arch model.py \
  --class-name MyNet \
  --weights pretrained.pth \
  --delta 0.1 \
  --output bayesian_model.pth
```

The `--delta` parameter controls initial posterior width: smaller values produce a narrower posterior (closer to deterministic), larger values produce wider uncertainty.

#### 2. Fine-tune

Fine-tune the converted model with ELBO loss on your data (expects `train/` and `test/` subdirectories):

```bash
python convert.py finetune \
  --arch model.py \
  --class-name MyNet \
  --weights bayesian_model.pth \
  --folder ./data \
  --epochs 10 \
  --lr 1e-4 \
  --batch-size 64 \
  --beta-schedule warmup \
  --output finetuned.pth
```

Custom transforms can be provided via `--transforms transforms.py` (a Python file defining a `transform` variable).

#### 3. Evaluate

Run Monte Carlo evaluation and compute uncertainty metrics:

```bash
python convert.py evaluate \
  --arch model.py \
  --class-name MyNet \
  --weights finetuned.pth \
  --folder ./data \
  --mc-samples 20 \
  --output results.json
```

Output includes accuracy, ECE, SCE, and mean/std of aleatoric and epistemic uncertainty.

## How It Works

### Bayesian Layers

Each weight `w` is replaced by a distribution `q(w|theta) = N(mu, softplus(rho))`. During each forward pass, weights are sampled via the reparameterization trick:

```
w = mu + softplus(rho) * epsilon,  epsilon ~ N(0, 1)
```

The prior is a scale-mixture of two Gaussians: `P(w) = pi * N(0, sigma1) + (1 - pi) * N(0, sigma2)`.

### ELBO Training

The loss combines the standard cross-entropy with a KL penalty:

```
L_ELBO = CE(y, f(x)) + beta * KL[q(w|theta) || P(w)]
```

where `beta` follows a schedule (`warmup`, `uniform`, or `blundell`).

### MOPED Initialization

Given deterministic weights `w_det`, MOPED sets:

- `mu = w_det` (posterior mean = pre-trained weight)
- `sigma = delta * |w_det|` (std proportional to weight magnitude)
- `rho = softplus_inverse(sigma)` (parameterization for unconstrained optimization)

### Uncertainty Decomposition

Running `T` stochastic forward passes produces a distribution of predictions. Uncertainty decomposes as:

- **Aleatoric** (data noise): `E[diag(p) - p * p^T]`
- **Epistemic** (model uncertainty): `E[(p - p_mean)(p - p_mean)^T]`
- **Predictive** (total): aleatoric + epistemic

## Examples

The `examples/` directory contains Jupyter notebooks with experiments on MNIST and EMNIST:

- Training and evaluating Bayesian LeNet and AlexNet
- Uncertainty visualization and analysis
- Robustness under image corruptions (blur, noise)
- Cross-dataset comparisons (MNIST vs EMNIST)
- Training curves and calibration analysis

## References

- Blundell, C. et al. *Weight Uncertainty in Neural Networks* (2015). [arXiv:1505.05424](https://arxiv.org/abs/1505.05424)
- Krishnan, R. et al. *Specifying Weight Priors in Bayesian Deep Neural Networks with Empirical Bayes* (2019). [arXiv:1906.05323](https://arxiv.org/abs/1906.05323)