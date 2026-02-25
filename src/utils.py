# src/utils.py

import random
import numpy as np
import torch
from sklearn.utils.class_weight import compute_class_weight


def set_seed(seed=42):
    """
    Sets random seeds for reproducibility across Python, NumPy, and PyTorch.
    Configures deterministic CUDA behavior for consistent experiment results.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_weights(y, device):
    """
    Computes class-balanced weights from label distribution and returns them as a tensor on the specified device.
    Used to mitigate class imbalance during training.
    """

    classes = np.unique(y)
    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y
    )
    return torch.tensor(weights, dtype=torch.float).to(device)
