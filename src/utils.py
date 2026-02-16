# src/utils.py

import random
import numpy as np
import torch
from sklearn.utils.class_weight import compute_class_weight


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_weights(y, device):
    classes = np.unique(y)
    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y
    )
    return torch.tensor(weights, dtype=torch.float).to(device)
