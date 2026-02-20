# src/config.py

import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

NUM_CLASSES = 3
HIDDEN_DIM = 256
PATIENCE = 5

ENTITY = "crqc-nyp"
PROJECT: "instagram-posts"