# bert.py

import wandb
import torch
import pandas as pd
import yaml

from src.config import DEVICE, SEED, PATIENCE
from model import CaptionBERT
from utils import set_seed, compute_weights
from src.dataset import create_dataloaders
from src.model import load_bert, CaptionBERT
from train import train_model


def _run(config):
    """
    Core training function that both sweep and baseline call.
    config must contain:
        max_len, dropout, learning_rate, freeze_bert, batch_size, hidden_dim, epochs
    """
    set_seed(SEED)

    # Load split data
    train_df = pd.read_csv("data/train_df.csv",
                           parse_dates=["publish_timestamp"])
    test_df = pd.read_csv("data/test_df.csv",
                          parse_dates=["publish_timestamp"])
    # Fill missing captions with ""
    train_df["caption"] = train_df["caption"].fillna("")
    test_df["caption"] = test_df["caption"].fillna("")
    # Set labels
    y_train = train_df["engagement_label"].values
    y_test = test_df["engagement_label"].values

    # Load BERT model and tokenizer
    bert_model, tokenizer = load_bert(config.max_len)

    # Tokenize train and test sets separately
    train_encodings = tokenizer(
        train_df["caption"].tolist(),
        padding="max_length",
        truncation=True,
        max_length=config.max_len,
        return_tensors="pt"
    )

    test_encodings = tokenizer(
        test_df["caption"].tolist(),
        padding="max_length",
        truncation=True,
        max_length=config.max_len,
        return_tensors="pt"
    )

    # set seed for dataloader shuffling order to make it deterministic
    g = torch.Generator().manual_seed(SEED)

    # DataLoader and dataset
    train_loader, test_loader = create_dataloaders(
        train_encodings,
        test_encodings,
        y_train,
        y_test,
        config.batch_size,
        g
    )

    model = CaptionBERT(
        bert_model,
        hidden_dim=config.hidden_dim,
        dropout=config.dropout
    ).to(DEVICE)

    if config.freeze_bert:
        for p in model.bert.parameters():
            p.requires_grad = False

    # Weights for class imbalance
    class_weights = compute_weights(y_train, DEVICE)

    # Loss & optimizer
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.learning_rate
    )

    # Train
    best_f1 = train_model(
        model,
        train_loader,
        test_loader,
        optimizer,
        criterion,
        DEVICE,
        epochs=config.epochs,
        patience=PATIENCE
    )

    # Log best F1
    wandb.log({"best_macro_f1": best_f1})

    # Save best model
    torch.save(model.state_dict(), "best_model.pt")


# ----------------------------
# Functions exposed to main.py
# ----------------------------

def run_sweep():
    """
    Function to be used with wandb.agent for sweeps.
    Reads config automatically from wandb.
    """
    wandb.init()
    config = wandb.config

    # Provide default epoch count for sweeps
    if not hasattr(config, "epochs"):
        config.epochs = 50

    _run(config)


def run_baseline(config_file="baseline.yaml", project="bert-sweep"):
    """
    Single-run baseline.
    config_override: dict of fixed parameters, e.g. max_len, dropout, learning_rate
    """
    # Load the baseline config from YAML
    with open(config_file, "r") as f:
        config_dict = yaml.safe_load(f)

    wandb.init(project=project, config=config_dict)
    config = wandb.config

    # Provide default epoch count for baseline
    if not hasattr(config, "epochs"):
        config.epochs = 1  # quick baseline

    _run(config)
