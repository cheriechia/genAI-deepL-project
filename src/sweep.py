# src/sweep.py

import wandb
import torch
import pandas as pd

from src.config import DEVICE, SEED, PATIENCE
from src.utils import set_seed, compute_weights
from src.dataset import create_dataloaders
from src.model import load_bert, CaptionBERT
from src.train import train_model


def sweep():

    wandb.init()
    config = wandb.config

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

    class_weights = compute_weights(y_train, DEVICE)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.learning_rate
    )

    best_f1 = train_model(
        model,
        train_loader,
        test_loader,
        optimizer,
        criterion,
        DEVICE,
        epochs=50,
        patience=PATIENCE
    )

    wandb.log({"best_macro_f1": best_f1})
