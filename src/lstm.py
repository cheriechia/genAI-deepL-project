# bert.py

import wandb
import torch
import pandas as pd
import yaml
import re

from collections import Counter
from src.config import DEVICE, SEED, PATIENCE
from src.utils import set_seed, compute_weights
from src.lstm_dataset import create_dataloaders
from src.lstm_model import CaptionRNN
from src.train import train_model

# Simple tokenizer: split on spaces, remove non-alphanumeric chars
def tokenize(text):
    if not isinstance(text, str):
        return []
    # lowercase and keep only words (a-z, 0-9)
    text = text.lower()
    tokens = re.findall(r'\b\w+\b', text)
    return tokens

def encode_caption(caption, vocab, max_len):
    tokens = tokenize(caption)
    seq = [vocab.get(tok, vocab["<UNK>"]) for tok in tokens]
    # pad or truncate
    if len(seq) < max_len:
        seq += [vocab["<PAD>"]] * (max_len - len(seq))
    else:
        seq = seq[:max_len]
    return seq

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

    # Build vocab
    all_tokens = []
    for caption in train_df['caption']:
        all_tokens.extend(tokenize(caption))

    # Count frequencies and build vocab
    counter = Counter(all_tokens)
    vocab = {"<PAD>": 0, "<UNK>": 1}  # reserve 0 for padding, 1 for unknown
    for i, word in enumerate(counter.keys(), start=2):
        vocab[word] = i
    vocab_size = len(vocab)

    # Apply encoding to train and test
    train_df["caption_seq"] = train_df["caption"].apply(
        lambda x: encode_caption(x, vocab, config.max_len)
    )

    test_df["caption_seq"] = test_df["caption"].apply(
        lambda x: encode_caption(x, vocab, config.max_len)
    )

    X_train = train_df["caption_seq"].tolist()
    y_train = train_df["engagement_label"].values

    X_test = test_df["caption_seq"].tolist()
    y_test = test_df["engagement_label"].values

    # set seed for dataloader shuffling order to make it deterministic
    g = torch.Generator().manual_seed(SEED)

    # DataLoader and dataset
    train_loader, test_loader = create_dataloaders(
        X_train,
        X_test,
        y_train,
        y_test,
        config.batch_size,
        g
    )

    model = CaptionRNN(
        vocab_size=vocab_size,
        embed_dim=config.embed_dim,
        hidden_dim=config.hidden_dim,
        dropout=config.dropout
    ).to(DEVICE)

    # Weights for class imbalance
    class_weights = compute_weights(y_train, DEVICE)

    # Loss & optimizer
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(
        model.parameters(),
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
