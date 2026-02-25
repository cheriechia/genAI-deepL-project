# bert.py

import wandb
import torch
import pandas as pd
import yaml

from src.config import DEVICE, SEED, PATIENCE
from src.utils import set_seed, compute_weights
from src.bert_dataset import create_dataloaders
from src.bert_model import load_bert, CaptionBERT
from src.train import train_model
from src.save_best import save_best_model

def data_preparation(train_df, test_df, tokenizer, max_len):
    """
    Prepares caption text data for BERT encoding.

    Handles missing captions, applies tokenizer preprocessing,
    and returns tokenized training and testing encodings as tensors.
    """

    # Fill missing captions with ""
    train_df["caption"] = train_df["caption"].fillna("")
    test_df["caption"] = test_df["caption"].fillna("")

    # Tokenize train and test sets separately
    train_encodings = tokenizer(
        train_df["caption"].tolist(),
        padding="max_length",
        truncation=True,
        max_length=max_len,
        return_tensors="pt"
    )

    test_encodings = tokenizer(
        test_df["caption"].tolist(),
        padding="max_length",
        truncation=True,
        max_length=max_len,
        return_tensors="pt"
    )

    return train_encodings, test_encodings

def _run(config, mode):
    """
    Executes one BERT classification training experiment.

    Loads tokenized caption data, initializes BERT-based classifier,
    configures model freezing strategy, trains with class-balanced loss,
    and saves the best checkpoint based on validation performance.
    """

    set_seed(SEED)

    # Load preprocessed features
    train_data = torch.load("features/bert/bert_train_inputs.pt", weights_only=False)
    test_data  = torch.load("features/bert/bert_test_inputs.pt", weights_only=False)
    train_encodings = train_data["X_train"]
    y_train = train_data["y_train"]
    test_encodings = test_data["X_test"]
    y_test = test_data["y_test"]

    # Load parameters from config
    try:
        model_name = str(config.model_name)
        max_len = int(config.max_len)
        batch_size = int(config.batch_size)
        hidden_dim = int(config.hidden_dim)
        dropout = float(config.dropout)
        learning_rate = float(config.learning_rate)
        epochs = int(config.epochs)

        freeze_bert = (
            config.freeze_bert if isinstance(config.freeze_bert, bool)
            else str(config.freeze_bert).lower() == "true"
        )

    except AttributeError as e:
        raise ValueError(f"Missing required config parameter: {e}")

    except ValueError as e:
        raise ValueError(f"Incorrect config value type: {e}")

    # Load BERT model and tokenizer
    bert_model, tokenizer = load_bert(model_name='bert-base-uncased')

    # set seed for dataloader shuffling order to make it deterministic
    g = torch.Generator().manual_seed(SEED)

    # DataLoader and dataset
    train_loader, test_loader = create_dataloaders(
        train_encodings,
        test_encodings,
        y_train,
        y_test,
        batch_size,
        g
    )

    model = CaptionBERT(
        bert_model,
        hidden_dim=hidden_dim,
        dropout=dropout
    ).to(DEVICE)

    # Weights for class imbalance
    class_weights = compute_weights(y_train, DEVICE)

    # Loss & optimizer
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

    if freeze_bert: # prevents BERT weights from being updated
        for p in model.bert.parameters(): # for all trainable tensors in BERT
            p.requires_grad = False         # don't compute gradients for tensor
        # Set optimizer for frozen BERT
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=learning_rate
        )
    else:
        for param in model.bert.parameters(): # freeze everything first
            param.requires_grad = False 
        for layer in model.bert.encoder.layer[-2:]: # unfreeze top 2 BERT layers
            for param in layer.parameters():
                param.requires_grad = True
        for param in model.fc_hidden.parameters(): # train classifier head
            param.requires_grad = True
        for param in model.fc_out.parameters():
            param.requires_grad = True
        # Set optimizer from partially unfrozen BERT
        optimizer = torch.optim.AdamW([
            {"params": model.bert.encoder.layer[-2:].parameters(), "lr": 2e-5},
            {"params": model.fc_hidden.parameters(), "lr": learning_rate},
            {"params": model.fc_out.parameters(), "lr": learning_rate},
        ])

    # Train
    best_f1, best_state_dict = train_model(
        model,
        train_loader,
        test_loader,
        optimizer,
        criterion,
        DEVICE,
        epochs=epochs,
        patience=PATIENCE
    )

    # Load best weights back into model
    model.load_state_dict(best_state_dict)

    # Save only ONCE here (best model of single run in sweep)
    save_best_model(model, model_name, mode, best_f1)

    
# ----------------------------
# Functions exposed to main.py
# ----------------------------

def run_sweep():
    """
    Function to be used with wandb.agent for sweeps.
    Reads config automatically from wandb.
    """
    wandb.init(save_code=False, settings=wandb.Settings(console="off"))
    config = wandb.config

    mode = "sweep"
    _run(config, mode)


def run_baseline(config_file="baseline.yaml", project="instagram-posts"):
    """
    Baseline run with 1 set of fixed parameters from config/[model]_baseline.yaml
    Initializes wandb with config and reads config from wandb.
    """
    # Load the baseline config from YAML
    with open(config_file, "r") as f:
        config_dict = yaml.safe_load(f)

    wandb.init(project=project, config=config_dict, save_code=False, settings=wandb.Settings(console="off"))
    config = wandb.config

    mode = "baseline"

    _run(config, mode)
