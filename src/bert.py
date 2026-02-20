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


def _run(config, mode):
    """
    Core training function that both sweep and baseline call.
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

    # if mode == "sweep":
    #     # Skip invalid combinations
    #     if freeze_bert and learning_rate in [2e-5, 5e-5]:
    #         print(f"Skipping run: freeze_bert={freeze_bert}, learning_rate={learning_rate}")
    #         return

    #     if not freeze_bert and learning_rate in [1e-3, 5e-4]:
    #         print(f"Skipping run: freeze_bert={freeze_bert}, learning_rate={learning_rate}")
    #         return


    # Load BERT model and tokenizer
    bert_model, tokenizer = load_bert(max_len)

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

    # # Log best F1
    # wandb.log({
    #     "model": model_name,
    #     "best_macro_f1": best_f1
    # })

    # Load best weights back into model
    model.load_state_dict(best_state_dict)

    # Save only ONCE here (best model of single run in sweep)
    save_best_model(model, model_name, mode, best_f1)

    # # Save best model
    # if mode == 'baseline':
    #     save_path = f"best_model_{model_name}.pt"
    #     torch.save(model.state_dict(), save_path)
    #     print(f"Saved best model to local {save_path}")
    # else:
    #     # Save model directly to W&B without keeping a permanent local copy
    #     with tempfile.NamedTemporaryFile(suffix=".pt") as tmp_file:
    #         torch.save(model.state_dict(), tmp_file.name)
            
    #         artifact = wandb.Artifact(
    #             name=f"model-{wandb.run.id}",
    #             type="model"
    #         )
    #         artifact.add_file(tmp_file.name)
    #         wandb.log_artifact(artifact)
    #         print(f"Saved best model to W&B artifact {artifact.name}")

    # artifact = wandb.Artifact(
    #     name=f"model-{wandb.run.id}",
    #     type="model"
    # )
    # artifact.add_file(save_path)
    # wandb.log_artifact(artifact)
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
    Single-run baseline.
    config_override: dict of fixed parameters, e.g. max_len, dropout, learning_rate
    """
    # Load the baseline config from YAML
    with open(config_file, "r") as f:
        config_dict = yaml.safe_load(f)

    wandb.init(project=project, config=config_dict, save_code=False, settings=wandb.Settings(console="off"))
    config = wandb.config
    
    mode = "baseline"

    _run(config, mode)
