# mlp.py

import wandb
import torch
import pandas as pd
import yaml
import joblib

from src.config import DEVICE, SEED, PATIENCE
from src.utils import set_seed, compute_weights
from src.mlp_dataset import create_dataloaders
from src.mlp_model import MetadataMLP
from src.train import train_model
from src.save_best import save_best_model

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def data_preparation(train_df, test_df):
    """
    Preprocesses metadata features for the MLP model.

    Applies scaling to numeric features and one-hot encoding to
    categorical features, then returns processed inputs and labels.
    Saves the fitted preprocessor for later inference use.
    """

    # Define columns to be used in metadata model
    numeric_cols = [
        "following",
        "follower_following_ratio",
        "is_weekend",
        "has_location",
        "is_carousel",
        "num_images",
        "is_sponsored",
        "caption_word_count",
        "num_hashtags"
    ]
    categorical_cols = ["day", "hour"]
    # select input features from dataframe
    X_train = train_df[numeric_cols + categorical_cols]
    X_test  = test_df[numeric_cols + categorical_cols]
    # Preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols)
        ]
    )
    # apply preprocessing
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc  = preprocessor.transform(X_test)

    # Set labels
    y_train = train_df["engagement_label"].values
    y_test = test_df["engagement_label"].values

    # save preprocessor
    joblib.dump(preprocessor, "models/preprocessor_mlp.pkl")
 
    return X_train_proc, X_test_proc, y_train, y_test

def _run(config, mode):
    """
    Executes a single training run for the metadata MLP model.

    Loads preprocessed features, initializes the model using
    config parameters, trains with class-balanced loss,
    and saves the best-performing checkpoint.
    """

    set_seed(SEED)

    # Load preprocessed features
    train_data = torch.load("features/mlp/mlp_train_inputs.pt", weights_only=False)
    test_data  = torch.load("features/mlp/mlp_test_inputs.pt", weights_only=False)
    X_train_proc = train_data["X_train"]
    y_train = train_data["y_train"]
    X_test_proc = test_data["X_test"]
    y_test = test_data["y_test"]
    
    # Load parameters from config
    try:
        model_name = str(config.model_name)
        hidden_dim1 = int(config.hidden_dim1)
        hidden_dim2 = int(config.hidden_dim2)
        dropout = float(config.dropout)
        learning_rate = float(config.learning_rate)
        epochs = int(config.epochs)
        batch_size = int(config.batch_size)
    except AttributeError as e:
        raise ValueError(f"Missing required config parameter: {e}")

    except ValueError as e:
        raise ValueError(f"Incorrect config value type: {e}")
    
    
    print("train_encodings length:", len(X_train_proc))
    print("test_encodings length:", len(X_test_proc))

    # set seed for dataloader shuffling order to make it deterministic
    g = torch.Generator().manual_seed(SEED)

    # DataLoader and dataset
    train_loader, test_loader = create_dataloaders(
        X_train_proc,
        X_test_proc,
        y_train,
        y_test,
        batch_size,
        g
    )
    
    # Input size matching one-hot expanded features
    input_dim = X_train_proc.shape[1]

    model = MetadataMLP(
        input_dim=input_dim,
        hidden_dim1=hidden_dim1, 
        hidden_dim2=hidden_dim2,
        dropout=dropout
    ).to(DEVICE)

    # Weights for class imbalance
    class_weights = compute_weights(y_train, DEVICE)

    # Loss & optimizer
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate
    )

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
    wandb.init(save_code=False, settings=wandb.Settings(console="off"))#, start_method="thread"))
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
