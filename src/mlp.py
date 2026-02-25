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


    # # Get new column names from ColumnTransformer
    # num_features = numeric_cols
    # cat_features = list(preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_cols))
    # all_features = num_features + cat_features

    # # Replace train_df and test_df
    # train_df_proc = pd.DataFrame(X_train_proc, columns=all_features, index=train_df.index)
    # test_df_proc  = pd.DataFrame(X_test_proc, columns=all_features, index=test_df.index)
    
    # train_df_proc, test_df_proc, 
    return X_train_proc, X_test_proc, y_train, y_test

def _run(config, mode):
    """
    Core training function that both sweep and baseline call.
    config must contain:
        max_len, dropout, learning_rate, freeze_bert, batch_size, hidden_dim, epochs
    """
    set_seed(SEED)

    # Load preprocessed features
    train_data = torch.load("features/mlp/mlp_train_inputs.pt", weights_only=False)
    test_data  = torch.load("features/mlp/mlp_test_inputs.pt", weights_only=False)
    X_train_proc = train_data["X_train"]
    y_train = train_data["y_train"]
    X_test_proc = test_data["X_test"]
    y_test = test_data["y_test"]
    # # Load split data
    # train_df = pd.read_csv("data/train_df.csv",
    #                        parse_dates=["publish_timestamp"])
    # test_df = pd.read_csv("data/test_df.csv",
    #                       parse_dates=["publish_timestamp"])
    
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
    
    # # Define columns to be used in metadata model
    # numeric_cols = [
    #     "following",
    #     "follower_following_ratio",
    #     "is_weekend",
    #     "has_location",
    #     "is_carousel",
    #     "num_images",
    #     "is_sponsored",
    #     "caption_word_count",
    #     "num_hashtags"
    # ]
    # categorical_cols = ["day", "hour"]
    # # select input features from dataframe
    # X_train = train_df[numeric_cols + categorical_cols]
    # X_test  = test_df[numeric_cols + categorical_cols]
    # # Preprocessor
    # preprocessor = ColumnTransformer(
    #     transformers=[
    #         ("num", StandardScaler(), numeric_cols),
    #         ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols)
    #     ]
    # )
    # # apply preprocessing
    # X_train_proc = preprocessor.fit_transform(X_train)
    # X_test_proc  = preprocessor.transform(X_test)
    
    # # Data preparation
    # X_train_proc, X_test_proc = data_preparation(train_df, test_df)

    # Set labels
    # y_train = train_df["engagement_label"].values
    # y_test = test_df["engagement_label"].values
    # print("Train size:", len(train_df))
    # print("Test size:", len(test_df))

    # print("y_train length:", len(y_train))
    # print("y_test length:", len(y_test))


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

    # # Log best F1
    # wandb.log({
    #     "model": model_name,
    #     "best_macro_f1": best_f1
    # })

    # # Save best model
    # save_path = f"best_model_{model_name}.pt"
    # torch.save(model.state_dict(), save_path)
    # print(f"Saved best model to {save_path}")


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
