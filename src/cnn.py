# cnn.py

import wandb
import torch
import torch.nn as nn
import pandas as pd
import yaml
import ast
import os

from src.config import DEVICE, SEED, PATIENCE
from src.utils import set_seed, compute_weights
from src.cnn_dataset import create_dataloaders
from src.cnn_model import ImageResNet
from src.train import train_model
from src.save_best import save_best_model

from torchvision import transforms
import torchvision.models as models

def extract_first_image_path(x):
    """
    Extracts the first valid image path from a list or string representation of a list.

    Handles nested list strings safely and normalizes file paths.
    Returns None if no valid image path is found.
    """

    if isinstance(x, list):
        # already a list, take first element
        return x[0] if x else None
    elif isinstance(x, str):
        x = x.strip()
        if x.startswith("[") and x.endswith("]"):
            # string representation of list → parse safely
            try:
                # Escape backslashes so \n doesn't become newline
                safe_x = x.replace("\\", "\\\\")
                lst = ast.literal_eval(safe_x)
                if isinstance(lst, list) and len(lst) > 0:
                    return os.path.normpath(lst[0])
                return None
            except Exception:
                return None
        else:
            # plain string path → just return as is
            return x
    else:
        return None

def data_preparation(train_df, test_df):
    """
    Prepares image data and transformation pipelines for CNN training.

    Keeps only the first image per sample, removes missing paths,
    and constructs training and testing image augmentation transforms.
    """

    # Keep only the first image
    train_df["image_path"] = train_df["image_path"].apply(extract_first_image_path)
    test_df["image_path"] = test_df["image_path"].apply(extract_first_image_path)
    
    # Drop rows with no images (if any)
    train_df = train_df.dropna(subset=["image_path"]).reset_index(drop=True)
    test_df = test_df.dropna(subset=["image_path"]).reset_index(drop=True)

    # Define image transforms
    IMAGE_SIZE = 224  # ResNet expects 224x224

    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),   # data augmentation
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],        # ImageNet mean
            std=[0.229, 0.224, 0.225]          # ImageNet std
        )
    ])

    test_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    return train_transform, test_transform

def _run(config, mode):
    """
    Executes one CNN training experiment.

    Handles dataset loading, model initialization with optional backbone freezing,
    config-driven hyperparameter setup, training with class-balanced loss,
    and best checkpoint saving.
    """

    set_seed(SEED)

    # Load preprocessed features
    train_data = torch.load("features/cnn/cnn_train_meta.pt", weights_only=False)
    test_data  = torch.load("features/cnn/cnn_test_meta.pt", weights_only=False)
    train_transform = train_data["X_train"]
    y_train = train_data["y_train"]
    test_transform = test_data["X_test"]
    y_test = test_data["y_test"]
    
    # # Load split data
    train_df = pd.read_csv("data/train_df.csv",
                           parse_dates=["publish_timestamp"])
    test_df = pd.read_csv("data/test_df.csv",
                          parse_dates=["publish_timestamp"])

    # Load parameters from config
    try:
        model_name = str(config.model_name)
        dropout = float(config.dropout)
        batch_size = int(config.batch_size)
        epochs = int(config.epochs)
        freeze_resnet = (
            config.freeze_resnet if isinstance(config.freeze_resnet, bool)
            else str(config.freeze_resnet).lower() == "true"
        )
        lr_backbone = config.lr_backbone
        if lr_backbone is not None:
            lr_backbone = float(lr_backbone)
        lr_head = float(config.lr_head)
    except AttributeError as e:
        raise ValueError(f"Missing required config parameter: {e}")

    except ValueError as e:
        raise ValueError(f"Incorrect config value type: {e}")

    # set seed for dataloader shuffling order to make it deterministic
    g = torch.Generator().manual_seed(SEED)

    # DataLoader and dataset
    train_loader, test_loader = create_dataloaders(
        train_df,
        test_df,
        train_transform,
        test_transform,
        batch_size,
        g
    )

    # Load pretrained ResNet backbone
    resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    # Save number of features before removing fc
    num_features = resnet.fc.in_features # 512
    # # Replace fc with identity so backbone outputs features
    resnet.fc = nn.Identity() # to be removed in ImageResNet function
    model = ImageResNet(
        resnet_model=resnet,
        num_features=num_features,
        dropout=dropout
    ).to(DEVICE)

    if freeze_resnet:
        # Freeze entire backbone
        for param in model.backbone.parameters():
            param.requires_grad = False

        # Ensure classifier is trainable
        for param in model.classifier.parameters():
            param.requires_grad = True

        # Optimizer: only classifier
        optimizer = torch.optim.Adam(
            model.classifier.parameters(),
            lr=lr_head
        )

    else:
         
        for param in model.backbone.parameters(): # Freeze everything first
            param.requires_grad = False
        for param in model.backbone.layer4.parameters(): # Unfreeze last ResNet block
            param.requires_grad = True
        for param in model.classifier.parameters(): # Keep classifier trainable
            param.requires_grad = True

        # Optimizer with different learning rates
        optimizer = torch.optim.Adam([
            {"params": model.backbone.layer4.parameters(), "lr": lr_backbone},
            {"params": model.classifier.parameters(), "lr": lr_head},
        ])

    # Weights for class imbalance
    class_weights = compute_weights(y_train, DEVICE)
    # Loss
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

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