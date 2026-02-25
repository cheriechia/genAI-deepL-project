import wandb
import torch
import torch.nn as nn
import yaml
import pandas as pd

from src.config import DEVICE, ENTITY, PROJECT, PATIENCE
from src.utils import compute_weights
from src.bert_model import load_bert, CaptionBERT
from src.mlp_model import MetadataMLP
from src.cnn_model import ImageResNet
import torchvision.models as models
from src.train import train_model
from src.save_best import save_best_model

from src.fusion_model import FusionModel
from torch.utils.data import TensorDataset, DataLoader


def get_run_by_id(entity=ENTITY, project=PROJECT, run_id=None):
    api = wandb.Api()
    return api.run(f"{entity}/{project}/{run_id}")


def load_best_bert(run):
    config = run.config

    # Load BERT model and tokenizer
    bert_model, tokenizer = load_bert(model_name='bert-base-uncased',
                                    max_seq_length=config["max_len"])
    model = CaptionBERT(
        bert_model,
        dropout=config["dropout"],
        hidden_dim=config["hidden_dim"]
    ).to(DEVICE)

    state_dict = torch.load("models/best_model_bert.pt")
    model.load_state_dict(state_dict)
    for param in model.parameters():
        param.requires_grad = False

    model.eval()

    return model, tokenizer, config

def load_best_mlp(run, input_dim=None):
    config = run.config

    model = MetadataMLP(
        input_dim=input_dim,
        hidden_dim1=config["hidden_dim1"], 
        hidden_dim2=config["hidden_dim2"],
        dropout=config["dropout"]
    ).to(DEVICE)

    state_dict = torch.load("models/best_model_mlp.pt")
    missing, unexpected = model.load_state_dict(state_dict)  # skip mismatched classifier keys
    print("Missing MLP keys:", missing)
    print("Unexpected MLP keys:", unexpected)
    model.load_state_dict(state_dict)
    for param in model.parameters():
        param.requires_grad = False

    model.eval()

    return model, config

def load_best_cnn(run):
    config = run.config

    # Load pretrained ResNet backbone
    resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    # Save number of features before removing fc
    num_features = resnet.fc.in_features

    # Replace fc with identity so backbone outputs features
    resnet.fc = nn.Identity() # to be removed in ImageResNet function

    # Initialize ImageResNet with backbone
    model = ImageResNet(
        resnet_model=resnet,
        num_features=num_features,
        dropout=config["dropout"]
    ).to(DEVICE)

    # Load saved model checkpoint, skipping classifier mismatch
    state_dict = torch.load("models/best_model_cnn.pt")
    missing, unexpected = model.load_state_dict(state_dict)  # skip mismatched classifier keys
    print("Missing CNN keys:", missing)
    print("Unexpected CNN keys:", unexpected)
    for param in model.parameters():
        param.requires_grad = False

    model.eval()  # important if using for feature extraction

    return model, config

def _run(config, mode):
    """
    Core training function that both sweep and baseline call.
    """

    # ----------------------------
    # Prepare Fusion Model
    # -------------------------

    # Load parameters from config for fusion model
    try:
        model_name = str(config.model_name)
        hidden_dim = int(config.hidden_dim)
        dropout = float(config.dropout)
        learning_rate = float(config.learning_rate)
        epochs = int(config.epochs)
        batch_size = int(config.batch_size)
        use_second_layer = config.use_second_layer
        use_gating = config.use_gating
    except AttributeError as e:
        raise ValueError(f"Missing required config parameter: {e}")
    except ValueError as e:
        raise ValueError(f"Incorrect config value type: {e}")
    
    X_train = torch.load("features/fusion/X_train_fusion.pt")
    X_test  = torch.load("features/fusion/X_test_fusion.pt")
    y_train = torch.load("features/fusion/y_train_fusion.pt")
    y_test  = torch.load("features/fusion/y_test_fusion.pt")

    train_dataset = TensorDataset(X_train, y_train)
    test_dataset  = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    input_dim = X_train.shape[1]

    # Initialize fusion model
    # fusion_model = FusionModel(input_dim=input_dim,
    #                         hidden_dim=hidden_dim,
    #                         dropout=dropout).to(DEVICE)
    fusion_model = FusionModel(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        dropout=dropout,
        use_second_layer=use_second_layer,
        use_gating=use_gating
    ).to(DEVICE)

    
    # Weights for class imbalance
    class_weights = compute_weights(y_train.cpu().numpy(), DEVICE)

    # Loss & optimizer
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(fusion_model.parameters(), lr=learning_rate)

    # Train
    best_f1, best_state_dict = train_model(
        fusion_model,
        train_loader,
        test_loader,
        optimizer,
        criterion,
        DEVICE,
        epochs=epochs,
        patience=PATIENCE
    )


    # Load best weights
    fusion_model.load_state_dict(best_state_dict)

    # Save only ONCE here (best model of single run in sweep)
    save_best_model(fusion_model, model_name, mode, best_f1)

    
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