import wandb
import torch
import yaml
import pandas as pd

from src.config import DEVICE, ENTITY, PROJECT, SEED, PATIENCE
from src.utils import set_seed, compute_weights
from src.bert_model import load_bert, CaptionBERT
from src.mlp_model import MetadataMLP
from src.cnn_model import ImageResNet
import torchvision.models as models
from src.train import train_model
from src.save_best import save_best_model

from src.fusion_model import FusionModel
from torch.utils.data import TensorDataset, DataLoader

from src.bert import data_preparation as data_prep_bert
from src.mlp import data_preparation as data_prep_mlp
from src.cnn import data_preparation as data_prep_cnn

from src.bert_dataset import create_dataloaders as dataloader_bert
from src.mlp_dataset import create_dataloaders as dataloader_mlp
from src.cnn_dataset import create_dataloaders as dataloader_cnn

from src.fusion_dataset import create_dataloaders as dataloader_fusion

def get_run_by_id(entity=ENTITY, project=PROJECT, run_id=None):
    api = wandb.Api()
    return api.run(f"{entity}/{project}/{run_id}")


def load_best_bert(run):
    config = run.config

    # Load BERT model and tokenizer
    bert_model, tokenizer = load_bert(model_name=config["model_name"],
                                    max_seq_length=config["max_len"])
    model = CaptionBERT(
        bert_model,
        dropout=config["dropout"],
        hidden_dim=config["hidden_dim"]
    ).to(DEVICE)

    state_dict = torch.load("best_model_bert.pt")
    model.load_state_dict(state_dict)
    for param in model.parameters():
        param.requires_grad = False

    model.eval()

    return model, tokenizer, config

def load_best_mlp(run):
    config = run.config

    model = MetadataMLP(
        input_dim=config["input_dim"],
        hidden_dim1=config["hidden_dim1"], 
        hidden_dim2=config["hidden_dim2"],
        dropout=config["dropout"]
    ).to(DEVICE)

    state_dict = torch.load("best_model_mlp.pt")
    model.load_state_dict(state_dict)
    for param in model.parameters():
        param.requires_grad = False

    model.eval()

    return model, config

def load_best_cnn(run):
    config = run.config

    resnet = models.resnet18(weights="IMAGENET1K_V1")
    model = ImageResNet(
        resnet,
        dropout=config["dropout"]
    ).to(DEVICE)

    state_dict = torch.load("best_model_cnn.pt")
    model.load_state_dict(state_dict)
    for param in model.parameters():
        param.requires_grad = False

    model.eval()

    return model, config

def _run(config, mode):
    """
    Core training function that both sweep and baseline call.
    """
    # # Load best models
    # with open("config/fusion_selected_runs.yaml", "r") as f:
    #     run_ids = yaml.safe_load(f)
    
    # best_bert_run = get_run_by_id(run_id=run_ids["bert"])
    # best_cnn_run = get_run_by_id(run_id=run_ids["cnn"])
    # best_mlp_run = get_run_by_id(run_id=run_ids["mlp"])

    # bert_model, tokenizer, bert_config = load_best_bert(best_bert_run)
    # cnn_model, cnn_config = load_best_cnn(best_cnn_run)
    # mlp_model, mlp_config = load_best_mlp(best_mlp_run)

    # -------------------------
    # Start of data preparation
    # -------------------------
    # set_seed(SEED)

    # # Load split data
    # train_df = pd.read_csv("data/train_df.csv",
    #                        parse_dates=["publish_timestamp"])
    # test_df = pd.read_csv("data/test_df.csv",
    #                       parse_dates=["publish_timestamp"])
    
    # Set labels
    # y_train = train_df["engagement_label"].values
    # y_test = test_df["engagement_label"].values
    
    # # (For BERT) Fill missing captions with ""
    # train_encodings, test_encodings = data_prep_bert(train_df, test_df, tokenizer, bert_config["max_len"])

    # # (For MLP) Define columns to be used in metadata model
    # X_train_proc, X_test_proc = data_prep_mlp(train_df, test_df)

    # # (For CNN) Keep only the first image
    # train_transform, test_transform = data_prep_cnn(train_df, test_df)

    # # set seed for dataloader shuffling order to make it deterministic
    # g = torch.Generator().manual_seed(SEED)

    # # (For BERT) DataLoader
    # bert_train_loader, bert_test_loader = dataloader_bert(
    #     train_encodings,
    #     test_encodings,
    #     y_train,
    #     y_test,
    #     bert_config["batch_size"],
    #     g,
    #     shuffle_train = False   # don't mix training labels
    # )

    # # (For MLP) DataLoader
    # mlp_train_loader, mlp_test_loader = dataloader_mlp(
    #     X_train_proc,
    #     X_test_proc,
    #     y_train,
    #     y_test,
    #     mlp_config["batch_size"],
    #     g,
    #     shuffle_train = False   # don't mix training labels
    # )

    # # (For CNN) DataLoader
    # cnn_train_loader, cnn_test_loader = dataloader_cnn(
    #     train_df,
    #     test_df,
    #     train_transform,
    #     test_transform,
    #     cnn_config["batch_size"],
    #     g,
    #     shuffle_train = False   # don't mix training labels
    # )

    # Create tensors for labels
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    y_test_tensor  = torch.tensor(y_test, dtype=torch.long)

    # -----------------
    # Merge into single tensor
    # -----------------
    # train_dataset = FusionFeatureDataset(
    #     bert_train_loader,
    #     cnn_train_loader,
    #     mlp_train_loader,
    #     bert_model,
    #     cnn_model,
    #     mlp_model,
    #     DEVICE
    # )

    # test_dataset = FusionFeatureDataset(
    #     bert_test_loader,
    #     cnn_test_loader,
    #     mlp_test_loader,
    #     bert_model,
    #     cnn_model,
    #     mlp_model,
    #     DEVICE
    # )

    # # Put models in eval mode
    # bert_model.eval()
    # cnn_model.eval()
    # mlp_model.eval()

    # all_features_train = []
    # all_features_test = []

    # with torch.no_grad():
    #     # ------ TRAIN ---------
    #     for bert_batch, cnn_batch, mlp_batch in zip(bert_train_loader, cnn_train_loader, mlp_train_loader):

    #         # BERT features
    #         input_ids = bert_batch["input_ids"].to(DEVICE)
    #         attention_mask = bert_batch["attention_mask"].to(DEVICE)

    #         bert_feat = bert_model(
    #             input_ids=input_ids,
    #             attention_mask=attention_mask,
    #             return_features=True
    #         )

    #         # CNN features
    #         images, _ = cnn_batch          # (X, y)
    #         images = images.to(DEVICE)

    #         cnn_feat = cnn_model(images, return_features=True)

    #         # MLP features
    #         metadata, _ = mlp_batch        # (X, y)
    #         metadata = metadata.to(DEVICE)

    #         mlp_feat = mlp_model(metadata, return_features=True)

    #         # Concatenate along feature dim
    #         fusion_feat = torch.cat([bert_feat, cnn_feat, mlp_feat], dim=1)  # (B, total_feat_dim)

    #         all_features_train.append(fusion_feat.cpu())

    #     # ------ TEST ---------
    #     for bert_batch, cnn_batch, mlp_batch in zip(bert_test_loader, cnn_test_loader, mlp_test_loader):
    #         # BERT features
    #         input_ids = bert_batch["input_ids"].to(DEVICE)
    #         attention_mask = bert_batch["attention_mask"].to(DEVICE)

    #         bert_feat = bert_model(
    #             input_ids=input_ids,
    #             attention_mask=attention_mask,
    #             return_features=True
    #         )

    #         # CNN features
    #         images, _ = cnn_batch
    #         images = images.to(DEVICE)

    #         cnn_feat = cnn_model(images, return_features=True)

    #         # MLP features
    #         metadata, _ = mlp_batch
    #         metadata = metadata.to(DEVICE)

    #         mlp_feat = mlp_model(metadata, return_features=True)

    #         # Concatenate
    #         fusion_feat = torch.cat([bert_feat, cnn_feat, mlp_feat], dim=1)

    #         all_features_test.append(fusion_feat.cpu())

    # # Stack batches into single tensors
    # X_train_fusion = torch.cat(all_features_train, dim=0)
    # X_test_fusion = torch.cat(all_features_test, dim=0)
    # # Use tensor labels
    # y_train_fusion = y_train_tensor
    # y_test_fusion  = y_test_tensor

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
    except AttributeError as e:
        raise ValueError(f"Missing required config parameter: {e}")
    except ValueError as e:
        raise ValueError(f"Incorrect config value type: {e}")
    
    X_train = torch.load("data/X_train_fusion.pt")
    X_test  = torch.load("data/X_test_fusion.pt")
    y_train = torch.load("data/y_train_fusion.pt")
    y_test  = torch.load("data/y_test_fusion.pt")

    train_dataset = TensorDataset(X_train, y_train)
    test_dataset  = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    input_dim = X_train.shape[1]

    # DataLoader and dataset
    # train_loader_fusion, test_loader_fusion = dataloader_fusion(
    #     X_train_fusion,
    #     X_test_fusion,
    #     y_train_fusion,
    #     y_test_fusion,
    #     batch_size,
    #     g,
    #     shuffle_train=True      # Mix training labels (default)
    # )
    # train_loader_fusion, test_loader_fusion = dataloader_fusion(
    #     bert_train_loader,
    #     cnn_train_loader,
    #     mlp_train_loader,
    #     bert_test_loader,
    #     cnn_test_loader,
    #     mlp_test_loader,
    #     bert_model,
    #     cnn_model,
    #     mlp_model,
    #     batch_size,
    #     g,
    #     shuffle_train=True      # Mix training labels (default)
    # )

    # Initialize fusion model
    # Example: input_dim = sum of all feature dims
    input_dim = train_loader_fusion.features.shape[1]
    fusion_model = FusionModel(input_dim=input_dim,
                            hidden_dim=hidden_dim,
                            dropout=dropout).to(DEVICE)
    
    # Weights for class imbalance
    class_weights = compute_weights(y_train_tensor.cpu().numpy(), DEVICE)

    # Loss & optimizer
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(fusion_model.parameters(), lr=learning_rate)

    # Train
    best_f1, best_state_dict = train_model(
        fusion_model,
        train_loader_fusion,
        test_loader_fusion,
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

    # # ???????
    # fusion_input_dim = (
    #     bert_config["hidden_dim"] +
    #     512 + # resnet18 output
    #     mlp_config["hidden_dim2"]
    # )

    # # fusion model
    # fusion_model = FusionModel(
    #     input_dim=fusion_input_dim,
    #     num_classes=3
    # ).to(DEVICE)

    # # optimizer
    #     optimizer = torch.optim.AdamW(
    #     fusion_model.parameters(),
    #     lr=1e-3
    # )

    # # fusion training loop
    # criterion = torch.nn.CrossEntropyLoss()

    # for epoch in range(num_epochs):

    #     fusion_model.train()

    #     for batch in train_loader:

    #         input_ids = batch["input_ids"].to(DEVICE)
    #         attention_mask = batch["attention_mask"].to(DEVICE)
    #         images = batch["image"].to(DEVICE)
    #         metadata = batch["metadata"].to(DEVICE)
    #         labels = batch["label"].to(DEVICE)

    #         with torch.no_grad():  # freeze experts
    #             bert_feat = bert_model(
    #                 input_ids,
    #                 attention_mask,
    #                 return_features=True
    #             )

    #             cnn_feat = cnn_model(
    #                 images,
    #                 return_features=True
    #             )

    #             mlp_feat = mlp_model(
    #                 metadata,
    #                 return_features=True
    #             )

    #         logits = fusion_model(
    #             bert_feat,
    #             cnn_feat,
    #             mlp_feat
    #         )

    #         loss = criterion(logits, labels)

    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()

    # fusion_model.eval()

    # # inference
    # with torch.no_grad():
    #     bert_feat = bert_model(..., return_features=True)
    #     cnn_feat = cnn_model(..., return_features=True)
    #     mlp_feat = mlp_model(..., return_features=True)

    #     logits = fusion_model(bert_feat, cnn_feat, mlp_feat)
    #     preds = torch.argmax(logits, dim=1)

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