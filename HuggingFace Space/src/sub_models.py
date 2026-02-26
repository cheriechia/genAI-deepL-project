import wandb
import torch
import torch.nn as nn
import os

from src.config import DEVICE, ENTITY, PROJECT#, PATIENCE
from src.bert_model import load_bert, CaptionBERT
from src.mlp_model import MetadataMLP
from src.cnn_model import ImageResNet
import torchvision.models as models
from huggingface_hub import hf_hub_download


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

    if os.environ.get("SPACE_ID"): # Huggingface space
        # Download model from HF Hub
        model_path = hf_hub_download(
            repo_id="chiaruiqi/instagram-posts-model",
            filename="models/best_model_bert.pt",
            token=os.environ["HF_TOKEN"]
        )
    else: # local
        # Temp model local directory for local testing
        model_path = "models/best_model_bert.pt"

    state_dict = torch.load(
        model_path,
        map_location=DEVICE
    )

    model.load_state_dict(state_dict)
    for param in model.parameters():
        param.requires_grad = False

    model.eval()

    return model, tokenizer, config

def load_best_mlp(run):
    config = run.config

    if os.environ.get("SPACE_ID"): # Huggingface space
        # Download model from HF Hub
        model_path = hf_hub_download(
            repo_id="chiaruiqi/instagram-posts-model",
            filename="models/best_model_mlp.pt",
            token=os.environ["HF_TOKEN"]
        )
    else: # local
        # Temp model local directory for local testing
        model_path = "models/best_model_mlp.pt"

    state_dict = torch.load(
        model_path,
        map_location=DEVICE
    )

    # derive input dim from state_dict
    first_weight = state_dict["fc1.weight"]
    hidden_dim1, input_dim = first_weight.shape

    second_weight = state_dict["fc2.weight"]
    hidden_dim2, _ = second_weight.shape


    model = MetadataMLP(
        input_dim=input_dim,
        hidden_dim1=hidden_dim1, 
        hidden_dim2=hidden_dim2,
        dropout=config["dropout"]
    ).to(DEVICE)

    
    missing, unexpected = model.load_state_dict(state_dict)
    print("Missing MLP keys:", missing)
    print("Unexpected MLP keys:", unexpected)
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
    resnet.fc = nn.Identity() # to be removed in ImageResNet too

    # Initialize ImageResNet with backbone
    model = ImageResNet(
        resnet_model=resnet,
        num_features=num_features,
        dropout=config["dropout"]
    ).to(DEVICE)

    if os.environ.get("SPACE_ID"): # Huggingface space
        # Download model from HF Hub
        model_path = hf_hub_download(
            repo_id="chiaruiqi/instagram-posts-model",
            filename="models/best_model_cnn.pt",
            token=os.environ["HF_TOKEN"]
        )
    else: # local
        # Temp model local directory for local testing
        model_path = "models/best_model_cnn.pt"

    # Load saved model checkpoint, skipping classifier mismatch
    state_dict = torch.load(
        model_path,
        map_location=DEVICE
    )
    missing, unexpected = model.load_state_dict(state_dict, strict=False)  # skip mismatched classifier keys
    print("Missing CNN keys:", missing)
    print("Unexpected CNN keys:", unexpected)
    for param in model.parameters():
        param.requires_grad = False

    model.eval()  # important if using for feature extraction

    return model, config