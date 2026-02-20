# src/cnn_model.py

import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

resnet = models.resnet18(weights="IMAGENET1K_V1")
class ImageResNet(nn.Module):
    def __init__(self, resnet_model, num_classes=3, dropout=0.5):
        super().__init__()
        self.resnet = resnet_model
        num_features = self.resnet.fc.in_features
        resnet_model.fc = nn.Identity()  # remove original fc
        self.backbone = resnet_model   # backbone outputs 512-dim features

        # Replace the final fc layer (classifier head after backbone)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_features, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x, return_features=False):
        features = self.backbone(x)   # (B, 512)
        hidden = self.classifier[:-1](features)  # up to 128-dim layer (match BERT)

        if return_features:
            return hidden  # for fusion

        logits = self.classifier(features)
        return logits

