# src/cnn_model.py

import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

resnet = models.resnet18(weights="IMAGENET1K_V1")
class ImageResNet(nn.Module):
    def __init__(self, resnet_model, num_classes=3, dropout=0.5):
        super().__init__()
        self.resnet = resnet_model
        
        # Replace the final fc layer (classifier head)
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_features, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        return self.resnet(x)
