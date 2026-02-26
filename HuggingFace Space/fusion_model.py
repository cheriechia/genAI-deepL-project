import torch.nn as nn
import torch

class FusionModel(torch.nn.Module):
    def __init__(self, input_dim, num_classes=3, hidden_dim=256, dropout=0.3):
        super().__init__()
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        return self.classifier(x)

