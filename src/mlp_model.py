# src/mlp_model.py

import torch.nn as nn

class MetadataMLP(nn.Module):
    def __init__(self, 
                 input_dim, 
                 hidden_dim1=64, 
                 hidden_dim2=32, 
                 dropout=0.3): # input_dim is number of processed features
        super().__init__()          # call constructor of nn.Module
        self.net = nn.Sequential(   # stacks layers in order
            nn.Linear(input_dim, hidden_dim1),   # Fully connected layer (40 features -> 64 hidden units)
            nn.ReLU(),                  # ReLU for non-linearity to learn complex patterns
            nn.Dropout(dropout),            # randomly drop 30% of neurons to prevent overfitting
            nn.Linear(hidden_dim1, hidden_dim2),          # 2nd hidden layer, compresses information
            nn.ReLU(),
            nn.Linear(hidden_dim2, 3)            # output layer, 3 engagement classes, outputs logits.
        )

    def forward(self, x): # forward pass
        return self.net(x)