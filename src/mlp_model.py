# src/mlp_model.py

import torch.nn as nn

class MetadataMLP(nn.Module):
    """
    Two-hidden-layer MLP for structured metadata classification.

    Applies fully connected layers with ReLU and dropout,
    optionally returning intermediate features for fusion.
    """

    def __init__(self, 
                 input_dim, 
                 hidden_dim1=64, 
                 hidden_dim2=32, 
                 dropout=0.3): # input_dim is number of processed features
        super().__init__()          # call constructor of nn.Module

        self.fc1 = nn.Linear(input_dim, hidden_dim1)    # Fully connected layer (40 features -> 64 hidden units)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)  # 2nd hidden layer, compresses information
        self.fc_out = nn.Linear(hidden_dim2, 3)         # output layer, 3 engagement classes, outputs logits.
        self.dropout = nn.Dropout(dropout)              # drop neurons to prevent overfitting
        self.relu = nn.ReLU()                           # ReLU for non-linearity to learn complex patterns
    
    def forward(self, x, return_features=False): # forward pass

        x = self.relu(self.fc1(x))
        x = self.dropout(x)

        x = self.relu(self.fc2(x))
        x = self.dropout(x)

        if return_features:
            return x    # 2nd hidden layer output (hidden_dim2)

        logits = self.fc_out(x)
        return logits
