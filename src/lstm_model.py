# src/lstm_model.py

import torch.nn as nn


class CaptionRNN(nn.Module):
    """
    LSTM-based caption classifier with embedding layer.

    Encodes token sequences into a fixed-length representation
    for multi-class prediction, with optional feature extraction.
    """

    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes=3, dropout=0.5):
        super().__init__()

        # Embedding layer for token-based text classification
        # Maps token id to dense vector
        # embed_dim affects how much semantic info each token carries
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        # Single-layer LSTM
        # each timestep is an embedding vector, which has size of embed_dim
        # returns: output, (hidden, cell)
        self.lstm = nn.LSTM(input_size=embed_dim, hidden_size=hidden_dim, batch_first=True)
        # Dropout after LSTM
        self.dropout = nn.Dropout(dropout)
        # Optional hidden layer
        self.fc_hidden = nn.Linear(hidden_dim, hidden_dim // 2)
        self.relu = nn.ReLU()
        self.fc_out = nn.Linear(hidden_dim // 2, num_classes)

    def forward(self, x, return_features=False):
        emb = self.embedding(x)                     # (batch_size, seq_len, embed_dim)
        _, (hidden, _) = self.lstm(emb)             # lstm output: output, (hidden, cell). output: (batch, seq_len, hidden_dim). hidden: (num_layers*num_directions, batch_size, hidden_dim).
        h = hidden.squeeze(0)                       # (batch_size, hidden_dim), can squeeze because num_layers*num_directions = 1
        h = self.dropout(h)                         # apply dropout
        h = self.fc_hidden(h)                       # optional hidden layer
        h = self.relu(h)                            # non-linearity
        h = self.dropout(h)                         # another dropout
        if return_features:
            return x
        out = self.fc_out(h)                        # (batch_size, num_classes), output logits
        return out