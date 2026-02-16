# src/bert_dataset.py

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class CaptionDataset(Dataset):
    def __init__(self, sequences, labels):
        self.X = torch.tensor(sequences, dtype=torch.long)
        self.y = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def create_dataloaders(train_encodings, test_encodings,
                       y_train, y_test,
                       batch_size, generator):
    # Creating dataset objects
    train_ds = CaptionDataset(train_encodings, y_train)
    test_ds = CaptionDataset(test_encodings, y_test)
    
    # DataLoader splits data into batches
    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True, generator=generator)
    test_loader = DataLoader(test_ds, batch_size=batch_size,
                             shuffle=False)

    return train_loader, test_loader
