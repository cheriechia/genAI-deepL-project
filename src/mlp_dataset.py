# src/dataset.py

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class MetadataDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y) # num of samples in dataset

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx] # return sample at index


def create_dataloaders(train_encodings, test_encodings,
                       y_train, y_test,
                       batch_size, generator, shuffle_train=True):
    # Creating dataset objects
    train_ds = MetadataDataset(train_encodings, y_train)
    test_ds = MetadataDataset(test_encodings, y_test)

    # DataLoader splits data into batches
    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=shuffle_train, generator=generator)
    test_loader = DataLoader(test_ds, batch_size=batch_size,
                             shuffle=False)

    return train_loader, test_loader
