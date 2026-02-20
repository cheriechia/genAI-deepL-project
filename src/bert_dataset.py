# src/bert_dataset.py

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class TextDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx].clone().detach() for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def create_dataloaders(train_encodings, test_encodings,
                       y_train, y_test,
                       batch_size, generator, shuffle_train=True):
    # Creating dataset objects
    train_ds = TextDataset(train_encodings, y_train)
    test_ds = TextDataset(test_encodings, y_test)
    
    # DataLoader splits data into batches
    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=shuffle_train, generator=generator)
    test_loader = DataLoader(test_ds, batch_size=batch_size,
                             shuffle=False)

    return train_loader, test_loader
