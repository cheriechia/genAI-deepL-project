# src/cnn_dataset.py

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image
from torchvision import transforms

class ImageDataset(Dataset):
    def __init__(self, df, img_col="image_path", target_col="engagement_label", transform=None):
        self.df = df.reset_index(drop=True)
        self.img_col = img_col
        self.target_col = target_col
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # print("Loading image:", self.df.iloc[idx]['image_path'])
        img_path = self.df.loc[idx, self.img_col]
        label = self.df.loc[idx, self.target_col]

        # Load image
        img = Image.open(img_path).convert("RGB")  # make sure it's RGB
        
        if self.transform:
            img = self.transform(img)
        
        return img, label


def create_dataloaders(train_encodings, test_encodings,
                       train_transform, test_transform,
                       batch_size, generator):
    # Creating dataset objects
    train_ds = ImageDataset(train_encodings, transform=train_transform)
    test_ds = ImageDataset(test_encodings, transform=test_transform)
    
    # DataLoader splits data into batches
    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True, generator=generator)
    test_loader = DataLoader(test_ds, batch_size=batch_size,
                             shuffle=False)

    return train_loader, test_loader
