import torch
from torch.utils.data import Dataset, TensorDataset
from torch.utils.data import DataLoader

from src.config import DEVICE

class FusionFeatureDataset(Dataset):
    def __init__(self, bert_loader, cnn_loader, mlp_loader,
                 bert_model, cnn_model, mlp_model, y_label):

        self.features = []
        self.labels = []

        bert_model.eval()
        cnn_model.eval()
        mlp_model.eval()

        with torch.no_grad():
            # ------ TRAIN ---------
            for bert_batch, cnn_batch, mlp_batch in zip(bert_loader, cnn_loader, mlp_loader):

                # BERT features
                input_ids = bert_batch["input_ids"].to(DEVICE)
                attention_mask = bert_batch["attention_mask"].to(DEVICE)

                bert_feat = bert_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_features=True
                )

                # CNN features
                images, _ = cnn_batch          # (X, y)
                images = images.to(DEVICE)

                cnn_feat = cnn_model(images, return_features=True)

                # MLP features
                metadata, _ = mlp_batch        # (X, y)
                metadata = metadata.to(DEVICE)

                mlp_feat = mlp_model(metadata, return_features=True)

                # Concatenate along feature dim
                fusion_feat = torch.cat([bert_feat, cnn_feat, mlp_feat], dim=1)  # (B, total_feat_dim)

                self.features.append(fusion_feat.cpu())

        self.features = torch.cat(self.features, dim=0)
        self.labels = y_label.clone().detach()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# def create_dataloaders(
#     X_train, X_test,
#     y_train, y_test,
#     batch_size,
#     generator,
#     shuffle_train=True
# ):
    # train_ds = TensorDataset(X_train, y_train)
    # test_ds  = TensorDataset(X_test, y_test)
def create_dataloaders(
    bert_train_loader,
    cnn_train_loader,
    mlp_train_loader,
    bert_test_loader,
    cnn_test_loader,
    mlp_test_loader,
    bert_model,
    cnn_model,
    mlp_model,
    batch_size,
    generator,
    shuffle_train=True
):
    train_dataset = FusionFeatureDataset(
        bert_train_loader,
        cnn_train_loader,
        mlp_train_loader,
        bert_model,
        cnn_model,
        mlp_model,
        DEVICE
    )

    test_dataset = FusionFeatureDataset(
        bert_test_loader,
        cnn_test_loader,
        mlp_test_loader,
        bert_model,
        cnn_model,
        mlp_model,
        DEVICE
    )


    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        generator=generator
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, test_loader