# src/train.py

import torch
import wandb
from src.evaluate_metrics import evaluate_metrics

def train_epoch(model, loader, optimizer, criterion, device):

    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    for batch in loader:
        # Detect if batch is a dict (BERT-style) or tuple (LSTM/MLP/CNN)
        if isinstance(batch, dict):
            # BERT-style
            X = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            y = batch["labels"].to(device)
            logits = model(X, attention_mask)

        else:
            # Tuple-style: (X, y) for LSTM, MLP, CNN
            X, y = batch
            # Move x y to same device
            X, y = X.to(device), y.to(device)
            # Feeds batch into model, perform forward pass, output logits
            logits = model(X)
            
        # Clear old gradients from previous batch
        optimizer.zero_grad()
        # Compute loss for this batch
        loss = criterion(logits, y)
        # Performs backpropagation: computes gradients of the loss with respect to each model parameter.
        loss.backward()
        # Updates model parameters using those gradients
        optimizer.step()

        # Adds the numeric loss value (converted from a tensor with .item()) to the total loss accumulator.
        total_loss += loss.item()

        preds = logits.argmax(dim=1)
        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(y.detach().cpu().numpy())

    # Average loss = total loss / number of batches
    train_loss = total_loss / len(loader)

    return train_loss, all_preds, all_labels

def eval_epoch(model, loader, criterion, device):
    # Set model to evaluation mode. Disables dropout, freezes batchnorm stats
    model.eval()

    # Initialize accumulator for total loss
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            if isinstance(batch, dict):
                X = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                y = batch["labels"].to(device)
                logits = model(X, attention_mask)

            else:
                X, y = batch
                X, y = X.to(device), y.to(device)    
                logits = model(X)

            loss = criterion(logits, y)
            total_loss += loss.item()

            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    # Compute average loss
    eval_loss = total_loss / len(loader)

    return eval_loss, all_preds, all_labels


def train_model(model,
                train_loader,
                test_loader,
                optimizer,
                criterion,
                device,
                epochs,
                patience):

    best_f1 = 0
    no_improve = 0
    class_names = ["Low", "Medium", "High"]

    for epoch in range(epochs):
        print(epoch)

        train_loss, train_preds, train_labels = train_epoch(
            model, train_loader, optimizer, criterion, device
        )

        test_loss, test_preds, test_labels = eval_epoch(
            model, test_loader, criterion, device
        )

        # Metrics
        train_acc, train_cm, train_macro_f1 = evaluate_metrics(
            train_preds, train_labels
        )

        test_acc, test_cm, test_macro_f1 = evaluate_metrics(
            test_preds, test_labels
        )


        # Early stopping based on test macro-F1
        if test_macro_f1 > best_f1:
            # Update best F1
            best_f1 = test_macro_f1

            # Log only if improve
            wandb.log({
                "epoch": epoch,

                "train/loss": train_loss,
                "train/macro_f1": train_macro_f1,
                "train/accuracy": train_acc,
                "train/confusion_matrix":
                    wandb.plot.confusion_matrix(
                        preds=train_preds,
                        y_true=train_labels,
                        class_names=class_names
                    ),

                "test/loss": test_loss,
                "test/macro_f1": test_macro_f1,
                "test/accuracy": test_acc,
                "test/confusion_matrix":
                    wandb.plot.confusion_matrix(
                        preds=test_preds,
                        y_true=test_labels,
                        class_names=class_names
                    )
            })

            # Reset patience counter
            no_improve = 0
        else:
            no_improve += 1

        print(f"\nEpoch {epoch+1}")
        print(f"Train | loss={train_loss:.4f}, acc={train_acc:.3f}, macro f1={train_macro_f1:.3f}")
        print("Train Confusion Matrix:")
        print(train_cm)
        print(f"Test  | loss={test_loss:.4f}, acc={test_acc:.3f}, macro f1={test_macro_f1:.3f}")
        print("Test Confusion Matrix:")
        print(test_cm)

        if no_improve >= patience:
            print("Early stopping triggered")
            break

    return best_f1
