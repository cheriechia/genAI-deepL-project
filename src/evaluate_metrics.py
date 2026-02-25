# src/evaluate_metrics.py

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score


def evaluate_metrics(all_preds, all_labels):
    """
    Computes classification evaluation metrics from model predictions.

    Converts prediction and label lists to NumPy arrays and calculates:
    - Accuracy (overall proportion of correct predictions)
    - Macro-averaged F1 score (unweighted mean of per-class F1 scores)
    - Confusion matrix (class-wise prediction breakdown)

    Macro-F1 is used to provide balanced performance evaluation
    across classes, particularly useful for imbalanced datasets.

    Parameters:
    - all_preds: Iterable of predicted class labels
    - all_labels: Iterable of ground-truth class labels

    Returns:
    - accuracy (float)
    - confusion matrix (ndarray)
    - macro_f1 (float)
    """


    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    accuracy = (all_preds == all_labels).mean()
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    cm = confusion_matrix(all_labels, all_preds)

    return accuracy, cm, macro_f1

