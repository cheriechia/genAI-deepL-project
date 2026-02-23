# src/evaluate_metrics.py

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score


def evaluate_metrics(all_preds, all_labels):

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    accuracy = (all_preds == all_labels).mean()
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    cm = confusion_matrix(all_labels, all_preds)

    return accuracy, cm, macro_f1

