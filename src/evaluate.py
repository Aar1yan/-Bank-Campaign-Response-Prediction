"""Reusable evaluation logic - the standard metric set used throughout this project.

The full evaluation (curves, confusion matrices, threshold sweep, business
impact analysis) is one-off exploratory work and stays in notebook 07. This
only keeps what's genuinely useful to call again later, e.g. from a test.
"""

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_predictions(y_true, y_proba, threshold):
    """Computes accuracy/precision/recall/F1 at the given threshold, plus ROC-AUC and PR-AUC."""
    y_pred = (y_proba >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "pr_auc": average_precision_score(y_true, y_proba),
    }
