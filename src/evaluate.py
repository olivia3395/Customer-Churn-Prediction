"""
Evaluation Utilities
=====================
Computes metrics and saves diagnostic plots (ROC, confusion matrix,
feature importance) to the reports/ directory.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    accuracy_score, classification_report, confusion_matrix,
    roc_curve, average_precision_score,
)

REPORTS_DIR = os.path.join(os.path.dirname(__file__), "..", "reports")


def compute_metrics(y_true, y_pred, y_prob) -> dict:
    """Return a flat dict of all evaluation metrics."""
    return {
        "roc_auc":          round(roc_auc_score(y_true, y_prob), 4),
        "avg_precision":    round(average_precision_score(y_true, y_prob), 4),
        "f1":               round(f1_score(y_true, y_pred), 4),
        "precision":        round(precision_score(y_true, y_pred), 4),
        "recall":           round(recall_score(y_true, y_pred), 4),
        "accuracy":         round(accuracy_score(y_true, y_pred), 4),
    }


def print_report(y_true, y_pred, y_prob):
    metrics = compute_metrics(y_true, y_pred, y_prob)
    print("\n── Evaluation Metrics ───────────────────────────────")
    for k, v in metrics.items():
        print(f"  {k:<20} {v:.4f}")
    print("\n── Classification Report ────────────────────────────")
    print(classification_report(y_true, y_pred, target_names=["Retained", "Churned"]))
    return metrics


def plot_roc_curve(y_true, y_prob, run_name: str = "model"):
    os.makedirs(REPORTS_DIR, exist_ok=True)
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, lw=2, label=f"ROC AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    plt.tight_layout()

    path = os.path.join(REPORTS_DIR, f"roc_curve_{run_name}.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"[PLOT] ROC curve saved → {path}")
    return path


def plot_confusion_matrix(y_true, y_pred, run_name: str = "model"):
    os.makedirs(REPORTS_DIR, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Retained", "Churned"],
        yticklabels=["Retained", "Churned"],
        ax=ax,
    )
    ax.set_ylabel("Actual")
    ax.set_xlabel("Predicted")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()

    path = os.path.join(REPORTS_DIR, f"confusion_matrix_{run_name}.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"[PLOT] Confusion matrix saved → {path}")
    return path


def plot_feature_importance(feature_names: list, importances: np.ndarray,
                             run_name: str = "model", top_n: int = 20):
    os.makedirs(REPORTS_DIR, exist_ok=True)

    idx = np.argsort(importances)[-top_n:]
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.barh(
        [feature_names[i] for i in idx],
        importances[idx],
        color="#4A90D9",
    )
    ax.set_xlabel("Importance")
    ax.set_title(f"Top {top_n} Feature Importances")
    plt.tight_layout()

    path = os.path.join(REPORTS_DIR, f"feature_importance_{run_name}.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"[PLOT] Feature importance saved → {path}")
    return path
