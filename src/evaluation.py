"""
Evaluation metrics and visualization module.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.utils.class_weight import compute_class_weight


def compute_weights(y_encoded):
    """Compute class weights for imbalanced datasets."""
    classes = np.unique(y_encoded)
    weights = compute_class_weight(
        class_weight="balanced", 
        classes=classes, 
        y=y_encoded
    )
    return {int(c): float(w) for c, w in zip(classes, weights)}


def per_class_metrics(cm, class_labels):
    """Compute per-class precision, recall, F1, accuracy from confusion matrix."""
    metrics = {}
    for idx, label in enumerate(class_labels):
        tp = cm[idx, idx]
        support = cm[idx].sum()
        predicted = cm[:, idx].sum()
        
        precision = float(tp / predicted) if predicted else 0.0
        recall = float(tp / support) if support else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        
        metrics[label] = {
            "support": int(support),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "accuracy": recall,
        }
    return metrics


def save_confusion_matrix(cm, class_labels, title, path):
    """Save confusion matrix as PNG image."""
    fig, ax = plt.subplots(figsize=(12, 12))
    im = ax.imshow(cm, cmap='Blues')
    ax.set_title(title, fontsize=14, pad=20)
    ax.set_xlabel("Predicted label", fontsize=12)
    ax.set_ylabel("True label", fontsize=12)
    ax.set_xticks(range(len(class_labels)))
    ax.set_yticks(range(len(class_labels)))
    ax.set_xticklabels(class_labels, rotation=90)
    ax.set_yticklabels(class_labels)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            value = cm[i, j]
            if value > 0:
                ax.text(
                    j, i, str(int(value)),
                    ha="center", va="center",
                    color="white" if value > cm.max() * 0.5 else "black",
                    fontsize=8,
                )

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"    Saved confusion matrix: {path.name}")
