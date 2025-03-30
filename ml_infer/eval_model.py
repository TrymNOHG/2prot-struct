"""
Everything related to evaluating the performance of a model:
    - Metrics
    - Graphs and plots
    - Etc.

For input we exepct predictions and correct labels

Done:
    - Precision, Recall, F1
    - Confusion Matrix
    
TODO:
    - y_proba from models ????
    - Cross-entropy loss
    - AUC ROC


Also:
    When using a windowed model, maybe we can construct some intereting ad-hoc metrics?
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    precision_score, recall_score, f1_score, confusion_matrix,
    log_loss, roc_auc_score, precision_recall_curve, roc_curve, auc
)

# amino_acids = "ACDEFGHIKLMNPQRSTVWY"
secondary_structures = "HECTGSPIB"


# TODO: y_probab
def evaluate_classification(y_true, y_pred, y_pred_proba=None):
    # Returns a dict containing the various metrics and figures
    # Convert character labels to integers if needed
    if isinstance(y_true[0], str):
        label_map = {c: i for i, c in enumerate(secondary_structures)}
        y_true = np.array([label_map[c] for c in y_true])
        y_pred = np.array([label_map[c] for c in y_pred])
    
    results = {}
    
    results["accuracy"] = np.mean(y_true == y_pred)
    
    # Macro
    results["precision_macro"] = precision_score(y_true, y_pred, average="macro")
    results["recall_macro"] = recall_score(y_true, y_pred, average="macro")
    results["f1_macro"] = f1_score(y_true, y_pred, average="macro")
    # Weighted
    results["precision_weighted"] = precision_score(y_true, y_pred, average="weighted")
    results["recall_weighted"] = recall_score(y_true, y_pred, average="weighted")
    results["f1_weighted"] = f1_score(y_true, y_pred, average="weighted")
    # Per-class metrics
    results["precision_per_class"] = precision_score(y_true, y_pred, labels=range(len(secondary_structures)), average=None)
    results["recall_per_class"] = recall_score(y_true, y_pred, labels=range(len(secondary_structures)), average=None)
    results["f1_per_class"] = f1_score(y_true, y_pred, labels=range(len(secondary_structures)), average=None)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=range(len(secondary_structures)))
    results["confusion_matrix"] = cm
    
    results["class_distribution"] = np.bincount(y_true, minlength=len(secondary_structures))
    results["pred_distribution"] = np.bincount(y_pred, minlength=len(secondary_structures))
    
    fig_cm = plot_confusion_matrix(cm, secondary_structures)
    results["confusion_matrix_plot"] = fig_cm

    return results


def evaluation_summary(results):
    plot_confusion_matrix(results["confusion_matrix"], secondary_structures)
    # TODO: print all other metrics
    for k, v in results.items():
        print(k, v)


def plot_confusion_matrix(cm, class_labels):
    plt.figure(figsize=(10, 8))
    
    # Normalize
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.nan_to_num(cm_norm)  # Replace NaN with 0
    
    # Create heatmap
    sns.heatmap(
        cm_norm, 
        annot=cm,
        fmt="d",
        cmap="Blues",
        xticklabels=class_labels,
        yticklabels=class_labels
    )
    
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.title("Confusion Matrix")
    
    fig = plt.gcf()
    plt.savefig("confusion_matrix.png")
    plt.close() 
    return fig
