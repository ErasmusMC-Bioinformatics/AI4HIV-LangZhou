#!/usr/bin/env python3
"""
Evaluate text classification experiments:
  1) Print sklearn classification report per model
  2) Print specificity (TNR = TN / (TN + FP)) per model
  3) Plot all confusion matrices in a 2x3 grid with labeled axes

Expected structure:
  ROOT_PATH/
    labels.csv   ← with column 'flag'
    modelA/
      predictions.csv ← with column 'prediction'
    modelB/
      predictions.csv
    ...
"""

import os
import math
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# --- Config ---
ROOT_PATH = "../exps/"  # change to your actual root path
LABEL_FILE = os.path.join(ROOT_PATH, "labels.csv")
GRID_SHAPE = (2, 3)  # rows, cols in confusion matrix plot grid

# --- Load ground truth ---
labels_df = pd.read_csv(LABEL_FILE)
y_true = labels_df["flag"].astype(int).to_numpy()

# --- Helper for specificity ---
def compute_specificity(y_true, y_pred):
    # Force binary confusion matrix with label order [0, 1]
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    if cm.shape != (2, 2):
        # Handle degenerate cases (e.g., only one class present)
        # Fall back to zeros safely
        tn = cm[0, 0] if cm.shape[0] > 0 and cm.shape[1] > 0 else 0
        fp = cm[0, 1] if cm.shape[0] > 0 and cm.shape[1] > 1 else 0
    else:
        tn, fp = cm[0, 0], cm[0, 1]
    denom = (tn + fp)
    return (tn / denom) if denom > 0 else float("nan"), cm

# --- Collect results for plotting ---
model_results = []

for subfolder in sorted(os.listdir(ROOT_PATH)):
    folder_path = os.path.join(ROOT_PATH, subfolder)
    pred_file = os.path.join(folder_path, "predictions.csv")

    if not os.path.isdir(folder_path) or not os.path.exists(pred_file):
        continue

    preds_df = pd.read_csv(pred_file)
    y_pred = preds_df["prediction"].astype(int).to_numpy()

    # Align lengths if needed (optional safety)
    n = min(len(y_true), len(y_pred))
    y_pred = y_pred[:n]
    y_t = y_true[:n]

    # Print report
    print(f"\n=== Model: {subfolder} ===")
    print(classification_report(y_t, y_pred, digits=4))

    # Compute and print specificity (TNR for class 1 as positive => TN over negatives)
    specificity, cm_for_print = compute_specificity(y_t, y_pred)
    print(f"Specificity (TNR, treating '1' as positive): {specificity:.4f}")
    # Optional: show raw confusion matrix counts for clarity
    tn, fp, fn, tp = cm_for_print.ravel() if cm_for_print.size == 4 else (float('nan'),)*4
    print(f"Confusion Matrix counts: TN={tn}, FP={fp}, FN={fn}, TP={tp}")

    model_results.append((subfolder, y_t, y_pred))

# --- Plot confusion matrices ---
num_models = len(model_results)
if num_models == 0:
    print("\nNo models found with predictions.csv; nothing to plot.")
else:
    rows, cols = GRID_SHAPE
    total_cells = rows * cols

    # If more models than cells, create enough rows automatically (keeps 3 columns)
    if num_models > total_cells:
        cols = 3
        rows = math.ceil(num_models / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    # axes could be a single Axes if rows*cols == 1
    if isinstance(axes, plt.Axes):
        axes = [axes]
    else:
        axes = axes.flatten()

    for i, (model_name, y_t, y_p) in enumerate(model_results):
        ax = axes[i]
        cm = confusion_matrix(y_t, y_p, labels=[0, 1])
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=["Exclusion (0)", "Inclusion (1)"]
        )
        disp.plot(ax=ax, colorbar=False)
        ax.set_title(model_name)
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")

    # Hide any unused subplots
    for j in range(len(model_results), len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()
