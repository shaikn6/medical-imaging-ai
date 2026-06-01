"""
generate_assets.py — run once to produce all four portfolio PNGs and
train the model.

Usage:
    python generate_assets.py
"""

from __future__ import annotations

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

DOCS_DIR = os.path.join(_HERE, "docs", "screenshots")
os.makedirs(DOCS_DIR, exist_ok=True)

CLASSES = ["Normal", "Pneumonia", "Cardiomegaly", "Pleural Effusion"]

# ============================================================
# 1. Train the model
# ============================================================

def train_model():
    from model.trainer import train
    print("\n" + "=" * 60)
    print("STEP 1 — Training model")
    print("=" * 60)
    history = train()
    return history


# ============================================================
# 2. Synthetic X-ray grid (synthetic_xrays.png)
# ============================================================

def plot_synthetic_xrays():
    from data.synthetic_xray import GENERATORS
    print("\n" + "=" * 60)
    print("STEP 2 — Generating synthetic_xrays.png")
    print("=" * 60)

    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    fig.patch.set_facecolor("white")

    titles = {
        "Normal":           "Normal",
        "Pneumonia":        "Pneumonia\n(consolidation)",
        "Cardiomegaly":     "Cardiomegaly\n(enlarged heart)",
        "Pleural Effusion": "Pleural Effusion\n(fluid level)",
    }

    for col, cls_name in enumerate(CLASSES):
        gen_fn = GENERATORS[cls_name]
        for row in range(2):
            ax = axes[row, col]
            img = gen_fn(224, seed=row * 10 + col)
            ax.imshow(img, cmap="gray", vmin=0, vmax=255)
            ax.axis("off")
            if row == 0:
                ax.set_title(titles[cls_name], fontsize=11, fontweight="bold", pad=8)

    fig.suptitle("Synthetic Chest X-Ray Images (224×224 grayscale)",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    out = os.path.join(DOCS_DIR, "synthetic_xrays.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out}")


# ============================================================
# 3. Training curves (training_curves.png)
# ============================================================

def plot_training_curves():
    print("\n" + "=" * 60)
    print("STEP 3 — Generating training_curves.png")
    print("=" * 60)

    history_path = os.path.join(_HERE, "model", "training_history.json")
    with open(history_path) as f:
        history = json.load(f)

    epochs = range(1, len(history["train_loss"]) + 1)

    dark = "#0e1117"
    surface = "#1e2530"
    grid_col = "#2d3748"
    blue = "#2d8cf0"
    cyan = "#00d4aa"
    orange = "#f6ad55"
    pink = "#fc8181"

    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor(dark)

    for ax in (ax_loss, ax_acc):
        ax.set_facecolor(surface)
        ax.tick_params(colors="#c8d8ea")
        ax.xaxis.label.set_color("#c8d8ea")
        ax.yaxis.label.set_color("#c8d8ea")
        ax.title.set_color("#fff")
        for spine in ax.spines.values():
            spine.set_edgecolor(grid_col)
        ax.grid(True, color=grid_col, linestyle="--", linewidth=0.7, alpha=0.8)

    # Loss
    ax_loss.plot(epochs, history["train_loss"], color=blue,  lw=2.5, label="Train loss", marker="o", markersize=4)
    ax_loss.plot(epochs, history["val_loss"],   color=orange, lw=2.5, label="Val loss",   marker="s", markersize=4)
    ax_loss.set_title("Loss", fontsize=13, fontweight="bold", pad=10)
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Cross-Entropy Loss")
    ax_loss.legend(facecolor=surface, edgecolor=grid_col, labelcolor="#c8d8ea")

    # Accuracy
    ax_acc.plot(epochs, [v * 100 for v in history["train_acc"]], color=cyan, lw=2.5, label="Train acc", marker="o", markersize=4)
    ax_acc.plot(epochs, [v * 100 for v in history["val_acc"]],   color=pink, lw=2.5, label="Val acc",   marker="s", markersize=4)
    ax_acc.axhline(85, color="#ffffff", linestyle=":", lw=1.2, alpha=0.5, label="85% target")
    ax_acc.set_title("Accuracy", fontsize=13, fontweight="bold", pad=10)
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy (%)")
    ax_acc.set_ylim(0, 105)
    ax_acc.legend(facecolor=surface, edgecolor=grid_col, labelcolor="#c8d8ea")

    best_val = max(history["val_acc"]) * 100
    fig.suptitle(f"Training Curves — Best Val Acc: {best_val:.1f}%",
                 color="#fff", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    out = os.path.join(DOCS_DIR, "training_curves.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=dark)
    plt.close(fig)
    print(f"  Saved → {out}")


# ============================================================
# 4. Grad-CAM examples (gradcam_example.png)
# ============================================================

def plot_gradcam_examples():
    from PIL import Image
    from data.synthetic_xray import GENERATORS
    from data.augmentation import get_val_transforms
    from model.cnn_classifier import load_model, CLASSES as MODEL_CLASSES
    from model.gradcam import GradCAM, overlay_heatmap

    print("\n" + "=" * 60)
    print("STEP 4 — Generating gradcam_example.png")
    print("=" * 60)

    ckpt = os.path.join(_HERE, "model", "xray_model.pth")
    model = load_model(ckpt)
    transforms = get_val_transforms()

    dark = "#0e1117"
    surface = "#1e2530"
    label_colors = {
        "Normal":           "#48bb78",
        "Pneumonia":        "#fc8181",
        "Cardiomegaly":     "#f6ad55",
        "Pleural Effusion": "#76e4f7",
    }

    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    fig.patch.set_facecolor(dark)

    for col, cls_name in enumerate(MODEL_CLASSES):
        gen_fn = GENERATORS[cls_name]
        cls_idx = MODEL_CLASSES.index(cls_name)
        arr = gen_fn(224, seed=5 + col)
        pil = Image.fromarray(arr, mode="L")
        tensor = transforms(pil).unsqueeze(0)

        with GradCAM(model) as cam:
            heatmap = cam.compute(tensor, target_class=cls_idx)

        overlay = overlay_heatmap(arr, heatmap)
        lc = label_colors[cls_name]

        # Row 0 — original
        ax0 = axes[0, col]
        ax0.imshow(arr, cmap="gray", vmin=0, vmax=255)
        ax0.axis("off")
        ax0.set_title(f"{cls_name}\n(Original)", fontsize=9, color=lc, fontweight="bold")
        ax0.set_facecolor(surface)

        # Row 1 — overlay
        ax1 = axes[1, col]
        ax1.imshow(overlay)
        ax1.axis("off")
        ax1.set_title("Grad-CAM Overlay", fontsize=9, color=lc)
        ax1.set_facecolor(surface)

    fig.suptitle("Grad-CAM Explainability — Activation Regions per Pathology Class",
                 color="#fff", fontsize=12, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    out = os.path.join(DOCS_DIR, "gradcam_example.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=dark)
    plt.close(fig)
    print(f"  Saved → {out}")


# ============================================================
# 5. ROC curves (roc_curves.png)
# ============================================================

def plot_roc_curves():
    import torch
    from torch.utils.data import DataLoader
    from sklearn.metrics import roc_curve
    from data.synthetic_xray import get_dataset_class
    from data.augmentation import get_val_transforms
    from model.cnn_classifier import load_model

    print("\n" + "=" * 60)
    print("STEP 5 — Generating roc_curves.png")
    print("=" * 60)

    ckpt  = os.path.join(_HERE, "model", "xray_model.pth")
    model = load_model(ckpt)
    data_dir = os.path.join(_HERE, "data", "synthetic_dataset")

    XRayDataset = get_dataset_class()
    val_ds = XRayDataset(data_dir, split="val", transform=get_val_transforms())
    loader = DataLoader(val_ds, batch_size=32, shuffle=False)

    all_labels, all_scores = [], []
    with torch.no_grad():
        for imgs, lbls in loader:
            logits = model(imgs)
            probs  = torch.softmax(logits, dim=-1).numpy()
            all_scores.append(probs)
            all_labels.append(lbls.numpy())

    all_scores = np.concatenate(all_scores, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    dark    = "#0e1117"
    surface = "#1e2530"
    grid_col = "#2d3748"
    class_colors = ["#2d8cf0", "#fc8181", "#f6ad55", "#76e4f7"]

    fig, ax = plt.subplots(figsize=(8, 7))
    fig.patch.set_facecolor(dark)
    ax.set_facecolor(surface)
    ax.tick_params(colors="#c8d8ea")
    ax.xaxis.label.set_color("#c8d8ea")
    ax.yaxis.label.set_color("#c8d8ea")
    ax.title.set_color("#fff")
    for spine in ax.spines.values():
        spine.set_edgecolor(grid_col)
    ax.grid(True, color=grid_col, linestyle="--", linewidth=0.7, alpha=0.8)

    from sklearn.metrics import roc_auc_score

    for k, (cls_name, color) in enumerate(zip(CLASSES, class_colors)):
        binary = (all_labels == k).astype(int)
        scores = all_scores[:, k]
        fpr, tpr, _ = roc_curve(binary, scores)
        try:
            auc = roc_auc_score(binary, scores)
        except ValueError:
            auc = float("nan")
        ax.plot(fpr, tpr, color=color, lw=2.5, label=f"{cls_name} (AUC={auc:.2f})")

    ax.plot([0, 1], [0, 1], color="#555", linestyle="--", lw=1.5, label="Random")
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title("ROC Curves — Per-Class AUC", fontsize=13, fontweight="bold", pad=12)
    ax.legend(facecolor=surface, edgecolor=grid_col, labelcolor="#c8d8ea", fontsize=10)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.05])

    out = os.path.join(DOCS_DIR, "roc_curves.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=dark)
    plt.close(fig)
    print(f"  Saved → {out}")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    train_model()
    plot_synthetic_xrays()
    plot_training_curves()
    plot_gradcam_examples()
    plot_roc_curves()

    print("\n" + "=" * 60)
    print("ALL DONE")
    print("=" * 60)
    print(f"PNGs in: {DOCS_DIR}")
    print(f"Model:   {os.path.join(_HERE, 'model', 'xray_model.pth')}")
