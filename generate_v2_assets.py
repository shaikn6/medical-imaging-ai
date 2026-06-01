"""
Generate V2 documentation screenshots and assets.

Produces:
  docs/screenshots/unet_segmentation.png
  docs/screenshots/multilabel_predictions.png
  docs/screenshots/gradcam_scorecam_comparison.png
  docs/screenshots/dicom_windowing.png

Run:
    python generate_v2_assets.py
"""

from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch

_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)

SCREENSHOTS_DIR = os.path.join(_ROOT, "docs", "screenshots")
os.makedirs(SCREENSHOTS_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tensor_to_display(tensor: torch.Tensor) -> np.ndarray:
    """(1, 1, H, W) normalised tensor → uint8 (H, W) for imshow."""
    arr = tensor.squeeze().cpu().numpy()
    arr = ((arr + 1.0) / 2.0 * 255.0).clip(0, 255).astype(np.uint8)
    return arr


# ---------------------------------------------------------------------------
# 1. U-Net segmentation
# ---------------------------------------------------------------------------

def generate_unet_screenshot():
    from models.unet import build_unet, SyntheticSegDataset

    print("  Generating U-Net segmentation screenshot …")
    ds = SyntheticSegDataset(n_samples=4, seed=7)
    model = build_unet()
    model.eval()

    fig, axes = plt.subplots(3, 4, figsize=(14, 9))
    fig.suptitle("U-Net Segmentation — Synthetic Chest X-Ray Lesion Masks", fontsize=14, fontweight="bold")

    for i in range(4):
        img_t, mask_t = ds[i]
        with torch.no_grad():
            logit = model(img_t.unsqueeze(0))
            pred_mask = torch.sigmoid(logit).squeeze().cpu().numpy()

        img_disp = img_t.squeeze().cpu().numpy()
        img_disp = ((img_disp + 1.0) / 2.0 * 255).clip(0, 255).astype(np.uint8)
        gt_mask   = mask_t.squeeze().cpu().numpy()

        axes[0, i].imshow(img_disp, cmap="gray")
        axes[0, i].set_title(f"Input Slice {i+1}")
        axes[0, i].axis("off")

        axes[1, i].imshow(gt_mask, cmap="hot", vmin=0, vmax=1)
        axes[1, i].set_title("GT Mask")
        axes[1, i].axis("off")

        axes[2, i].imshow(pred_mask, cmap="hot", vmin=0, vmax=1)
        axes[2, i].set_title(f"Predicted (Dice)")
        axes[2, i].axis("off")

    plt.tight_layout()
    out = os.path.join(SCREENSHOTS_DIR, "unet_segmentation.png")
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {out}")


# ---------------------------------------------------------------------------
# 2. Multi-label predictions
# ---------------------------------------------------------------------------

def generate_multilabel_screenshot():
    from models.multilabel_classifier import (
        build_multilabel_model,
        MULTILABEL_CLASSES,
        SyntheticMultiLabelDataset,
    )

    print("  Generating multi-label classifier screenshot …")
    ds = SyntheticMultiLabelDataset(n_samples=4, seed=42)
    model = build_multilabel_model()
    model.eval()

    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    fig.suptitle("Multi-Label Classifier — 5 Simultaneous Pathology Predictions", fontsize=13, fontweight="bold")

    colors = ["#f85149", "#f0883e", "#d2a8ff", "#56d364", "#58a6ff"]

    for i in range(4):
        img_t, label_t = ds[i]
        with torch.no_grad():
            probs = torch.sigmoid(model(img_t.unsqueeze(0))).squeeze().numpy()

        img_disp = img_t.squeeze().numpy()
        img_disp = ((img_disp + 1.0) / 2.0 * 255).clip(0, 255).astype(np.uint8)

        axes[0, i].imshow(img_disp, cmap="gray")
        gt = [MULTILABEL_CLASSES[j] for j in range(5) if label_t[j] > 0]
        axes[0, i].set_title(f"GT: {', '.join(gt)}", fontsize=8)
        axes[0, i].axis("off")

        bars = axes[1, i].barh(MULTILABEL_CLASSES, probs * 100, color=colors)
        axes[1, i].set_xlim(0, 100)
        axes[1, i].axvline(50, color="white", linestyle="--", alpha=0.5, linewidth=1)
        axes[1, i].set_xlabel("Probability (%)", fontsize=8)
        axes[1, i].tick_params(labelsize=7)
        axes[1, i].set_facecolor("#1e1e2e")
        axes[1, i].spines[:].set_color("#444")

    fig.patch.set_facecolor("#0d1117")
    for ax in axes.flatten():
        ax.tick_params(colors="#c9d1d9")
        ax.xaxis.label.set_color("#c9d1d9")

    plt.tight_layout()
    out = os.path.join(SCREENSHOTS_DIR, "multilabel_predictions.png")
    plt.savefig(out, dpi=120, bbox_inches="tight", facecolor="#0d1117")
    plt.close()
    print(f"    Saved: {out}")


# ---------------------------------------------------------------------------
# 3. Grad-CAM / ScoreCAM comparison
# ---------------------------------------------------------------------------

def generate_gradcam_scorecam_screenshot():
    from model.cnn_classifier import build_model
    from model.gradcam import GradCAM, overlay_heatmap
    from explainability.gradcam_video import ScoreCAM

    print("  Generating Grad-CAM / ScoreCAM comparison screenshot …")

    cls_model = build_model()
    cls_model.eval()

    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    fig.suptitle("Grad-CAM vs ScoreCAM — Attention Comparison Across Pathology Classes", fontsize=12, fontweight="bold")

    from data.synthetic_xray import GENERATORS, CLASSES as V1_CLASSES
    gen_fns = list(GENERATORS.values())

    for i in range(4):
        arr = gen_fns[i](224, seed=i * 7)
        img_t = torch.tensor(arr.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0)
        img_t = (img_t - 0.5) / 0.5

        with GradCAM(cls_model) as gcam:
            gcam_map = gcam.compute(img_t, target_class=i)
        with ScoreCAM(cls_model, batch_size=8) as scam:
            scam_map = scam.compute(img_t, target_class=i)

        gcam_overlay = overlay_heatmap(arr, gcam_map)
        scam_overlay = overlay_heatmap(arr, scam_map)

        axes[0, i].imshow(gcam_overlay)
        axes[0, i].set_title(f"Grad-CAM: {V1_CLASSES[i]}", fontsize=9)
        axes[0, i].axis("off")

        axes[1, i].imshow(scam_overlay)
        axes[1, i].set_title(f"ScoreCAM: {V1_CLASSES[i]}", fontsize=9)
        axes[1, i].axis("off")

    fig.patch.set_facecolor("#0d1117")
    plt.tight_layout()
    out = os.path.join(SCREENSHOTS_DIR, "gradcam_scorecam_comparison.png")
    plt.savefig(out, dpi=120, bbox_inches="tight", facecolor="#0d1117")
    plt.close()
    print(f"    Saved: {out}")


# ---------------------------------------------------------------------------
# 4. DICOM windowing
# ---------------------------------------------------------------------------

def generate_dicom_windowing_screenshot():
    from preprocessing.dicom_handler import (
        generate_synthetic_dicom,
        apply_hu_windowing,
        apply_windowing,
        preprocess_dicom,
        _CT_WINDOW_PRESETS,
    )

    print("  Generating DICOM windowing screenshot …")

    ds_ct = generate_synthetic_dicom(modality="CT", size=256, seed=5)
    ds_cr = generate_synthetic_dicom(modality="CR", size=256, seed=5)

    fig = plt.figure(figsize=(16, 8))
    fig.suptitle("DICOM Windowing — CT Presets and X-Ray Normalisation", fontsize=13, fontweight="bold")
    gs = gridspec.GridSpec(2, 5, figure=fig, hspace=0.4, wspace=0.3)

    # CT presets — top row
    presets = list(_CT_WINDOW_PRESETS.keys())
    hu = ds_ct.hu_array
    for j, preset in enumerate(presets[:4]):
        windowed = apply_hu_windowing(hu, preset=preset)
        ax = fig.add_subplot(gs[0, j])
        ax.imshow(windowed, cmap="gray")
        wc, ww = _CT_WINDOW_PRESETS[preset]
        ax.set_title(f"CT: {preset}\nWC={wc:.0f} WW={ww:.0f}", fontsize=8)
        ax.axis("off")

    # Model input — CT
    model_inp = preprocess_dicom(ds_ct).squeeze()
    ax = fig.add_subplot(gs[0, 4])
    ax.imshow(((model_inp + 1) / 2 * 255).clip(0, 255).astype(np.uint8), cmap="gray")
    ax.set_title("CT → Model Input\n[-1, 1] normalised", fontsize=8)
    ax.axis("off")

    # X-ray: raw, windowed, model input — bottom row
    ax_raw = fig.add_subplot(gs[1, 0])
    ax_raw.imshow(ds_cr.pixel_array, cmap="gray")
    ax_raw.set_title("X-Ray: Raw pixels", fontsize=8)
    ax_raw.axis("off")

    windowed_xr = apply_windowing(ds_cr.pixel_array, ds_cr.WindowCenter, ds_cr.WindowWidth)
    ax_wind = fig.add_subplot(gs[1, 1])
    ax_wind.imshow(windowed_xr, cmap="gray")
    ax_wind.set_title(f"After windowing\nWC={ds_cr.WindowCenter:.0f}", fontsize=8)
    ax_wind.axis("off")

    model_inp_cr = preprocess_dicom(ds_cr).squeeze()
    ax_model = fig.add_subplot(gs[1, 2])
    ax_model.imshow(((model_inp_cr + 1) / 2 * 255).clip(0, 255).astype(np.uint8), cmap="gray")
    ax_model.set_title("X-Ray → Model Input\n[-1, 1] normalised", fontsize=8)
    ax_model.axis("off")

    # Metadata text
    ax_meta = fig.add_subplot(gs[1, 3:])
    meta_str = (
        f"Patient ID: {ds_ct.PatientID}\n"
        f"Study Date: {ds_ct.StudyDate}\n"
        f"Modality:   {ds_ct.Modality}\n"
        f"Rows×Cols:  {ds_ct.Rows}×{ds_ct.Columns}\n"
        f"Bits Alloc: {ds_ct.BitsAllocated}\n"
        f"Pix Spacing:{ds_ct.PixelSpacing[0]:.3f} mm\n"
        f"Rescale Int:{ds_ct.RescaleIntercept}\n"
        f"HU range:   [{float(hu.min()):.0f}, {float(hu.max()):.0f}]"
    )
    ax_meta.text(
        0.05, 0.95, meta_str,
        transform=ax_meta.transAxes,
        fontsize=9, verticalalignment="top",
        fontfamily="monospace",
        color="#c9d1d9",
        bbox=dict(boxstyle="round", facecolor="#161b22", alpha=0.8),
    )
    ax_meta.set_title("DICOM Metadata", fontsize=9)
    ax_meta.axis("off")

    fig.patch.set_facecolor("#0d1117")
    plt.tight_layout()
    out = os.path.join(SCREENSHOTS_DIR, "dicom_windowing.png")
    plt.savefig(out, dpi=120, bbox_inches="tight", facecolor="#0d1117")
    plt.close()
    print(f"    Saved: {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Generating V2 documentation assets …")
    generate_unet_screenshot()
    generate_multilabel_screenshot()
    generate_gradcam_scorecam_screenshot()
    generate_dicom_windowing_screenshot()
    print("All V2 assets generated.")
