"""
ScoreCAM — Gradient-Free Saliency Maps (Wang et al. 2020).

ScoreCAM produces sharper, more faithful activation maps than Grad-CAM
because it does not rely on backpropagation.  Instead, each feature map
from the target conv layer is upsampled, used as a binary mask over the
input, and the change in class score is measured in a pure forward pass.
That score becomes the importance weight for the feature map channel.

Reference: https://arxiv.org/abs/1910.01279

Usage::

    from src.scorecam import generate_scorecam, save_scorecam_overlay

    heatmap = generate_scorecam(model, image_tensor, target_class=1)
    save_scorecam_overlay(heatmap, original_image_np, "output.png")
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Core ScoreCAM function
# ---------------------------------------------------------------------------

def generate_scorecam(
    model: nn.Module,
    image_tensor: torch.Tensor,
    target_class: Optional[int] = None,
    target_layer: Optional[nn.Module] = None,
    batch_size: int = 16,
) -> np.ndarray:
    """
    Compute a ScoreCAM heatmap for the given image tensor.

    This is a pure forward-pass implementation — no backward pass required.

    Parameters
    ----------
    model : nn.Module
        Trained model.  Must expose ``last_conv_layer`` property (XRayCNN)
        or accept an explicit ``target_layer``.
    image_tensor : torch.Tensor
        Shape (1, C, H, W).  The batch dimension must be present.
    target_class : int | None
        Class index to explain.  Defaults to the model's predicted class.
    target_layer : nn.Module | None
        Conv layer whose feature maps are used.  Defaults to
        ``model.last_conv_layer``.
    batch_size : int
        Number of masked images to score in a single forward pass.
        Trade-off: larger = faster, but more memory.

    Returns
    -------
    np.ndarray
        Float32 heatmap of shape (H, W) with values in [0, 1].
    """
    model.eval()

    if image_tensor.dim() != 4 or image_tensor.shape[0] != 1:
        raise ValueError(
            f"image_tensor must have shape (1, C, H, W); got {tuple(image_tensor.shape)}"
        )

    device = next(model.parameters()).device
    image_tensor = image_tensor.to(device)

    # Resolve target layer
    layer = _resolve_target_layer(model, target_layer)

    # Step 1: single forward pass to capture feature maps + determine class
    feature_maps = _capture_feature_maps(model, layer, image_tensor)  # (1, K, h, w)

    if target_class is None:
        with torch.no_grad():
            logits = model(image_tensor)
        target_class = int(logits.argmax(dim=-1).item())

    feature_maps.shape[1]
    h_in, w_in = image_tensor.shape[2], image_tensor.shape[3]

    # Step 2: upsample each feature map to input size, normalise → mask
    upsampled = F.interpolate(
        feature_maps,
        size=(h_in, w_in),
        mode="bilinear",
        align_corners=False,
    )  # (1, K, H, W)

    # Normalise each channel map to [0, 1] independently
    masks = _normalise_maps(upsampled.squeeze(0))  # (K, H, W)

    # Step 3: for each mask, compute masked input score for target class
    baseline = _get_baseline(image_tensor)  # zeros baseline (black image)
    scores = _score_masked_inputs(
        model, image_tensor, baseline, masks, target_class, batch_size
    )  # (K,)

    # Step 4: softmax-weighted sum of original feature maps
    weights = torch.softmax(scores, dim=0)  # (K,)
    cam = (weights.view(-1, 1, 1) * upsampled.squeeze(0)).sum(dim=0)  # (H, W)
    cam = F.relu(cam)

    cam_np = cam.detach().cpu().numpy().astype(np.float32)
    cam_np = _normalise_array(cam_np)
    return cam_np


# ---------------------------------------------------------------------------
# Overlay / visualisation
# ---------------------------------------------------------------------------

def save_scorecam_overlay(
    heatmap: np.ndarray,
    original: np.ndarray,
    output_path: str,
    alpha: float = 0.45,
    colormap: str = "jet",
) -> None:
    """
    Blend ScoreCAM heatmap over the original image and save as PNG.

    Parameters
    ----------
    heatmap : np.ndarray
        Float32 (H, W) in [0, 1] from ``generate_scorecam``.
    original : np.ndarray
        Grayscale (H, W) or RGB (H, W, 3) uint8 array.
    output_path : str
        Destination file path (e.g. "output.png").
    alpha : float
        Heatmap opacity in the blend.
    colormap : str
        Matplotlib colormap name (default "jet").
    """
    import matplotlib
    import matplotlib.pyplot as plt

    colormap_fn = matplotlib.colormaps[colormap]
    heat_rgb = (colormap_fn(heatmap)[:, :, :3] * 255).astype(np.uint8)

    # Convert original to RGB
    if original.ndim == 2 or (original.ndim == 3 and original.shape[2] == 1):
        gray = original.squeeze().astype(np.float32)
        orig_rgb = np.stack([gray, gray, gray], axis=-1)
    else:
        orig_rgb = original.astype(np.float32)

    # Resize heatmap if spatial dims differ
    if heat_rgb.shape[:2] != orig_rgb.shape[:2]:
        from PIL import Image as PILImage
        heat_pil = PILImage.fromarray(heat_rgb).resize(
            (orig_rgb.shape[1], orig_rgb.shape[0]), PILImage.BILINEAR
        )
        heat_rgb = np.array(heat_pil).astype(np.float32)
    else:
        heat_rgb = heat_rgb.astype(np.float32)

    blended = np.clip(
        (1 - alpha) * orig_rgb + alpha * heat_rgb, 0, 255
    ).astype(np.uint8)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(orig_rgb.astype(np.uint8), cmap="gray" if orig_rgb.shape[2] == 3 else None)
    axes[0].set_title("Original")
    axes[0].axis("off")

    im = axes[1].imshow(heatmap, cmap=colormap, vmin=0, vmax=1)
    axes[1].set_title("ScoreCAM")
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    axes[2].imshow(blended)
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def heatmap_to_overlay(
    heatmap: np.ndarray,
    original: np.ndarray,
    alpha: float = 0.45,
    colormap: str = "jet",
) -> np.ndarray:
    """
    Return a blended uint8 RGB (H, W, 3) array without saving to disk.

    Useful for embedding in larger figures or Streamlit dashboards.
    """
    import matplotlib

    colormap_fn = matplotlib.colormaps[colormap]
    heat_rgb = (colormap_fn(heatmap)[:, :, :3] * 255).astype(np.float32)

    if original.ndim == 2 or (original.ndim == 3 and original.shape[2] == 1):
        gray = original.squeeze().astype(np.float32)
        orig_rgb = np.stack([gray, gray, gray], axis=-1).astype(np.float32)
    else:
        orig_rgb = original.astype(np.float32)

    blended = np.clip((1 - alpha) * orig_rgb + alpha * heat_rgb, 0, 255).astype(np.uint8)
    return blended


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _resolve_target_layer(
    model: nn.Module,
    target_layer: Optional[nn.Module],
) -> nn.Module:
    """Return target layer, defaulting to model.last_conv_layer."""
    if target_layer is not None:
        return target_layer
    if hasattr(model, "last_conv_layer"):
        return model.last_conv_layer  # type: ignore[attr-defined]
    # Fallback: find last Conv2d in the model
    last_conv = None
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            last_conv = module
    if last_conv is None:
        raise ValueError(
            "Cannot determine target layer: model has no Conv2d layers "
            "and no 'last_conv_layer' property."
        )
    return last_conv


def _capture_feature_maps(
    model: nn.Module,
    layer: nn.Module,
    image_tensor: torch.Tensor,
) -> torch.Tensor:
    """Run a forward pass with a hook to capture the layer's output."""
    captured: list[torch.Tensor] = []

    def _hook(_module: nn.Module, _input: tuple, output: torch.Tensor) -> None:
        captured.append(output.detach())

    handle = layer.register_forward_hook(_hook)
    try:
        with torch.no_grad():
            model(image_tensor)
    finally:
        handle.remove()

    if not captured:
        raise RuntimeError("Forward hook did not fire — check target layer.")
    return captured[0]


def _normalise_maps(maps: torch.Tensor) -> torch.Tensor:
    """
    Normalise each channel of (K, H, W) to [0, 1] independently.
    Channels with zero range are left as zeros.
    """
    k = maps.shape[0]
    flat = maps.view(k, -1)  # (K, H*W)
    mins = flat.min(dim=1).values.view(k, 1, 1)
    maxs = flat.max(dim=1).values.view(k, 1, 1)
    ranges = maxs - mins
    ranges = torch.where(ranges < 1e-8, torch.ones_like(ranges), ranges)
    return (maps - mins) / ranges  # (K, H, W)


def _get_baseline(image_tensor: torch.Tensor) -> torch.Tensor:
    """Return a zero-valued baseline tensor with the same shape."""
    return torch.zeros_like(image_tensor)


def _score_masked_inputs(
    model: nn.Module,
    image_tensor: torch.Tensor,
    baseline: torch.Tensor,
    masks: torch.Tensor,
    target_class: int,
    batch_size: int,
) -> torch.Tensor:
    """
    For each mask channel, compute the model score for target_class.

    Returns a (K,) tensor of raw logit scores.
    """
    num_channels = masks.shape[0]
    scores = torch.zeros(num_channels, device=image_tensor.device)

    for start in range(0, num_channels, batch_size):
        end = min(start + batch_size, num_channels)
        batch_masks = masks[start:end]  # (B, H, W)
        B = batch_masks.shape[0]

        # Expand image and baseline to batch size
        img_batch = image_tensor.expand(B, -1, -1, -1)  # (B, C, H, W)
        base_batch = baseline.expand(B, -1, -1, -1)     # (B, C, H, W)
        mask_4d = batch_masks.unsqueeze(1)               # (B, 1, H, W)

        masked = img_batch * mask_4d + base_batch * (1.0 - mask_4d)

        with torch.no_grad():
            logits = model(masked)  # (B, num_classes)

        scores[start:end] = logits[:, target_class]

    return scores


def _normalise_array(arr: np.ndarray) -> np.ndarray:
    """Normalise a float32 numpy array to [0, 1]."""
    arr_min, arr_max = arr.min(), arr.max()
    if arr_max - arr_min < 1e-8:
        return np.zeros_like(arr)
    return (arr - arr_min) / (arr_max - arr_min)
