"""
Grad-CAM implementation for XRayCNN.

Algorithm (Selvaraju et al. 2017):
  1. Forward pass — capture feature maps from target Conv2d layer.
  2. Backward pass on the target class score — capture gradients at same layer.
  3. Global-average-pool the gradients → importance weights α_k.
  4. Weighted sum:  L = ReLU( Σ_k α_k · A_k )
  5. Bilinear upsample L to input spatial size (224×224).

Usage::

    from model.cnn_classifier import load_model
    from model.gradcam import GradCAM, overlay_heatmap

    model = load_model("model/xray_model.pth")
    cam = GradCAM(model)
    heatmap = cam.compute(image_tensor, target_class=1)
    blended  = overlay_heatmap(original_pil, heatmap)
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


class GradCAM:
    """
    Grad-CAM context manager / callable.

    Parameters
    ----------
    model : nn.Module
        Must expose a ``last_conv_layer`` property (see XRayCNN).
    target_layer : nn.Module | None
        Override the default last-conv layer.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module | None = None):
        self.model = model
        self.model.eval()

        if target_layer is None:
            self.target_layer = model.last_conv_layer  # type: ignore[attr-defined]
        else:
            self.target_layer = target_layer

        self._feature_maps: torch.Tensor | None = None
        self._gradients: torch.Tensor | None = None

        self._fwd_hook = self.target_layer.register_forward_hook(self._save_features)
        self._bwd_hook = self.target_layer.register_full_backward_hook(self._save_gradients)

    # ------------------------------------------------------------------
    # Private hook callbacks
    # ------------------------------------------------------------------

    def _save_features(self, _module, _input, output: torch.Tensor) -> None:
        self._feature_maps = output.detach()

    def _save_gradients(self, _module, _grad_input, grad_output: tuple) -> None:
        self._gradients = grad_output[0].detach()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute(
        self,
        image_tensor: torch.Tensor,
        target_class: int | None = None,
    ) -> np.ndarray:
        """
        Compute Grad-CAM heatmap for *image_tensor*.

        Parameters
        ----------
        image_tensor : torch.Tensor
            Shape (1, C, H, W) — batch dimension required.
        target_class : int | None
            Class index to explain. Defaults to the predicted class.

        Returns
        -------
        np.ndarray
            Float32 array of shape (H, W) with values in [0, 1].
        """
        self.model.zero_grad()
        image_tensor = image_tensor.requires_grad_(True)

        logits = self.model(image_tensor)          # (1, num_classes)

        if target_class is None:
            target_class = int(logits.argmax(dim=-1).item())

        # Backward on target class score
        score = logits[0, target_class]
        score.backward()

        # Grad-CAM weights: global-average-pool over spatial dims
        gradients = self._gradients        # (1, C, h, w)
        feature_maps = self._feature_maps  # (1, C, h, w)

        if gradients is None or feature_maps is None:
            raise RuntimeError("Hooks did not fire — check target layer registration.")

        weights = gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)

        # Weighted combination + ReLU
        cam = (weights * feature_maps).sum(dim=1, keepdim=True)  # (1, 1, h, w)
        cam = F.relu(cam)

        # Upsample to input spatial size
        h, w = image_tensor.shape[2], image_tensor.shape[3]
        cam = F.interpolate(cam, size=(h, w), mode="bilinear", align_corners=False)

        # Normalise to [0, 1]
        cam = cam.squeeze().cpu().numpy().astype(np.float32)
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)

        self.model.zero_grad()
        return cam

    def remove_hooks(self) -> None:
        """Call when done to free hook memory."""
        self._fwd_hook.remove()
        self._bwd_hook.remove()

    # Context-manager support
    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.remove_hooks()


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

def heatmap_to_rgb(heatmap: np.ndarray) -> np.ndarray:
    """
    Apply jet colormap to a float32 [0,1] heatmap → uint8 RGB (H, W, 3).
    """
    import matplotlib  # lazy import keeps CLI import fast
    colormap = matplotlib.colormaps["jet"]
    rgb = (colormap(heatmap)[:, :, :3] * 255).astype(np.uint8)
    return rgb


def overlay_heatmap(
    original: np.ndarray | Image.Image,
    heatmap: np.ndarray,
    alpha: float = 0.45,
) -> np.ndarray:
    """
    Blend the Grad-CAM heatmap over the original image.

    Parameters
    ----------
    original : np.ndarray | PIL.Image
        Grayscale or RGB source image.
    heatmap : np.ndarray
        Float32 array [0, 1] of shape (H, W).
    alpha : float
        Heatmap weight in the blend.

    Returns
    -------
    np.ndarray  uint8 RGB (H, W, 3)
    """
    if isinstance(original, Image.Image):
        original = np.array(original)

    # Ensure RGB
    if original.ndim == 2 or (original.ndim == 3 and original.shape[2] == 1):
        gray = original.squeeze()
        original_rgb = np.stack([gray, gray, gray], axis=-1)
    else:
        original_rgb = original

    # Resize heatmap to match original if needed
    if heatmap.shape != original_rgb.shape[:2]:
        h, w = original_rgb.shape[:2]
        heatmap_resized = np.array(
            Image.fromarray((heatmap * 255).astype(np.uint8)).resize((w, h), Image.BILINEAR)
        ).astype(np.float32) / 255.0
    else:
        heatmap_resized = heatmap

    heat_rgb = heatmap_to_rgb(heatmap_resized).astype(np.float32)
    base_rgb = original_rgb.astype(np.float32)

    blended = (1 - alpha) * base_rgb + alpha * heat_rgb
    return np.clip(blended, 0, 255).astype(np.uint8)


def compute_gradcam(
    model: nn.Module,
    image_tensor: torch.Tensor,
    target_class: int | None = None,
) -> np.ndarray:
    """
    Convenience function — create a GradCAM, compute, then remove hooks.

    Returns float32 heatmap in [0, 1].
    """
    with GradCAM(model) as cam:
        return cam.compute(image_tensor, target_class)
