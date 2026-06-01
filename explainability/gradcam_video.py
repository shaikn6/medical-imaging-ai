"""
Grad-CAM sequence visualisation + ScoreCAM implementation.

Features:
  1. GradCAMSequence — generates Grad-CAM heatmaps for a series of 10 synthetic
     'scan slices' and stacks them into an animated GIF.
  2. ScoreCAM — perturbation-based attribution that does NOT require backward
     hooks (more faithful than gradient-based methods).
  3. SideBySideComparison — renders: original | Grad-CAM | ScoreCAM | U-Net mask
     for a single slice as a side-by-side PNG.

Usage::

    from explainability.gradcam_video import (
        GradCAMSequence,
        ScoreCAM,
        make_side_by_side,
    )

    # Animated GIF across 10 slices
    seq = GradCAMSequence(model, n_slices=10)
    seq.save_gif("scan_attention.gif", fps=2)

    # ScoreCAM heatmap (no backward pass required)
    scam = ScoreCAM(model, target_layer=model.features[-3])
    heatmap = scam.compute(image_tensor)

    # Side-by-side panel
    panel = make_side_by_side(image_tensor, model, seg_model, target_class=1)
    panel.save("comparison.png")
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from model.gradcam import GradCAM, overlay_heatmap  # noqa: E402


# ---------------------------------------------------------------------------
# ScoreCAM
# ---------------------------------------------------------------------------

class ScoreCAM:
    """
    ScoreCAM (Wang et al. 2020) — perturbation-based saliency.

    Unlike Grad-CAM, ScoreCAM does not rely on gradients or backward hooks.
    It instead:
      1. Forward-passes the image to collect intermediate feature maps.
      2. Upsamples each feature map to input size and uses it as a mask.
      3. Runs the masked input through the model and collects the target-class score.
      4. Weight = softmax-normalised target score.
      5. Final map = Σ weight_k * mask_k → ReLU → normalise.

    Parameters
    ----------
    model : nn.Module
        XRayCNN or any model with a ``features`` Sequential and ``classifier``.
    target_layer : nn.Module | None
        Feature layer to extract maps from. Defaults to model.last_conv_layer.
    batch_size : int
        Number of masked images to score per forward pass (memory trade-off).
    """

    def __init__(
        self,
        model: nn.Module,
        target_layer: nn.Module | None = None,
        batch_size: int = 16,
    ):
        self.model = model
        self.model.eval()
        self.batch_size = batch_size

        self._target_layer = target_layer or model.last_conv_layer  # type: ignore[attr-defined]
        self._feature_maps: torch.Tensor | None = None
        self._fwd_hook = self._target_layer.register_forward_hook(self._save_features)

    def _save_features(self, _module, _input, output: torch.Tensor) -> None:
        self._feature_maps = output.detach()

    def compute(
        self,
        image_tensor: torch.Tensor,
        target_class: int | None = None,
    ) -> np.ndarray:
        """
        Compute ScoreCAM heatmap.

        Parameters
        ----------
        image_tensor : torch.Tensor
            Shape (1, C, H, W).
        target_class : int | None
            Class to explain. Defaults to predicted class.

        Returns
        -------
        np.ndarray  float32 (H, W) in [0, 1].
        """
        with torch.no_grad():
            logits = self.model(image_tensor)
        if target_class is None:
            target_class = int(logits.argmax(dim=-1).item())

        # feature_maps: (1, n_channels, h, w)
        if self._feature_maps is None:
            raise RuntimeError("Feature hook did not fire.")
        feature_maps = self._feature_maps  # (1, C, h, w)
        n_channels = feature_maps.shape[1]
        H, W = image_tensor.shape[2], image_tensor.shape[3]

        # Baseline score (masked image = zeros)
        baseline = torch.zeros_like(image_tensor)
        with torch.no_grad():
            baseline_score = torch.softmax(self.model(baseline), dim=-1)[0, target_class].item()

        # Collect per-channel scores
        scores = np.zeros(n_channels, dtype=np.float32)

        # Process in batches
        for start in range(0, n_channels, self.batch_size):
            end = min(start + self.batch_size, n_channels)
            batch_masked = []

            for k in range(start, end):
                # Upsample feature map k to input size
                fmap_k = feature_maps[0, k:k+1, :, :].unsqueeze(0)  # (1, 1, h, w)
                mask = F.interpolate(fmap_k, size=(H, W), mode="bilinear", align_corners=False)
                # Normalise mask to [0, 1]
                m_min, m_max = mask.min(), mask.max()
                if (m_max - m_min).abs() > 1e-8:
                    mask = (mask - m_min) / (m_max - m_min)
                # Apply mask (broadcast across colour channels)
                masked = image_tensor * mask
                batch_masked.append(masked)

            batch = torch.cat(batch_masked, dim=0)  # (batch, C, H, W)
            with torch.no_grad():
                batch_logits = self.model(batch)
                batch_probs  = torch.softmax(batch_logits, dim=-1)[:, target_class]

            scores[start:end] = batch_probs.cpu().numpy()

        # Weight channels by softmax of their scores
        weights = np.exp(scores - scores.max())
        weights /= weights.sum() + 1e-8

        # Weighted sum of upsampled feature maps
        cam = np.zeros((H, W), dtype=np.float32)
        for k in range(n_channels):
            fmap_k = feature_maps[0, k, :, :].unsqueeze(0).unsqueeze(0)
            upsampled = F.interpolate(fmap_k, size=(H, W), mode="bilinear", align_corners=False)
            cam += weights[k] * upsampled.squeeze().cpu().numpy()

        # ReLU and normalise
        cam = np.maximum(cam, 0)
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)

        return cam.astype(np.float32)

    def remove_hook(self) -> None:
        """Remove the forward hook to free resources."""
        self._fwd_hook.remove()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.remove_hook()


# ---------------------------------------------------------------------------
# GradCAMSequence — animated GIF
# ---------------------------------------------------------------------------

class GradCAMSequence:
    """
    Generate Grad-CAM for a series of synthetic 'scan slices' and save as GIF.

    Parameters
    ----------
    model : nn.Module
        XRayCNN (or equivalent).
    n_slices : int
        Number of slices in the sequence (default 10).
    size : int
        Image size in pixels (default 224).
    target_class : int | None
        Fixed class to explain across all slices. If None, uses per-slice prediction.
    seed : int
        Base seed for reproducibility.
    """

    def __init__(
        self,
        model: nn.Module,
        n_slices: int = 10,
        size: int = 224,
        target_class: int | None = None,
        seed: int = 0,
    ):
        self.model = model
        self.n_slices = n_slices
        self.size = size
        self.target_class = target_class
        self.seed = seed

    def _make_slice_tensor(self, slice_idx: int) -> tuple[torch.Tensor, np.ndarray]:
        """Generate a synthetic slice and return (tensor, uint8_array)."""
        from data.synthetic_xray import generate_pneumonia_xray, generate_normal_xray

        np.random.default_rng(self.seed + slice_idx)

        # Alternate between normal and slightly lesion-shifted slices
        if slice_idx < self.n_slices // 2:
            arr = generate_normal_xray(self.size, seed=self.seed + slice_idx)
        else:
            arr = generate_pneumonia_xray(self.size, seed=self.seed + slice_idx)

        # Add slight vertical shift to simulate slice movement
        shift = int((slice_idx / self.n_slices) * 20 - 10)
        arr_shifted = np.roll(arr, shift, axis=0).astype(np.uint8)

        # Tensor
        tensor = torch.tensor(
            arr_shifted.astype(np.float32) / 255.0, dtype=torch.float32
        ).unsqueeze(0).unsqueeze(0)
        tensor = (tensor - 0.5) / 0.5

        return tensor, arr_shifted

    def generate_frames(self) -> list[Image.Image]:
        """
        Generate list of RGB PIL frames — one per slice with Grad-CAM overlay.

        Returns
        -------
        list[Image.Image]   RGB frames, shape (size, size, 3).
        """
        frames: list[Image.Image] = []

        for idx in range(self.n_slices):
            tensor, arr = self._make_slice_tensor(idx)
            with GradCAM(self.model) as cam:
                heatmap = cam.compute(tensor, target_class=self.target_class)
            overlay = overlay_heatmap(arr, heatmap)
            frame = Image.fromarray(overlay, mode="RGB")
            # Annotate slice number
            frames.append(frame)

        return frames

    def save_gif(
        self,
        output_path: str,
        fps: int = 2,
    ) -> str:
        """
        Save animated GIF of Grad-CAM attention across scan slices.

        Parameters
        ----------
        output_path : str
            Destination file path (e.g. "scan_attention.gif").
        fps : int
            Frames per second.

        Returns
        -------
        str  The output_path.
        """
        frames = self.generate_frames()
        duration_ms = int(1000 / fps)

        frames[0].save(
            output_path,
            save_all=True,
            append_images=frames[1:],
            loop=0,
            duration=duration_ms,
            optimize=False,
        )
        return output_path


# ---------------------------------------------------------------------------
# Side-by-side comparison panel
# ---------------------------------------------------------------------------

def make_side_by_side(
    image_tensor: torch.Tensor,
    cls_model: nn.Module,
    seg_model: nn.Module | None = None,
    target_class: int | None = None,
    size: int = 224,
) -> Image.Image:
    """
    Render a 4-panel side-by-side image:
      [Original | Grad-CAM | ScoreCAM | Segmentation mask]

    Parameters
    ----------
    image_tensor : torch.Tensor
        (1, 1, H, W) normalised input.
    cls_model : nn.Module
        XRayCNN (for Grad-CAM and ScoreCAM).
    seg_model : nn.Module | None
        UNet (for segmentation mask panel). If None, shows blank panel.
    target_class : int | None
        Class index to explain. Defaults to predicted class.
    size : int
        Height/width of each panel in the output image.

    Returns
    -------
    PIL.Image  RGB image of shape (size, 4*size).
    """
    # ---- Original --------------------------------------------------------
    image_tensor = image_tensor.detach()
    arr = image_tensor.squeeze().cpu().numpy()
    arr = ((arr + 1.0) / 2.0 * 255.0).clip(0, 255).astype(np.uint8)
    original_rgb = np.stack([arr, arr, arr], axis=-1)

    # ---- Grad-CAM --------------------------------------------------------
    with GradCAM(cls_model) as cam:
        gcam = cam.compute(image_tensor, target_class=target_class)
    gcam_rgb = overlay_heatmap(arr, gcam)

    # ---- ScoreCAM --------------------------------------------------------
    with ScoreCAM(cls_model) as scam:
        scam_map = scam.compute(image_tensor, target_class=target_class)
    scam_rgb = overlay_heatmap(arr, scam_map)

    # ---- Segmentation mask -----------------------------------------------
    if seg_model is not None:
        # Resize to 256 for UNet, then back to size
        inp = F.interpolate(image_tensor, size=(256, 256), mode="bilinear", align_corners=False)
        seg_mask = seg_model.predict_mask(inp, threshold=0.5).squeeze().cpu().numpy()
        seg_mask_resized = np.array(
            Image.fromarray((seg_mask * 255).astype(np.uint8)).resize((size, size), Image.NEAREST)
        )
        seg_rgb = np.stack([seg_mask_resized, seg_mask_resized, seg_mask_resized], axis=-1)
    else:
        seg_rgb = np.zeros((size, size, 3), dtype=np.uint8)

    # ---- Resize each panel to (size, size) and stack horizontally --------
    def _to_pil_resized(rgb_arr: np.ndarray) -> Image.Image:
        return Image.fromarray(rgb_arr.astype(np.uint8)).resize((size, size), Image.BILINEAR)

    panels = [
        _to_pil_resized(original_rgb),
        _to_pil_resized(gcam_rgb),
        _to_pil_resized(scam_rgb),
        _to_pil_resized(seg_rgb),
    ]

    canvas = Image.new("RGB", (size * 4, size))
    for i, panel in enumerate(panels):
        canvas.paste(panel, (i * size, 0))

    return canvas


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile

    sys.path.insert(0, _ROOT)
    from model.cnn_classifier import build_model
    from models.unet import build_unet

    cls_model = build_model()
    cls_model.eval()
    seg_model = build_unet()
    seg_model.eval()

    with tempfile.TemporaryDirectory() as tmp:
        gif_path = os.path.join(tmp, "scan_attention.gif")
        seq = GradCAMSequence(cls_model, n_slices=5, size=64, seed=0)
        seq.save_gif(gif_path, fps=2)
        assert os.path.exists(gif_path) and os.path.getsize(gif_path) > 0
        print(f"  GIF saved: {gif_path} ({os.path.getsize(gif_path)} bytes)")

        x = torch.randn(1, 1, 64, 64)
        with ScoreCAM(cls_model, batch_size=4) as scam:
            hm = scam.compute(x)
        assert hm.shape == (64, 64) and hm.dtype == np.float32
        print(f"  ScoreCAM heatmap shape: {hm.shape}")

        panel = make_side_by_side(x, cls_model, seg_model, size=64)
        assert panel.size == (256, 64)
        print(f"  Side-by-side panel size: {panel.size}")

    print("gradcam_video smoke test passed.")
