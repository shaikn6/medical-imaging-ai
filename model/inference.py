"""
Full inference pipeline: preprocess → predict → Grad-CAM.

Usage::

    from model.inference import InferencePipeline

    pipeline = InferencePipeline("model/xray_model.pth")
    result = pipeline.run(pil_image_or_path)
    print(result.predicted_class, result.confidence)
    result.overlay.save("gradcam_output.png")
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from model.cnn_classifier import load_model, CLASSES          # noqa: E402
from model.gradcam import GradCAM, overlay_heatmap            # noqa: E402
from data.augmentation import get_val_transforms              # noqa: E402


IMG_SIZE = 224


@dataclass
class InferenceResult:
    predicted_class: str
    predicted_idx: int
    confidence: float
    probabilities: dict[str, float]
    heatmap: np.ndarray         # float32 [0, 1] Grad-CAM map
    overlay: Image.Image        # PIL RGB blended image


class InferencePipeline:
    """
    End-to-end: PIL image → prediction + Grad-CAM overlay.

    Parameters
    ----------
    checkpoint_path : str
        Path to the saved .pth checkpoint.
    device : str
        "cpu" or "cuda".
    """

    def __init__(self, checkpoint_path: str, device: str = "cpu"):
        self.device = torch.device(device)
        self.model = load_model(checkpoint_path, device=device)
        self.transforms = get_val_transforms()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _preprocess(self, image: Image.Image | str | np.ndarray) -> torch.Tensor:
        """Convert various input types to a (1, 1, 224, 224) tensor."""
        if isinstance(image, str):
            image = Image.open(image)
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        if image.mode != "L":
            image = image.convert("L")
        image = image.resize((IMG_SIZE, IMG_SIZE))

        tensor = self.transforms(image).unsqueeze(0).to(self.device)
        return tensor

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        image: Image.Image | str | np.ndarray,
        target_class: int | None = None,
    ) -> InferenceResult:
        """
        Run full inference + Grad-CAM.

        Parameters
        ----------
        image : PIL.Image | str | np.ndarray
            Grayscale chest X-ray (path, PIL, or numpy array).
        target_class : int | None
            Class to explain. Defaults to predicted class.

        Returns
        -------
        InferenceResult
        """
        # Load/convert PIL for later overlay
        if isinstance(image, str):
            pil_image = Image.open(image).convert("L").resize((IMG_SIZE, IMG_SIZE))
        elif isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image.astype(np.uint8) if image.dtype != np.uint8 else image)
            pil_image = pil_image.convert("L").resize((IMG_SIZE, IMG_SIZE))
        else:
            pil_image = image.convert("L").resize((IMG_SIZE, IMG_SIZE))

        tensor = self._preprocess(pil_image)

        # Prediction
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=-1).squeeze().cpu().numpy()

        pred_idx = int(probs.argmax())
        pred_class = CLASSES[pred_idx]
        confidence = float(probs[pred_idx])

        # Grad-CAM
        explain_class = target_class if target_class is not None else pred_idx
        with GradCAM(self.model) as cam:
            heatmap = cam.compute(tensor, target_class=explain_class)

        overlay = Image.fromarray(overlay_heatmap(np.array(pil_image), heatmap))

        return InferenceResult(
            predicted_class=pred_class,
            predicted_idx=pred_idx,
            confidence=confidence,
            probabilities={cls: float(p) for cls, p in zip(CLASSES, probs)},
            heatmap=heatmap,
            overlay=overlay,
        )


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    ckpt = os.path.join(_HERE, "xray_model.pth")
    if not os.path.exists(ckpt):
        print(f"Checkpoint not found: {ckpt}")
        print("Run  python -m model.trainer  first.")
        sys.exit(1)

    # Generate a quick synthetic test image
    from data.synthetic_xray import generate_pneumonia_xray
    arr = generate_pneumonia_xray(seed=7)
    pil = Image.fromarray(arr, mode="L")

    pipeline = InferencePipeline(ckpt)
    result = pipeline.run(pil)
    print(f"Predicted: {result.predicted_class} ({result.confidence:.1%})")
    print("Class probabilities:")
    for cls, p in result.probabilities.items():
        print(f"  {cls}: {p:.3f}")
