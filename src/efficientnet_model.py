"""
EfficientNet-B4 Backbone — V2

Provides a second model option using torchvision's EfficientNet-B4 with the
same 4-class multi-label head as XRayCNN.  Adapted for (1, 224, 224) grayscale
input by replacing the first conv channel dimension.

Also exposes ``compare_models`` — a side-by-side per-class AUC utility.

Usage::

    from src.efficientnet_model import build_efficientnet_model, compare_models

    model = build_efficientnet_model(num_classes=4, pretrained=False)
    logits = model(torch.randn(2, 1, 224, 224))   # (2, 4)

    results = compare_models(model_a, model_b, dataloader)
    # {'model_a': {'Normal': 0.94, ...}, 'model_b': {'Normal': 0.91, ...}}
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights

from model.cnn_classifier import CLASSES, NUM_CLASSES


# ---------------------------------------------------------------------------
# Model definition
# ---------------------------------------------------------------------------

class EfficientNetXRay(nn.Module):
    """
    EfficientNet-B4 adapted for grayscale chest X-ray classification.

    Changes vs. stock EfficientNet-B4:
    - First conv layer input channels changed from 3 → 1 (grayscale)
    - Classifier head replaced with a linear layer matching ``num_classes``
    - No sigmoid applied — raw logits are returned (same contract as XRayCNN)

    The ``last_conv_layer`` property points to the last Conv2d in the
    feature extractor so ScoreCAM / Grad-CAM can target it.
    """

    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        # Load without pre-trained weights for the full model first,
        # then swap channels (pre-trained weights handled in factory).
        self.backbone = efficientnet_b4(weights=None)

        # Adapt first conv: 3 → 1 input channel
        orig_conv = self.backbone.features[0][0]  # Conv2d
        self.backbone.features[0][0] = nn.Conv2d(
            in_channels=1,
            out_channels=orig_conv.out_channels,
            kernel_size=orig_conv.kernel_size,
            stride=orig_conv.stride,
            padding=orig_conv.padding,
            bias=orig_conv.bias is not None,
        )

        # Replace classifier head
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.4, inplace=True),
            nn.Linear(in_features, num_classes),
        )

        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, 1, H, W) → (B, C)
        return self.backbone(x)

    @property
    def last_conv_layer(self) -> nn.Conv2d:
        """Return the last Conv2d in the EfficientNet feature extractor."""
        last: Optional[nn.Conv2d] = None
        for module in self.backbone.features.modules():
            if isinstance(module, nn.Conv2d):
                last = module
        if last is None:
            raise RuntimeError("No Conv2d found in EfficientNet backbone features.")
        return last

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return softmax class probabilities (no gradient)."""
        with torch.no_grad():
            logits = self(x)
        return torch.softmax(logits, dim=-1)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Return argmax class indices."""
        return self.predict_proba(x).argmax(dim=-1)


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------

def build_efficientnet_model(
    num_classes: int = NUM_CLASSES,
    pretrained: bool = True,
) -> EfficientNetXRay:
    """
    Build an EfficientNet-B4 model for grayscale X-ray classification.

    Parameters
    ----------
    num_classes : int
        Number of output classes.  Default 4 (matches XRayCNN).
    pretrained : bool
        If True, initialises the backbone with ImageNet-1k weights,
        then averages the first-conv weights across the 3 RGB channels
        to produce a single-channel initialisation.

    Returns
    -------
    EfficientNetXRay
        Model in eval mode.
    """
    model = EfficientNetXRay(num_classes=num_classes)

    if pretrained:
        # Load ImageNet weights for the backbone parts we keep
        imagenet_model = efficientnet_b4(weights=EfficientNet_B4_Weights.IMAGENET1K_V1)
        imagenet_state = imagenet_model.state_dict()
        model_state = model.state_dict()

        # Copy all matching keys (exclude first conv weight/bias and classifier)
        first_conv_key = "backbone.features.0.0.weight"
        classifier_prefix = "backbone.classifier"

        filtered_state: Dict[str, torch.Tensor] = {}
        for key, value in imagenet_state.items():
            mapped_key = f"backbone.{key}"
            if mapped_key not in model_state:
                continue
            if mapped_key == first_conv_key:
                # Average the 3-channel weights to 1-channel
                filtered_state[mapped_key] = value.mean(dim=1, keepdim=True)
            elif mapped_key.startswith(classifier_prefix):
                continue  # Skip — our head has different dimensions
            else:
                filtered_state[mapped_key] = value

        model_state.update(filtered_state)
        model.load_state_dict(model_state, strict=False)

    model.eval()
    return model


# ---------------------------------------------------------------------------
# Model comparison utility
# ---------------------------------------------------------------------------

def compare_models(
    model_a: nn.Module,
    model_b: nn.Module,
    dataloader: Iterable,
    class_names: Optional[list[str]] = None,
    device: str = "cpu",
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate two models side-by-side and return per-class AUC for each.

    Parameters
    ----------
    model_a, model_b : nn.Module
        Models to evaluate.  Both must accept (B, 1, H, W) input and
        return (B, num_classes) logits.
    dataloader : Iterable
        Yields ``(images, labels)`` batches.  ``labels`` must be 1-D integer
        class indices.
    class_names : list[str] | None
        Class names for the output dict.  Defaults to CLASSES.
    device : str
        Torch device to run inference on.

    Returns
    -------
    dict
        ``{
            'model_a': {'Normal': 0.94, 'Pneumonia': 0.91, ...},
            'model_b': {'Normal': 0.88, 'Pneumonia': 0.87, ...},
          }``
    """
    names = class_names or CLASSES
    torch_device = torch.device(device)

    model_a = model_a.to(torch_device).eval()
    model_b = model_b.to(torch_device).eval()

    all_labels: list[np.ndarray] = []
    scores_a: list[np.ndarray] = []
    scores_b: list[np.ndarray] = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(torch_device)
            labels_np = labels.cpu().numpy() if isinstance(labels, torch.Tensor) else np.asarray(labels)

            probs_a = torch.softmax(model_a(images), dim=-1).cpu().numpy()
            probs_b = torch.softmax(model_b(images), dim=-1).cpu().numpy()

            all_labels.append(labels_np)
            scores_a.append(probs_a)
            scores_b.append(probs_b)

    y_true = np.concatenate(all_labels)         # (N,)
    y_scores_a = np.concatenate(scores_a, axis=0)  # (N, C)
    y_scores_b = np.concatenate(scores_b, axis=0)  # (N, C)

    def _per_class_auc(y_true: np.ndarray, y_scores: np.ndarray, names: list[str]) -> Dict[str, float]:
        result: Dict[str, float] = {}
        for k, name in enumerate(names):
            binary = (y_true == k).astype(int)
            if binary.sum() == 0 or binary.sum() == len(binary):
                result[name] = float("nan")
            else:
                try:
                    result[name] = float(roc_auc_score(binary, y_scores[:, k]))
                except ValueError:
                    result[name] = float("nan")
        return result

    return {
        "model_a": _per_class_auc(y_true, y_scores_a, names),
        "model_b": _per_class_auc(y_true, y_scores_b, names),
    }
