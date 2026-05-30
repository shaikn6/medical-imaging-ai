"""
Lightweight 4-class chest X-ray CNN.

Architecture: 4 convolutional blocks (32 → 64 → 128 → 256 channels)
followed by a two-layer classifier head with dropout.

Input:  (B, 1, 224, 224)  — grayscale, normalised to [-1, 1]
Output: (B, 4)            — logits for Normal / Pneumonia / Cardiomegaly /
                             Pleural Effusion
"""

import torch
import torch.nn as nn


CLASSES = ["Normal", "Pneumonia", "Cardiomegaly", "Pleural Effusion"]
NUM_CLASSES = len(CLASSES)


class XRayCNN(nn.Module):
    """
    Convolutional neural network for grayscale chest X-ray classification.

    The ``features`` attribute holds all Conv2d layers so that Grad-CAM
    can register hooks on the final convolutional block without reaching
    into the classifier head.
    """

    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1 — 224×224 → 112×112
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 2 — 112×112 → 56×56
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 3 — 56×56 → 28×28
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 4 — 28×28 → 4×4 (AdaptiveAvgPool)
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(4),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, 1, H, W) → (B, C)
        return self.classifier(self.features(x))

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    @property
    def last_conv_layer(self) -> nn.Module:
        """Return the last Conv2d layer (target for Grad-CAM hooks)."""
        last = None
        for module in self.features.modules():
            if isinstance(module, nn.Conv2d):
                last = module
        if last is None:
            raise RuntimeError("No Conv2d layer found in self.features")
        return last

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return softmax class probabilities."""
        with torch.no_grad():
            logits = self(x)
        return torch.softmax(logits, dim=-1)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Return argmax class indices."""
        return self.predict_proba(x).argmax(dim=-1)


def build_model(num_classes: int = NUM_CLASSES) -> XRayCNN:
    """Factory — returns an untrained XRayCNN."""
    return XRayCNN(num_classes=num_classes)


def load_model(checkpoint_path: str, device: str = "cpu") -> XRayCNN:
    """Load a saved model checkpoint."""
    model = build_model()
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model
