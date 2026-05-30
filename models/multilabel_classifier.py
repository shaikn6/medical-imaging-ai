"""
Multi-label chest X-ray classifier — 5 pathology labels.

Extends the V1 binary XRayCNN to simultaneously detect 5 pathologies:
  0 - Pneumonia
  1 - Cardiomegaly
  2 - Effusion
  3 - Atelectasis
  4 - Nodule

Architecture changes vs V1:
  - Output head uses Sigmoid (not Softmax) for independent per-label probability.
  - Loss: Binary Cross-Entropy with logits (BCEWithLogitsLoss), summed across labels.

Multi-label metrics:
  - Per-class AUC (one-vs-rest)
  - Macro-averaged AUC
  - Hamming loss (fraction of incorrectly predicted labels)

Usage::

    from models.multilabel_classifier import (
        MultiLabelXRayCNN,
        build_multilabel_model,
        MultiLabelMetrics,
        evaluate_multilabel,
    )

    model = build_multilabel_model()
    logits = model(torch.randn(4, 1, 224, 224))   # (4, 5)
    probs  = torch.sigmoid(logits)                # per-label probabilities

    metrics = evaluate_multilabel(model, dataloader)
    print(metrics.macro_auc, metrics.hamming_loss)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from model.cnn_classifier import XRayCNN  # Reuse V1 encoder architecture


# ---------------------------------------------------------------------------
# Label definitions
# ---------------------------------------------------------------------------

MULTILABEL_CLASSES = [
    "Pneumonia",
    "Cardiomegaly",
    "Effusion",
    "Atelectasis",
    "Nodule",
]
NUM_LABELS = len(MULTILABEL_CLASSES)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class MultiLabelXRayCNN(nn.Module):
    """
    Multi-label chest X-ray classifier.

    Reuses the 4-block CNN encoder from V1 XRayCNN and replaces the single
    classification head with a multi-label head (Sigmoid output, not Softmax).

    Input:  (B, 1, 224, 224)
    Output: (B, NUM_LABELS)  — raw logits (apply sigmoid for probabilities)
    """

    def __init__(self, num_labels: int = NUM_LABELS):
        super().__init__()
        # Borrow the feature extractor from V1
        _backbone = XRayCNN(num_classes=num_labels)
        self.features = _backbone.features

        # Multi-label head  (no final softmax — sigmoid applied externally)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(512, num_labels),
            # NOTE: NO Sigmoid here — BCEWithLogitsLoss expects raw logits
        )
        self.num_labels = num_labels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))

    @property
    def last_conv_layer(self) -> nn.Module:
        """Return the last Conv2d layer for Grad-CAM."""
        last = None
        for module in self.features.modules():
            if isinstance(module, nn.Conv2d):
                last = module
        if last is None:
            raise RuntimeError("No Conv2d layer found in self.features")
        return last

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return per-label sigmoid probabilities (B, num_labels)."""
        with torch.no_grad():
            return torch.sigmoid(self(x))

    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """Return binary predictions (B, num_labels) at the given threshold."""
        return (self.predict_proba(x) >= threshold).float()


def build_multilabel_model(num_labels: int = NUM_LABELS) -> MultiLabelXRayCNN:
    """Factory — returns an untrained MultiLabelXRayCNN."""
    return MultiLabelXRayCNN(num_labels=num_labels)


# ---------------------------------------------------------------------------
# Synthetic multi-label dataset
# ---------------------------------------------------------------------------

class SyntheticMultiLabelDataset(Dataset):
    """
    Synthetic multi-label chest X-ray dataset.

    Each image is a 224×224 grayscale array with one or more pathology
    features synthetically embedded. Labels are binary vectors of length 5.

    Pathology embedding:
      Pneumonia    — consolidation blob in lower lung
      Cardiomegaly — enlarged cardiac silhouette
      Effusion     — bright base of lung
      Atelectasis  — linear opacity in upper lung
      Nodule       — small round bright spot
    """

    def __init__(self, n_samples: int = 400, size: int = 224, seed: int = 42):
        self.n = n_samples
        self.size = size
        self.rng = np.random.default_rng(seed)
        self._images, self._labels = self._generate_all()

    def _generate_all(self):
        images = []
        labels = []
        for i in range(self.n):
            img, lbl = self._make_sample(i)
            images.append(img)
            labels.append(lbl)
        return images, labels

    def _make_sample(self, idx: int):
        rng = np.random.default_rng(idx)
        size = self.size

        # Base X-ray background
        img = rng.normal(80, 15, (size, size)).astype(np.float32)
        y_g, x_g = np.ogrid[:size, :size]

        # Lung ovals
        left_lung  = ((x_g - size // 3) ** 2 / (size // 7) ** 2 +
                      (y_g - size // 2) ** 2 / (size // 5) ** 2) <= 1.0
        right_lung = ((x_g - 2 * size // 3) ** 2 / (size // 7) ** 2 +
                      (y_g - size // 2) ** 2 / (size // 5) ** 2) <= 1.0
        img[left_lung]  -= 20
        img[right_lung] -= 20

        # Cardiac silhouette (normal size)
        heart_cx, heart_cy = size // 2, int(size * 0.55)
        heart_rx, heart_ry = size // 10, size // 8
        heart = ((x_g - heart_cx) ** 2 / heart_rx ** 2 +
                 (y_g - heart_cy) ** 2 / heart_ry ** 2) <= 1.0
        img[heart] += 40

        # Randomly assign labels (at least one label True)
        label = np.zeros(NUM_LABELS, dtype=np.float32)
        while label.sum() == 0:
            label = (rng.random(NUM_LABELS) > 0.6).astype(np.float32)

        # Embed each active pathology
        # 0: Pneumonia — consolidation
        if label[0] > 0:
            cx_p = int(size // 3 + rng.integers(-10, 10))
            cy_p = int(size * 0.65 + rng.integers(-5, 5))
            patch = ((x_g - cx_p) ** 2 / (size // 12) ** 2 +
                     (y_g - cy_p) ** 2 / (size // 10) ** 2) <= 1.0
            img[patch & left_lung] += rng.normal(50, 10, img[patch & left_lung].shape)

        # 1: Cardiomegaly — bigger heart
        if label[1] > 0:
            big_heart = ((x_g - heart_cx) ** 2 / (heart_rx * 1.4) ** 2 +
                         (y_g - heart_cy) ** 2 / (heart_ry * 1.4) ** 2) <= 1.0
            img[big_heart & ~heart] += rng.normal(35, 6, img[big_heart & ~heart].shape)

        # 2: Effusion — bright base left lung
        if label[2] > 0:
            cutoff = int(size * 0.72)
            base = np.zeros((size, size), dtype=bool)
            base[cutoff:, :size // 2] = True
            base = base & left_lung
            img[base] += rng.normal(55, 8, img[base].shape)

        # 3: Atelectasis — linear band in upper right lung
        if label[3] > 0:
            row_start = int(size * 0.38)
            row_end   = row_start + 4
            band = np.zeros((size, size), dtype=bool)
            band[row_start:row_end, :] = True
            band = band & right_lung
            img[band] += rng.normal(30, 5, img[band].shape)

        # 4: Nodule — small round spot
        if label[4] > 0:
            ncx = int(2 * size // 3 + rng.integers(-20, 20))
            ncy = int(size * 0.4 + rng.integers(-10, 10))
            nr  = int(rng.integers(4, 10))
            nodule = ((x_g - ncx) ** 2 + (y_g - ncy) ** 2) <= nr ** 2
            img[nodule & right_lung] += rng.normal(40, 8, img[nodule & right_lung].shape)

        arr = np.clip(img, 0, 255).astype(np.uint8)
        return arr, label

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int):
        arr   = self._images[idx].astype(np.float32) / 255.0
        arr   = (arr - 0.5) / 0.5
        img   = torch.tensor(arr, dtype=torch.float32).unsqueeze(0)   # (1, H, W)
        label = torch.tensor(self._labels[idx], dtype=torch.float32)  # (5,)
        return img, label


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

@dataclass
class MultiLabelMetrics:
    """Container for multi-label classification metrics."""
    per_class_auc:  list[float]   # AUC per pathology label
    macro_auc:      float         # Mean AUC across all labels
    hamming_loss:   float         # Fraction of incorrectly predicted labels
    per_class_names: list[str]    # Label names aligned with per_class_auc

    def __str__(self) -> str:
        lines = ["MultiLabel Metrics:"]
        for name, auc in zip(self.per_class_names, self.per_class_auc):
            lines.append(f"  {name:<15s}  AUC={auc:.4f}")
        lines.append(f"  {'Macro AUC':<15s}  {self.macro_auc:.4f}")
        lines.append(f"  {'Hamming Loss':<15s}  {self.hamming_loss:.4f}")
        return "\n".join(lines)


def evaluate_multilabel(
    model: MultiLabelXRayCNN,
    dataloader: DataLoader,
    threshold: float = 0.5,
    device: str = "cpu",
) -> MultiLabelMetrics:
    """
    Compute multi-label classification metrics on a dataloader.

    Parameters
    ----------
    model : MultiLabelXRayCNN
    dataloader : DataLoader
    threshold : float
        Decision threshold for binary predictions.
    device : str

    Returns
    -------
    MultiLabelMetrics
    """
    from sklearn.metrics import roc_auc_score

    model.eval()
    all_probs  = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            probs = torch.sigmoid(model(images)).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(labels.numpy())

    probs_arr  = np.concatenate(all_probs,  axis=0)   # (N, 5)
    labels_arr = np.concatenate(all_labels, axis=0)   # (N, 5)

    # Per-class AUC
    per_class_auc = []
    for k in range(NUM_LABELS):
        unique = np.unique(labels_arr[:, k])
        if len(unique) < 2:
            per_class_auc.append(float("nan"))
        else:
            per_class_auc.append(float(roc_auc_score(labels_arr[:, k], probs_arr[:, k])))

    valid_aucs = [a for a in per_class_auc if not np.isnan(a)]
    macro_auc = float(np.mean(valid_aucs)) if valid_aucs else float("nan")

    # Hamming loss
    preds = (probs_arr >= threshold).astype(int)
    hamming = float(np.mean(preds != labels_arr.astype(int)))

    return MultiLabelMetrics(
        per_class_auc=per_class_auc,
        macro_auc=macro_auc,
        hamming_loss=hamming,
        per_class_names=MULTILABEL_CLASSES,
    )


def train_multilabel_demo(
    n_samples: int = 400,
    epochs: int = 5,
    batch_size: int = 16,
    lr: float = 1e-3,
    test_split: float = 0.2,
    seed: int = 42,
    device: str = "cpu",
) -> dict:
    """
    Quick training demo on synthetic multi-label data.

    Returns
    -------
    dict with keys: metrics (MultiLabelMetrics), model (trained)
    """
    torch.manual_seed(seed)
    dataset = SyntheticMultiLabelDataset(n_samples=n_samples, seed=seed)

    n_test  = int(len(dataset) * test_split)
    n_train = len(dataset) - n_test
    train_ds, test_ds = torch.utils.data.random_split(
        dataset, [n_train, n_test],
        generator=torch.Generator().manual_seed(seed),
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

    model = build_multilabel_model().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for _ in range(epochs):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

    metrics = evaluate_multilabel(model, test_loader, device=device)
    return {"metrics": metrics, "model": model}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Multi-label classifier demo (400 samples, 5 epochs) …")
    result = train_multilabel_demo()
    print(result["metrics"])
