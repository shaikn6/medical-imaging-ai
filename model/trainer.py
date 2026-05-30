"""
Training loop for XRayCNN.

Trains on 800 synthetic X-ray images for 15 epochs using:
  - Adam optimizer
  - CrossEntropyLoss
  - StepLR learning-rate scheduler

Saves the best-val-accuracy checkpoint to model/xray_model.pth and
per-epoch metrics to model/training_history.json.

Run directly:
    python -m model.trainer
"""

from __future__ import annotations

import json
import os
import sys
import time

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

# ---------------------------------------------------------------------------
# Resolve project root so the module works when invoked from any cwd
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from data.synthetic_xray import generate_dataset, get_dataset_class   # noqa: E402
from data.augmentation import get_train_transforms, get_val_transforms # noqa: E402
from model.cnn_classifier import build_model, CLASSES                  # noqa: E402

# ---------------------------------------------------------------------------
# Hyper-parameters
# ---------------------------------------------------------------------------

IMAGES_PER_CLASS = 200
BATCH_SIZE = 32
EPOCHS = 15
LR = 1e-3
LR_STEP = 5
LR_GAMMA = 0.5

DEFAULT_DATA_DIR = os.path.join(_ROOT, "data", "synthetic_dataset")
DEFAULT_MODEL_PATH = os.path.join(_HERE, "xray_model.pth")
DEFAULT_HISTORY_PATH = os.path.join(_HERE, "training_history.json")


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> tuple[float, float]:
    """Run one epoch. If optimizer is None → eval mode."""
    training = optimizer is not None
    model.train(training)
    total_loss, correct, total = 0.0, 0, 0

    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            if training:
                optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            if training:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += images.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train(
    data_dir: str = DEFAULT_DATA_DIR,
    model_path: str = DEFAULT_MODEL_PATH,
    history_path: str = DEFAULT_HISTORY_PATH,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    images_per_class: int = IMAGES_PER_CLASS,
    device_str: str = "cpu",
) -> dict:
    """
    Generate data (if not already on disk), train, and save artefacts.

    Returns the training history dict.
    """
    device = torch.device(device_str)

    # ---- Data ----
    if not os.path.isdir(os.path.join(data_dir, "train")):
        print("Generating synthetic dataset …")
        generate_dataset(data_dir, images_per_class=images_per_class)
    else:
        print(f"Using existing dataset at {data_dir}")

    XRayDataset = get_dataset_class()
    train_ds = XRayDataset(data_dir, split="train", transform=get_train_transforms())
    val_ds   = XRayDataset(data_dir, split="val",   transform=get_val_transforms())

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"Train: {len(train_ds)} images | Val: {len(val_ds)} images")

    # ---- Model ----
    model = build_model().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=LR)
    scheduler = StepLR(optimizer, step_size=LR_STEP, gamma=LR_GAMMA)

    history: dict = {
        "train_loss": [], "train_acc": [],
        "val_loss":   [], "val_acc":   [],
        "classes": CLASSES,
        "epochs": epochs,
    }
    best_val_acc = 0.0
    t0 = time.time()

    print(f"\nTraining for {epochs} epochs on {device} …")
    print("-" * 55)

    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = _run_epoch(model, train_loader, criterion, optimizer, device)
        va_loss, va_acc = _run_epoch(model, val_loader, criterion, None, device)
        scheduler.step()

        history["train_loss"].append(round(tr_loss, 4))
        history["train_acc"].append(round(tr_acc,  4))
        history["val_loss"].append(round(va_loss, 4))
        history["val_acc"].append(round(va_acc,  4))

        print(
            f"  Epoch {epoch:2d}/{epochs}  "
            f"train_loss={tr_loss:.4f}  train_acc={tr_acc:.3f}  "
            f"val_loss={va_loss:.4f}  val_acc={va_acc:.3f}"
        )

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            torch.save(model.state_dict(), model_path)
            print(f"             ↳ saved checkpoint (val_acc={va_acc:.3f})")

    elapsed = time.time() - t0
    print(f"\nTraining complete in {elapsed:.1f}s  |  best val_acc={best_val_acc:.3f}")
    print(f"Model saved to: {model_path}")

    history["best_val_acc"] = round(best_val_acc, 4)
    history["elapsed_seconds"] = round(elapsed, 1)

    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"History saved to: {history_path}")

    return history


if __name__ == "__main__":
    train()
