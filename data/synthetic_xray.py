"""
Synthetic chest X-ray image generator.

Generates realistic-looking grayscale 224×224 images for 4 pathology classes
using numpy. No real patient data is used.

Classes:
    0 - Normal
    1 - Pneumonia
    2 - Cardiomegaly
    3 - Pleural Effusion
"""

import os
import json
import random
import numpy as np
from PIL import Image


CLASSES = ["Normal", "Pneumonia", "Cardiomegaly", "Pleural Effusion"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
IMG_SIZE = 224


# ---------------------------------------------------------------------------
# Per-class generators
# ---------------------------------------------------------------------------

def _base_xray(rng: np.random.Generator, size: int = IMG_SIZE) -> np.ndarray:
    """Background tissue noise + faint ribcage lines."""
    img = rng.normal(80, 15, (size, size)).astype(np.float32)

    # Faint horizontal ribcage lines
    for row in range(40, size - 20, 16):
        thickness = int(rng.integers(1, 3))
        img[row:row + thickness, 20:size - 20] += rng.normal(8, 3, (thickness, size - 40))

    return img


def _add_lung_ovals(
    rng: np.random.Generator, img: np.ndarray, size: int = IMG_SIZE
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Draw bilateral lung ovals (darker = air-filled)."""
    y, x = np.ogrid[:size, :size]

    # Left lung (patient's left = image right)
    left_mask = ((x - 65) ** 2 / 35 ** 2 + (y - 110) ** 2 / 65 ** 2) <= 1.0
    img[left_mask] += rng.normal(-20, 5, img[left_mask].shape)

    # Right lung (slightly larger)
    right_mask = ((x - 155) ** 2 / 40 ** 2 + (y - 110) ** 2 / 70 ** 2) <= 1.0
    img[right_mask] += rng.normal(-20, 5, img[right_mask].shape)

    return img, left_mask, right_mask


def _add_cardiac_silhouette(
    rng: np.random.Generator, img: np.ndarray,
    cx: int = 105, cy: int = 120, rx: int = 20, ry: int = 30, size: int = IMG_SIZE
) -> np.ndarray:
    """Bright oval representing the cardiac shadow."""
    y, x = np.ogrid[:size, :size]
    cardiac = ((x - cx) ** 2 / rx ** 2 + (y - cy) ** 2 / ry ** 2) <= 1.0
    img[cardiac] += rng.normal(40, 6, img[cardiac].shape)
    return img


def generate_normal_xray(size: int = IMG_SIZE, seed: int | None = None) -> np.ndarray:
    """Normal chest X-ray: bilateral clear lungs, standard cardiac size."""
    rng = np.random.default_rng(seed)
    img = _base_xray(rng, size)
    img, _, _ = _add_lung_ovals(rng, img, size)
    img = _add_cardiac_silhouette(rng, img, size=size)
    return np.clip(img, 0, 255).astype(np.uint8)


def generate_pneumonia_xray(size: int = IMG_SIZE, seed: int | None = None) -> np.ndarray:
    """
    Pneumonia: one lung has a denser (brighter) consolidation patch
    simulating alveolar filling.
    """
    rng = np.random.default_rng(seed)
    img = _base_xray(rng, size)
    img, left_mask, _ = _add_lung_ovals(rng, img, size)
    img = _add_cardiac_silhouette(rng, img, size=size)

    # Consolidation in left lower lobe
    y, x = np.ogrid[:size, :size]
    consolidation = (
        ((x - 65) ** 2 / 22 ** 2 + (y - 140) ** 2 / 28 ** 2) <= 1.0
    ) & left_mask
    img[consolidation] += rng.normal(50, 10, img[consolidation].shape)

    return np.clip(img, 0, 255).astype(np.uint8)


def generate_cardiomegaly_xray(size: int = IMG_SIZE, seed: int | None = None) -> np.ndarray:
    """
    Cardiomegaly: cardiac silhouette is ~40% wider than normal.
    Cardiothoracic ratio > 0.5.
    """
    rng = np.random.default_rng(seed)
    img = _base_xray(rng, size)
    img, _, _ = _add_lung_ovals(rng, img, size)
    # Wider cardiac silhouette
    img = _add_cardiac_silhouette(rng, img, cx=105, cy=122, rx=28, ry=36, size=size)

    return np.clip(img, 0, 255).astype(np.uint8)


def generate_pleural_effusion_xray(size: int = IMG_SIZE, seed: int | None = None) -> np.ndarray:
    """
    Pleural Effusion: bottom 25% of one lung field has a uniform bright
    opacity (fluid level blunting the costophrenic angle).
    """
    rng = np.random.default_rng(seed)
    img = _base_xray(rng, size)
    img, left_mask, _ = _add_lung_ovals(rng, img, size)
    img = _add_cardiac_silhouette(rng, img, size=size)

    # Fluid in left lower zone (bottom 25% of left lung field)
    cutoff_y = int(size * 0.68)
    fluid_region = np.zeros((size, size), dtype=bool)
    fluid_region[cutoff_y:, :90] = True
    fluid_region = fluid_region & left_mask
    img[fluid_region] += rng.normal(55, 8, img[fluid_region].shape)

    return np.clip(img, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Dispatch table
# ---------------------------------------------------------------------------

GENERATORS = {
    "Normal": generate_normal_xray,
    "Pneumonia": generate_pneumonia_xray,
    "Cardiomegaly": generate_cardiomegaly_xray,
    "Pleural Effusion": generate_pleural_effusion_xray,
}


# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------

def generate_dataset(
    output_dir: str,
    images_per_class: int = 200,
    train_ratio: float = 0.8,
    seed: int = 42,
) -> dict:
    """
    Generate a full synthetic dataset on disk.

    Layout::

        output_dir/
          train/
            Normal/        *.png
            Pneumonia/     *.png
            ...
          val/
            Normal/        *.png
            ...
          metadata.json

    Returns a dict with split counts.
    """
    random.seed(seed)
    np.random.seed(seed)

    n_train = int(images_per_class * train_ratio)
    n_val = images_per_class - n_train
    metadata: dict = {"classes": CLASSES, "class_to_idx": CLASS_TO_IDX, "splits": {}}

    for split, n in [("train", n_train), ("val", n_val)]:
        metadata["splits"][split] = {}
        for cls_name in CLASSES:
            cls_dir = os.path.join(output_dir, split, cls_name)
            os.makedirs(cls_dir, exist_ok=True)
            gen_fn = GENERATORS[cls_name]

            offset = 0 if split == "train" else n_train
            for i in range(n):
                img_arr = gen_fn(IMG_SIZE, seed=seed + offset + i)
                img = Image.fromarray(img_arr, mode="L")
                img.save(os.path.join(cls_dir, f"{cls_name.replace(' ', '_')}_{offset + i:04d}.png"))

            metadata["splits"][split][cls_name] = n

    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Dataset written to {output_dir}")
    print(f"  train: {n_train}/class  val: {n_val}/class  classes: {CLASSES}")
    return metadata


# ---------------------------------------------------------------------------
# PyTorch Dataset wrapper
# ---------------------------------------------------------------------------

def get_dataset_class():
    """Return XRayDataset class (lazy import to avoid hard torch dependency)."""
    from torch.utils.data import Dataset
    from torchvision import transforms

    class XRayDataset(Dataset):
        """Grayscale X-ray dataset loaded from disk."""

        DEFAULT_TRANSFORM = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])

        def __init__(self, root: str, split: str = "train", transform=None):
            self.root = root
            self.split = split
            self.transform = transform or self.DEFAULT_TRANSFORM
            self.samples: list[tuple[str, int]] = []

            for cls_name, cls_idx in CLASS_TO_IDX.items():
                cls_dir = os.path.join(root, split, cls_name)
                if not os.path.isdir(cls_dir):
                    continue
                for fname in sorted(os.listdir(cls_dir)):
                    if fname.lower().endswith(".png"):
                        self.samples.append((os.path.join(cls_dir, fname), cls_idx))

        def __len__(self) -> int:
            return len(self.samples)

        def __getitem__(self, idx: int):
            path, label = self.samples[idx]
            img = Image.open(path).convert("L")
            if self.transform:
                img = self.transform(img)
            return img, label

    return XRayDataset


if __name__ == "__main__":
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        generate_dataset(tmp, images_per_class=4)
        print("Smoke test passed.")
