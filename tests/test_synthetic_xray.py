"""
Tests for data/synthetic_xray.py
"""

import os
import sys
import tempfile

import numpy as np
import pytest
from PIL import Image

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from data.synthetic_xray import (
    CLASSES,
    GENERATORS,
    IMG_SIZE,
    generate_cardiomegaly_xray,
    generate_normal_xray,
    generate_pleural_effusion_xray,
    generate_pneumonia_xray,
    generate_dataset,
)


# ---------------------------------------------------------------------------
# Per-generator tests
# ---------------------------------------------------------------------------

class TestGenerators:
    @pytest.mark.parametrize("fn", list(GENERATORS.values()))
    def test_output_shape(self, fn):
        img = fn(224, seed=0)
        assert img.shape == (224, 224), f"Expected (224,224), got {img.shape}"

    @pytest.mark.parametrize("fn", list(GENERATORS.values()))
    def test_output_dtype(self, fn):
        img = fn(224, seed=0)
        assert img.dtype == np.uint8, f"Expected uint8, got {img.dtype}"

    @pytest.mark.parametrize("fn", list(GENERATORS.values()))
    def test_pixel_range(self, fn):
        img = fn(224, seed=0)
        assert img.min() >= 0
        assert img.max() <= 255

    def test_normal_is_deterministic(self):
        a = generate_normal_xray(seed=42)
        b = generate_normal_xray(seed=42)
        assert np.array_equal(a, b)

    def test_different_seeds_differ(self):
        a = generate_pneumonia_xray(seed=1)
        b = generate_pneumonia_xray(seed=2)
        assert not np.array_equal(a, b)

    def test_pneumonia_has_brighter_lung(self):
        """Consolidation patch should raise mean brightness vs normal."""
        normal = generate_normal_xray(seed=10).astype(float)
        pneumonia = generate_pneumonia_xray(seed=10).astype(float)
        # Left lung area (consolidation zone)
        region_slice = (slice(110, 170), slice(40, 90))
        assert pneumonia[region_slice].mean() > normal[region_slice].mean()

    def test_cardiomegaly_wider_cardiac(self):
        """Cardiac silhouette columns should be brighter in cardiomegaly."""
        normal = generate_normal_xray(seed=10).astype(float)
        cardio = generate_cardiomegaly_xray(seed=10).astype(float)
        cardiac_col_slice = (slice(90, 150), slice(80, 130))
        assert cardio[cardiac_col_slice].mean() >= normal[cardiac_col_slice].mean()

    def test_effusion_brighter_base(self):
        """Bottom-left lung zone should be brighter in effusion."""
        normal = generate_normal_xray(seed=10).astype(float)
        effusion = generate_pleural_effusion_xray(seed=10).astype(float)
        base_slice = (slice(155, 185), slice(30, 85))
        assert effusion[base_slice].mean() > normal[base_slice].mean()


# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------

class TestGenerateDataset:
    def test_creates_expected_dirs(self):
        with tempfile.TemporaryDirectory() as tmp:
            generate_dataset(tmp, images_per_class=4, train_ratio=0.75)
            for split in ("train", "val"):
                for cls in CLASSES:
                    assert os.path.isdir(os.path.join(tmp, split, cls))

    def test_image_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            meta = generate_dataset(tmp, images_per_class=4, train_ratio=0.5)
            for cls in CLASSES:
                assert meta["splits"]["train"][cls] == 2
                assert meta["splits"]["val"][cls] == 2

    def test_images_are_valid_png(self):
        with tempfile.TemporaryDirectory() as tmp:
            generate_dataset(tmp, images_per_class=2, train_ratio=0.5)
            for split in ("train", "val"):
                for cls in CLASSES:
                    folder = os.path.join(tmp, split, cls)
                    files = os.listdir(folder)
                    assert len(files) == 1
                    img = Image.open(os.path.join(folder, files[0]))
                    assert img.size == (IMG_SIZE, IMG_SIZE)


# ---------------------------------------------------------------------------
# Dataset wrapper
# ---------------------------------------------------------------------------

class TestXRayDataset:
    def test_len_and_getitem(self):
        from data.synthetic_xray import get_dataset_class
        XRayDataset = get_dataset_class()
        with tempfile.TemporaryDirectory() as tmp:
            # images_per_class=4, train_ratio=0.5 → 2 train images per class
            generate_dataset(tmp, images_per_class=4, train_ratio=0.5)
            ds = XRayDataset(tmp, split="train")
            assert len(ds) == 2 * len(CLASSES)
            img, label = ds[0]
            import torch
            assert isinstance(img, torch.Tensor)
            assert img.shape == (1, IMG_SIZE, IMG_SIZE)
            assert 0 <= int(label) < len(CLASSES)
