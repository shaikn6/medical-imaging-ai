"""
Tests for model/gradcam.py
"""

import os
import sys

import numpy as np
import pytest
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from model.cnn_classifier import build_model, NUM_CLASSES
from model.gradcam import (
    GradCAM,
    compute_gradcam,
    heatmap_to_rgb,
    overlay_heatmap,
)


@pytest.fixture(scope="module")
def model_and_input():
    model = build_model()
    model.eval()
    x = torch.randn(1, 1, 224, 224)
    return model, x


class TestGradCAM:
    def test_compute_returns_float32_array(self, model_and_input):
        model, x = model_and_input
        with GradCAM(model) as cam:
            heatmap = cam.compute(x)
        assert isinstance(heatmap, np.ndarray)
        assert heatmap.dtype == np.float32

    def test_heatmap_shape_matches_input(self, model_and_input):
        model, x = model_and_input
        with GradCAM(model) as cam:
            heatmap = cam.compute(x)
        assert heatmap.shape == (224, 224)

    def test_heatmap_values_in_range(self, model_and_input):
        model, x = model_and_input
        with GradCAM(model) as cam:
            heatmap = cam.compute(x)
        assert heatmap.min() >= 0.0 - 1e-6
        assert heatmap.max() <= 1.0 + 1e-6

    def test_explicit_target_class(self, model_and_input):
        model, x = model_and_input
        for cls_idx in range(NUM_CLASSES):
            with GradCAM(model) as cam:
                heatmap = cam.compute(x, target_class=cls_idx)
            assert heatmap.shape == (224, 224)

    def test_different_classes_produce_different_heatmaps(self, model_and_input):
        model, x = model_and_input
        heatmaps = []
        for cls_idx in range(NUM_CLASSES):
            with GradCAM(model) as cam:
                h = cam.compute(x, target_class=cls_idx)
            heatmaps.append(h)
        # At least one pair should differ
        all_equal = all(np.array_equal(heatmaps[0], h) for h in heatmaps[1:])
        assert not all_equal

    def test_compute_gradcam_convenience(self, model_and_input):
        model, x = model_and_input
        heatmap = compute_gradcam(model, x)
        assert isinstance(heatmap, np.ndarray)
        assert heatmap.shape == (224, 224)

    def test_hooks_removed_after_context(self, model_and_input):
        model, x = model_and_input
        with GradCAM(model) as cam:
            pass  # exit context removes hooks
        # Should still be able to do a clean forward pass
        with torch.no_grad():
            logits = model(x)
        assert logits.shape == (1, NUM_CLASSES)

    def test_no_hook_leak_across_instances(self, model_and_input):
        """Multiple sequential GradCAM instances should not accumulate hooks."""
        model, x = model_and_input
        for _ in range(3):
            with GradCAM(model) as cam:
                cam.compute(x, target_class=0)
        # Still works
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, NUM_CLASSES)


class TestHeatmapToRgb:
    def test_output_shape(self):
        hm = np.random.rand(224, 224).astype(np.float32)
        rgb = heatmap_to_rgb(hm)
        assert rgb.shape == (224, 224, 3)

    def test_output_uint8(self):
        hm = np.zeros((224, 224), dtype=np.float32)
        rgb = heatmap_to_rgb(hm)
        assert rgb.dtype == np.uint8

    def test_range_0_255(self):
        hm = np.random.rand(64, 64).astype(np.float32)
        rgb = heatmap_to_rgb(hm)
        assert rgb.min() >= 0
        assert rgb.max() <= 255


class TestOverlayHeatmap:
    def test_output_shape_from_ndarray(self):
        gray = np.random.randint(0, 255, (224, 224), dtype=np.uint8)
        hm = np.random.rand(224, 224).astype(np.float32)
        out = overlay_heatmap(gray, hm)
        assert out.shape == (224, 224, 3)
        assert out.dtype == np.uint8

    def test_output_shape_from_pil(self):
        from PIL import Image
        pil = Image.fromarray(np.zeros((224, 224), dtype=np.uint8), mode="L")
        hm = np.ones((224, 224), dtype=np.float32)
        out = overlay_heatmap(pil, hm)
        assert out.shape == (224, 224, 3)

    def test_mismatched_sizes_handled(self):
        gray = np.zeros((224, 224), dtype=np.uint8)
        hm = np.ones((28, 28), dtype=np.float32)
        out = overlay_heatmap(gray, hm)
        assert out.shape == (224, 224, 3)

    def test_all_zero_heatmap_no_crash(self):
        gray = np.full((224, 224), 100, dtype=np.uint8)
        hm = np.zeros((224, 224), dtype=np.float32)
        out = overlay_heatmap(gray, hm)
        assert out.shape == (224, 224, 3)
