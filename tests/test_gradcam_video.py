"""
Tests for explainability/gradcam_video.py
"""

import os
import sys
import tempfile

import numpy as np
import pytest
import torch
from PIL import Image

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from model.cnn_classifier import build_model
from models.unet import build_unet
from explainability.gradcam_video import (
    ScoreCAM,
    GradCAMSequence,
    make_side_by_side,
)


@pytest.fixture(scope="module")
def cls_model():
    m = build_model()
    m.eval()
    return m


@pytest.fixture(scope="module")
def seg_model():
    m = build_unet()
    m.eval()
    return m


@pytest.fixture(scope="module")
def small_tensor():
    return torch.randn(1, 1, 64, 64)


# ---------------------------------------------------------------------------
# ScoreCAM
# ---------------------------------------------------------------------------

class TestScoreCAM:
    def test_returns_float32_array(self, cls_model, small_tensor):
        with ScoreCAM(cls_model, batch_size=4) as scam:
            hm = scam.compute(small_tensor)
        assert isinstance(hm, np.ndarray)
        assert hm.dtype == np.float32

    def test_heatmap_shape_matches_input(self, cls_model, small_tensor):
        with ScoreCAM(cls_model, batch_size=4) as scam:
            hm = scam.compute(small_tensor)
        assert hm.shape == (64, 64)

    def test_heatmap_values_in_0_1(self, cls_model, small_tensor):
        with ScoreCAM(cls_model, batch_size=4) as scam:
            hm = scam.compute(small_tensor)
        assert hm.min() >= 0.0 - 1e-6
        assert hm.max() <= 1.0 + 1e-6

    def test_explicit_target_class(self, cls_model, small_tensor):
        for cls in range(4):
            with ScoreCAM(cls_model, batch_size=4) as scam:
                hm = scam.compute(small_tensor, target_class=cls)
            assert hm.shape == (64, 64)

    def test_no_gradient_required(self, cls_model, small_tensor):
        """ScoreCAM must not require grad on the input tensor."""
        x = small_tensor.detach()
        assert not x.requires_grad
        with ScoreCAM(cls_model, batch_size=4) as scam:
            hm = scam.compute(x)
        assert isinstance(hm, np.ndarray)

    def test_hook_removed_after_context(self, cls_model, small_tensor):
        with ScoreCAM(cls_model, batch_size=4):
            pass
        # Model should still work normally after hook removal
        with torch.no_grad():
            out = cls_model(small_tensor)
        assert out.shape == (1, 4)


# ---------------------------------------------------------------------------
# GradCAMSequence
# ---------------------------------------------------------------------------

class TestGradCAMSequence:
    def test_generate_frames_count(self, cls_model):
        seq = GradCAMSequence(cls_model, n_slices=5, size=64, seed=0)
        frames = seq.generate_frames()
        assert len(frames) == 5

    def test_frames_are_pil_images(self, cls_model):
        seq = GradCAMSequence(cls_model, n_slices=3, size=64, seed=0)
        frames = seq.generate_frames()
        for f in frames:
            assert isinstance(f, Image.Image)

    def test_frames_correct_size(self, cls_model):
        seq = GradCAMSequence(cls_model, n_slices=3, size=64, seed=0)
        frames = seq.generate_frames()
        for f in frames:
            assert f.size == (64, 64)

    def test_save_gif_creates_file(self, cls_model):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "test.gif")
            seq = GradCAMSequence(cls_model, n_slices=3, size=32, seed=1)
            out = seq.save_gif(path, fps=2)
            assert os.path.exists(out)
            assert os.path.getsize(out) > 0

    def test_save_gif_returns_path(self, cls_model):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "ret.gif")
            seq = GradCAMSequence(cls_model, n_slices=2, size=32, seed=0)
            result = seq.save_gif(path)
            assert result == path


# ---------------------------------------------------------------------------
# Side-by-side panel
# ---------------------------------------------------------------------------

class TestMakeSideBySide:
    def test_output_is_pil(self, cls_model, small_tensor):
        panel = make_side_by_side(small_tensor, cls_model, size=64)
        assert isinstance(panel, Image.Image)

    def test_output_width_4x_size(self, cls_model, small_tensor):
        panel = make_side_by_side(small_tensor, cls_model, size=64)
        assert panel.size == (64 * 4, 64)

    def test_with_seg_model(self, cls_model, seg_model, small_tensor):
        import torch.nn.functional as F
        x = F.interpolate(small_tensor, size=(64, 64))
        panel = make_side_by_side(x, cls_model, seg_model=seg_model, size=64)
        assert panel.size == (256, 64)

    def test_without_seg_model(self, cls_model, small_tensor):
        panel = make_side_by_side(small_tensor, cls_model, seg_model=None, size=64)
        assert panel.size == (64 * 4, 64)
