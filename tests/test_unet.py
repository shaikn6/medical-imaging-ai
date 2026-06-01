"""
Tests for models/unet.py
"""

import os
import sys

import pytest
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from models.unet import (
    UNet,
    build_unet,
    DiceBCELoss,
    SyntheticSegDataset,
    dice_coefficient,
    train_unet_demo,
)


# ---------------------------------------------------------------------------
# Model shape tests
# ---------------------------------------------------------------------------

class TestUNetShapes:
    @pytest.fixture(scope="class")
    def model(self):
        m = build_unet()
        m.eval()
        return m

    def test_build_unet_type(self):
        assert isinstance(build_unet(), UNet)

    def test_output_shape_256(self, model):
        x = torch.randn(1, 1, 256, 256)
        out = model(x)
        assert out.shape == (1, 1, 256, 256), f"Got {out.shape}"

    def test_output_shape_224(self, model):
        x = torch.randn(2, 1, 224, 224)
        out = model(x)
        assert out.shape == (2, 1, 224, 224)

    def test_output_shape_batch_4(self, model):
        x = torch.randn(4, 1, 256, 256)
        out = model(x)
        assert out.shape == (4, 1, 256, 256)

    def test_output_is_logit_unbounded(self, model):
        """Output should NOT be bounded to [0,1] — it's a raw logit."""
        x = torch.randn(2, 1, 256, 256)
        with torch.no_grad():
            out = model(x)
        # At least some values should be outside [0,1] for raw logit
        prob = torch.sigmoid(out)
        assert prob.min() >= 0.0 and prob.max() <= 1.0 + 1e-6

    def test_predict_mask_shape(self, model):
        x = torch.randn(2, 1, 256, 256)
        mask = model.predict_mask(x)
        assert mask.shape == (2, 1, 256, 256)

    def test_predict_mask_binary(self, model):
        x = torch.randn(2, 1, 256, 256)
        mask = model.predict_mask(x)
        unique = torch.unique(mask)
        assert set(unique.tolist()).issubset({0.0, 1.0})

    def test_no_nan_in_output(self, model):
        x = torch.randn(2, 1, 256, 256)
        out = model(x)
        assert not torch.isnan(out).any()

    def test_spatial_output_equals_input(self, model):
        """Encoder + decoder should preserve spatial dimensions."""
        for h, w in [(128, 128), (256, 256), (224, 224)]:
            x = torch.randn(1, 1, h, w)
            out = model(x)
            assert out.shape[-2:] == (h, w), f"Mismatch: input={h}x{w}, output={out.shape[-2:]}"


# ---------------------------------------------------------------------------
# Loss tests
# ---------------------------------------------------------------------------

class TestDiceBCELoss:
    def test_loss_positive(self):
        criterion = DiceBCELoss()
        logit  = torch.randn(4, 1, 64, 64)
        target = (torch.rand(4, 1, 64, 64) > 0.5).float()
        loss = criterion(logit, target)
        assert float(loss) > 0

    def test_perfect_prediction_low_loss(self):
        criterion = DiceBCELoss()
        target = torch.ones(2, 1, 32, 32)
        # Large positive logit → sigmoid ≈ 1 ≈ target
        logit  = torch.full((2, 1, 32, 32), 10.0)
        loss   = criterion(logit, target)
        assert float(loss) < 0.1, f"Expected low loss, got {loss.item():.4f}"

    def test_loss_is_scalar(self):
        criterion = DiceBCELoss()
        logit  = torch.randn(2, 1, 16, 16)
        target = torch.zeros(2, 1, 16, 16)
        loss = criterion(logit, target)
        assert loss.shape == ()


# ---------------------------------------------------------------------------
# Dice coefficient
# ---------------------------------------------------------------------------

class TestDiceCoefficient:
    def test_perfect_overlap(self):
        logit  = torch.full((1, 1, 16, 16), 10.0)  # sigmoid ≈ 1
        target = torch.ones(1, 1, 16, 16)
        d = dice_coefficient(logit, target)
        assert d > 0.99, f"Expected ~1.0, got {d:.4f}"

    def test_no_overlap(self):
        logit  = torch.full((1, 1, 16, 16), -10.0)  # sigmoid ≈ 0
        target = torch.ones(1, 1, 16, 16)
        d = dice_coefficient(logit, target)
        assert d < 0.1, f"Expected ~0, got {d:.4f}"

    def test_dice_in_range(self):
        logit  = torch.randn(4, 1, 32, 32)
        target = (torch.rand(4, 1, 32, 32) > 0.5).float()
        d = dice_coefficient(logit, target)
        assert 0.0 <= d <= 1.0 + 1e-6


# ---------------------------------------------------------------------------
# Synthetic dataset
# ---------------------------------------------------------------------------

class TestSyntheticSegDataset:
    def test_len(self):
        ds = SyntheticSegDataset(n_samples=20, seed=0)
        assert len(ds) == 20

    def test_item_shapes(self):
        ds = SyntheticSegDataset(n_samples=4, seed=0)
        img, mask = ds[0]
        assert img.shape  == (1, 256, 256)
        assert mask.shape == (1, 256, 256)

    def test_image_range(self):
        ds = SyntheticSegDataset(n_samples=4, seed=0)
        img, _ = ds[0]
        # Normalised: approx [-1, 1] after (x - 0.5)/0.5
        assert float(img.min()) >= -2.0
        assert float(img.max()) <= 2.0

    def test_mask_binary(self):
        ds = SyntheticSegDataset(n_samples=10, seed=0)
        for i in range(10):
            _, mask = ds[i]
            unique = torch.unique(mask)
            assert set(unique.tolist()).issubset({0.0, 1.0})

    def test_mask_has_positive_pixels(self):
        """At least some masks should have active lesion pixels."""
        ds = SyntheticSegDataset(n_samples=20, seed=0)
        any_positive = False
        for i in range(20):
            _, mask = ds[i]
            if mask.sum() > 0:
                any_positive = True
                break
        assert any_positive


# ---------------------------------------------------------------------------
# Training demo (fast — few epochs, small dataset)
# ---------------------------------------------------------------------------

class TestTrainUNetDemo:
    def test_demo_returns_dict(self):
        result = train_unet_demo(n_samples=40, epochs=2, batch_size=8, seed=0)
        assert "test_dice" in result
        assert "train_dice_history" in result
        assert "model" in result

    def test_demo_dice_positive(self):
        result = train_unet_demo(n_samples=40, epochs=2, batch_size=8, seed=0)
        assert result["test_dice"] > 0.0

    def test_demo_model_is_unet(self):
        result = train_unet_demo(n_samples=20, epochs=1, batch_size=8, seed=0)
        assert isinstance(result["model"], UNet)
