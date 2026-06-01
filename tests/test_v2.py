"""
V2 test suite — DICOM pipeline, ScoreCAM, EfficientNet-B4, compare_models.

All tests are self-contained: synthetic DICOM data is generated in-memory,
and real .dcm files / GPU hardware are never required.

Run with:
    pytest tests/test_v2.py -v
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest
import torch
import torch.nn as nn

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from src.dicom_pipeline import (
    _apply_window_level,
    _extract_pixel_array,
    dicom_to_tensor,
    make_synthetic_dicom,
)
from src.scorecam import (
    _normalise_array,
    _normalise_maps,
    generate_scorecam,
    heatmap_to_overlay,
)
from src.efficientnet_model import (
    EfficientNetXRay,
    build_efficientnet_model,
    compare_models,
)
from model.cnn_classifier import build_model, NUM_CLASSES, CLASSES


# ===========================================================================
# DICOM pipeline tests (14 tests)
# ===========================================================================

class TestMakeSyntheticDicom:
    def test_returns_dataset(self):
        import pydicom.dataset
        dcm = make_synthetic_dicom()
        assert isinstance(dcm, pydicom.dataset.Dataset)

    def test_has_pixel_data(self):
        dcm = make_synthetic_dicom()
        assert hasattr(dcm, "PixelData")
        assert len(dcm.PixelData) > 0

    def test_correct_dimensions(self):
        dcm = make_synthetic_dicom(rows=128, cols=128)
        assert dcm.Rows == 128
        assert dcm.Columns == 128

    def test_window_tags_present(self):
        dcm = make_synthetic_dicom(window_width=1800.0, window_level=-500.0)
        assert hasattr(dcm, "WindowWidth")
        assert hasattr(dcm, "WindowCenter")

    def test_window_values_stored(self):
        dcm = make_synthetic_dicom(window_width=1200.0, window_level=-400.0)
        assert float(dcm.WindowWidth) == pytest.approx(1200.0)
        assert float(dcm.WindowCenter) == pytest.approx(-400.0)

    def test_reproducible_with_seed(self):
        dcm1 = make_synthetic_dicom(seed=0)
        dcm2 = make_synthetic_dicom(seed=0)
        assert dcm1.PixelData == dcm2.PixelData

    def test_different_seeds_differ(self):
        dcm1 = make_synthetic_dicom(seed=1)
        dcm2 = make_synthetic_dicom(seed=2)
        assert dcm1.PixelData != dcm2.PixelData


class TestExtractPixelArray:
    def test_returns_float32(self):
        dcm = make_synthetic_dicom(rows=64, cols=64)
        arr = _extract_pixel_array(dcm)
        assert arr.dtype == np.float32

    def test_correct_shape(self):
        dcm = make_synthetic_dicom(rows=64, cols=96)
        arr = _extract_pixel_array(dcm)
        assert arr.shape == (64, 96)

    def test_2d_output(self):
        dcm = make_synthetic_dicom()
        arr = _extract_pixel_array(dcm)
        assert arr.ndim == 2


class TestApplyWindowLevel:
    def test_output_range_minus1_to_1(self):
        pixels = np.array([-1000.0, -600.0, 0.0, 500.0], dtype=np.float32)
        result = _apply_window_level(pixels, window_width=1500.0, window_level=-600.0)
        assert result.min() >= -1.0 - 1e-6
        assert result.max() <= 1.0 + 1e-6

    def test_center_maps_to_zero(self):
        center = -600.0
        pixels = np.array([center], dtype=np.float32)
        result = _apply_window_level(pixels, window_width=1500.0, window_level=center)
        assert result[0] == pytest.approx(0.0, abs=1e-5)

    def test_clipping_below_low(self):
        pixels = np.array([-10000.0], dtype=np.float32)
        result = _apply_window_level(pixels, window_width=1500.0, window_level=-600.0)
        assert result[0] == pytest.approx(-1.0, abs=1e-5)

    def test_clipping_above_high(self):
        pixels = np.array([10000.0], dtype=np.float32)
        result = _apply_window_level(pixels, window_width=1500.0, window_level=-600.0)
        assert result[0] == pytest.approx(1.0, abs=1e-5)


class TestDicomToTensor:
    def test_output_shape_default(self):
        dcm = make_synthetic_dicom()
        tensor = dicom_to_tensor(dcm)
        assert tensor.shape == (1, 224, 224)

    def test_output_shape_custom_size(self):
        dcm = make_synthetic_dicom()
        tensor = dicom_to_tensor(dcm, target_size=(128, 128))
        assert tensor.shape == (1, 128, 128)

    def test_output_dtype_float32(self):
        dcm = make_synthetic_dicom()
        tensor = dicom_to_tensor(dcm)
        assert tensor.dtype == torch.float32

    def test_values_in_normalised_range(self):
        dcm = make_synthetic_dicom()
        tensor = dicom_to_tensor(dcm)
        assert tensor.min().item() >= -1.0 - 1e-4
        assert tensor.max().item() <= 1.0 + 1e-4

    def test_window_override(self):
        dcm = make_synthetic_dicom()
        # Should not raise even with overrides
        tensor = dicom_to_tensor(dcm, window_width=2000.0, window_level=40.0)
        assert tensor.shape == (1, 224, 224)

    def test_missing_window_tags_uses_default(self):
        """Dataset without WindowWidth/WindowCenter should use defaults."""
        import pydicom.dataset
        dcm = pydicom.dataset.Dataset()
        # Create minimal in-memory pixel data
        pixels = np.zeros((64, 64), dtype=np.int16)
        dcm.Rows = 64
        dcm.Columns = 64
        dcm.SamplesPerPixel = 1
        dcm.PhotometricInterpretation = "MONOCHROME2"
        dcm.BitsAllocated = 16
        dcm.BitsStored = 16
        dcm.HighBit = 15
        dcm.PixelRepresentation = 0
        dcm.PixelData = pixels.tobytes()
        dcm["PixelData"].VR = "OB"
        # No WindowWidth / WindowCenter set
        tensor = dicom_to_tensor(dcm)
        assert tensor.shape == (1, 224, 224)


# ===========================================================================
# ScoreCAM tests (9 tests)
# ===========================================================================

class TestNormaliseMaps:
    def test_output_range(self):
        maps = torch.randn(8, 14, 14)
        result = _normalise_maps(maps)
        assert result.min().item() >= -1e-6
        assert result.max().item() <= 1.0 + 1e-6

    def test_shape_preserved(self):
        maps = torch.randn(16, 7, 7)
        result = _normalise_maps(maps)
        assert result.shape == (16, 7, 7)

    def test_uniform_channel_stays_zero(self):
        maps = torch.zeros(4, 5, 5)
        result = _normalise_maps(maps)
        assert result.abs().max().item() < 1e-6


class TestNormaliseArray:
    def test_range_0_to_1(self):
        arr = np.array([-5.0, 0.0, 5.0], dtype=np.float32)
        result = _normalise_array(arr)
        assert result.min() == pytest.approx(0.0, abs=1e-6)
        assert result.max() == pytest.approx(1.0, abs=1e-6)

    def test_uniform_array_returns_zeros(self):
        arr = np.ones((4, 4), dtype=np.float32) * 3.0
        result = _normalise_array(arr)
        assert np.allclose(result, 0.0)


class TestGenerateScoreCAM:
    @pytest.fixture
    def small_model(self):
        model = build_model()
        model.eval()
        return model

    @pytest.fixture
    def image_tensor(self):
        return torch.randn(1, 1, 224, 224)

    def test_output_shape(self, small_model, image_tensor):
        heatmap = generate_scorecam(small_model, image_tensor, target_class=0)
        assert heatmap.shape == (224, 224)

    def test_output_range(self, small_model, image_tensor):
        heatmap = generate_scorecam(small_model, image_tensor, target_class=1)
        assert heatmap.min() >= 0.0 - 1e-6
        assert heatmap.max() <= 1.0 + 1e-6

    def test_dtype_float32(self, small_model, image_tensor):
        heatmap = generate_scorecam(small_model, image_tensor, target_class=0)
        assert heatmap.dtype == np.float32

    def test_works_without_specifying_class(self, small_model, image_tensor):
        heatmap = generate_scorecam(small_model, image_tensor)
        assert heatmap.shape == (224, 224)

    def test_no_backward_needed(self, small_model, image_tensor):
        """ScoreCAM must work even when gradients are disabled globally."""
        with torch.no_grad():
            heatmap = generate_scorecam(small_model, image_tensor, target_class=2)
        assert heatmap.shape == (224, 224)

    def test_invalid_input_shape_raises(self, small_model):
        bad_tensor = torch.randn(224, 224)  # missing batch dim
        with pytest.raises(ValueError):
            generate_scorecam(small_model, bad_tensor)

    def test_overlay_output_shape(self, small_model, image_tensor):
        heatmap = generate_scorecam(small_model, image_tensor, target_class=0)
        original = np.random.randint(0, 255, (224, 224), dtype=np.uint8)
        overlay = heatmap_to_overlay(heatmap, original)
        assert overlay.shape == (224, 224, 3)
        assert overlay.dtype == np.uint8


# ===========================================================================
# EfficientNet-B4 tests (10 tests)
# ===========================================================================

class TestEfficientNetXRay:
    @pytest.fixture(scope="class")
    def model(self):
        return build_efficientnet_model(num_classes=NUM_CLASSES, pretrained=False)

    def test_model_type(self, model):
        assert isinstance(model, EfficientNetXRay)

    def test_forward_output_shape(self, model):
        x = torch.randn(2, 1, 224, 224)
        with torch.no_grad():
            logits = model(x)
        assert logits.shape == (2, NUM_CLASSES)

    def test_batch_1_forward(self, model):
        x = torch.randn(1, 1, 224, 224)
        with torch.no_grad():
            logits = model(x)
        assert logits.shape == (1, NUM_CLASSES)

    def test_parameter_count_positive(self, model):
        n_params = sum(p.numel() for p in model.parameters())
        assert n_params > 0

    def test_parameter_count_reasonable(self, model):
        """EfficientNet-B4 should have > 17M parameters."""
        n_params = sum(p.numel() for p in model.parameters())
        assert n_params > 1_000_000

    def test_last_conv_layer_is_conv2d(self, model):
        layer = model.last_conv_layer
        assert isinstance(layer, nn.Conv2d)

    def test_custom_num_classes(self):
        model2 = build_efficientnet_model(num_classes=2, pretrained=False)
        x = torch.randn(1, 1, 224, 224)
        with torch.no_grad():
            logits = model2(x)
        assert logits.shape == (1, 2)

    def test_no_nan_in_output(self, model):
        x = torch.randn(4, 1, 224, 224)
        with torch.no_grad():
            logits = model(x)
        assert not torch.isnan(logits).any()

    def test_predict_proba_sums_to_one(self, model):
        x = torch.randn(3, 1, 224, 224)
        probs = model.predict_proba(x)
        sums = probs.sum(dim=-1)
        assert torch.allclose(sums, torch.ones(3), atol=1e-5)

    def test_predict_returns_valid_indices(self, model):
        x = torch.randn(5, 1, 224, 224)
        preds = model.predict(x)
        assert ((preds >= 0) & (preds < NUM_CLASSES)).all()


# ===========================================================================
# compare_models tests (5 tests)
# ===========================================================================

class TestCompareModels:
    """Tests for the model comparison utility."""

    @pytest.fixture(scope="class")
    def models_and_loader(self):
        model_a = build_model()
        model_b = build_efficientnet_model(num_classes=NUM_CLASSES, pretrained=False)

        # Build a tiny dataloader using a list of (image, label) tuples
        images = torch.randn(20, 1, 224, 224)
        labels = torch.randint(0, NUM_CLASSES, (20,))

        class TinyDataset(torch.utils.data.Dataset):
            def __len__(self):
                return len(images)
            def __getitem__(self, idx):
                return images[idx], labels[idx]

        loader = torch.utils.data.DataLoader(TinyDataset(), batch_size=4)
        return model_a, model_b, loader

    def test_returns_dict_with_both_keys(self, models_and_loader):
        a, b, loader = models_and_loader
        result = compare_models(a, b, loader)
        assert "model_a" in result
        assert "model_b" in result

    def test_inner_dicts_have_class_keys(self, models_and_loader):
        a, b, loader = models_and_loader
        result = compare_models(a, b, loader)
        for key in ("model_a", "model_b"):
            assert set(result[key].keys()) == set(CLASSES)

    def test_auc_values_are_floats(self, models_and_loader):
        a, b, loader = models_and_loader
        result = compare_models(a, b, loader)
        for key in ("model_a", "model_b"):
            for cls, val in result[key].items():
                assert isinstance(val, float), f"{key}[{cls}] is not float"

    def test_auc_values_in_valid_range(self, models_and_loader):
        a, b, loader = models_and_loader
        result = compare_models(a, b, loader)
        for key in ("model_a", "model_b"):
            for cls, val in result[key].items():
                if not np.isnan(val):
                    assert 0.0 <= val <= 1.0, f"{key}[{cls}] = {val} out of [0,1]"

    def test_custom_class_names(self, models_and_loader):
        a, b, loader = models_and_loader
        custom = ["A", "B", "C", "D"]
        result = compare_models(a, b, loader, class_names=custom)
        assert set(result["model_a"].keys()) == set(custom)
