"""
Tests for preprocessing/dicom_handler.py
"""

import os
import sys

import numpy as np
import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from preprocessing.dicom_handler import (
    DicomMetadata,
    generate_synthetic_dicom,
    apply_windowing,
    apply_hu_windowing,
    dicom_to_model_input,
    preprocess_dicom,
    _CT_WINDOW_PRESETS,
)


# ---------------------------------------------------------------------------
# Metadata generation
# ---------------------------------------------------------------------------

class TestGenerateSyntheticDicom:
    @pytest.mark.parametrize("modality", ["CT", "CR", "DX"])
    def test_returns_dicom_metadata(self, modality):
        ds = generate_synthetic_dicom(modality=modality, seed=0)
        assert isinstance(ds, DicomMetadata)

    @pytest.mark.parametrize("modality", ["CT", "CR", "DX"])
    def test_modality_field(self, modality):
        ds = generate_synthetic_dicom(modality=modality, seed=0)
        assert ds.Modality == modality

    def test_patient_id_non_empty(self):
        ds = generate_synthetic_dicom(seed=0)
        assert len(ds.PatientID) > 0

    def test_study_date_format(self):
        ds = generate_synthetic_dicom(seed=1)
        assert len(ds.StudyDate) == 8
        assert ds.StudyDate.isdigit()

    def test_pixel_array_shape_ct(self):
        ds = generate_synthetic_dicom(modality="CT", size=128, seed=0)
        assert ds.pixel_array.shape == (128, 128)

    def test_pixel_array_shape_cr(self):
        ds = generate_synthetic_dicom(modality="CR", size=64, seed=0)
        assert ds.pixel_array.shape == (64, 64)

    def test_window_center_width_positive(self):
        ds = generate_synthetic_dicom(seed=0)
        assert ds.WindowWidth > 0

    def test_ct_rescale_intercept(self):
        ds = generate_synthetic_dicom(modality="CT", seed=0)
        assert ds.RescaleIntercept == -1024.0
        assert ds.RescaleSlope == 1.0

    def test_pixel_spacing_two_values(self):
        ds = generate_synthetic_dicom(seed=0)
        assert len(ds.PixelSpacing) == 2

    def test_different_seeds_differ(self):
        ds1 = generate_synthetic_dicom(seed=1)
        ds2 = generate_synthetic_dicom(seed=2)
        assert ds1.PatientID != ds2.PatientID

    def test_reproducible_with_same_seed(self):
        ds1 = generate_synthetic_dicom(seed=42)
        ds2 = generate_synthetic_dicom(seed=42)
        assert ds1.PatientID == ds2.PatientID
        assert np.array_equal(ds1.pixel_array, ds2.pixel_array)


# ---------------------------------------------------------------------------
# hu_array property
# ---------------------------------------------------------------------------

class TestHuArray:
    def test_ct_hu_conversion(self):
        ds = generate_synthetic_dicom(modality="CT", seed=0)
        hu = ds.hu_array
        # Stored values offset by RescaleIntercept=-1024
        stored = ds.pixel_array.astype(np.float32)
        expected = stored * ds.RescaleSlope + ds.RescaleIntercept
        assert np.allclose(hu, expected)

    def test_xray_hu_array_equals_pixel_array(self):
        ds = generate_synthetic_dicom(modality="CR", seed=0)
        assert np.array_equal(ds.hu_array, ds.pixel_array.astype(np.float32))

    def test_ct_hu_range(self):
        """CT HU values should include negative values (air = -1000 HU)."""
        ds = generate_synthetic_dicom(modality="CT", size=256, seed=0)
        hu = ds.hu_array
        assert hu.min() < 0, f"CT HU min={hu.min()}, expected negative values"


# ---------------------------------------------------------------------------
# Windowing
# ---------------------------------------------------------------------------

class TestApplyWindowing:
    def test_output_uint8(self):
        arr = np.arange(0, 256, dtype=np.float32)
        out = apply_windowing(arr, window_center=128.0, window_width=256.0)
        assert out.dtype == np.uint8

    def test_output_range_0_255(self):
        arr = np.random.uniform(-2000, 2000, (64, 64)).astype(np.float32)
        out = apply_windowing(arr, window_center=0.0, window_width=400.0)
        assert out.min() >= 0
        assert out.max() <= 255

    def test_clipping_below_window(self):
        arr = np.array([-1000.0, -500.0, 0.0, 500.0, 1000.0])
        out = apply_windowing(arr, window_center=0.0, window_width=100.0)
        assert int(out[0]) == 0
        assert int(out[4]) == 255

    def test_center_maps_to_midpoint(self):
        """Window center should map to ~128 in the output."""
        arr = np.array([50.0, 50.0])
        out = apply_windowing(arr, window_center=50.0, window_width=100.0)
        assert abs(int(out[0]) - 128) <= 1

    def test_hu_windowing_lung_preset(self):
        ds = generate_synthetic_dicom(modality="CT", seed=0)
        out = apply_hu_windowing(ds.hu_array, preset="lung")
        assert out.dtype == np.uint8
        assert out.min() >= 0 and out.max() <= 255

    @pytest.mark.parametrize("preset", list(_CT_WINDOW_PRESETS.keys()))
    def test_all_ct_presets(self, preset):
        ds = generate_synthetic_dicom(modality="CT", seed=0)
        out = apply_hu_windowing(ds.hu_array, preset=preset)
        assert out.shape == ds.pixel_array.shape


# ---------------------------------------------------------------------------
# Model input preprocessing
# ---------------------------------------------------------------------------

class TestDicomToModelInput:
    def test_output_shape_default(self):
        windowed = np.random.randint(0, 255, (256, 256), dtype=np.uint8)
        out = dicom_to_model_input(windowed)
        assert out.shape == (1, 224, 224)

    def test_output_dtype(self):
        windowed = np.zeros((64, 64), dtype=np.uint8)
        out = dicom_to_model_input(windowed, target_size=64)
        assert out.dtype == np.float32

    def test_output_range(self):
        windowed = np.full((64, 64), 128, dtype=np.uint8)
        out = dicom_to_model_input(windowed, target_size=64)
        assert out.min() >= -1.0 - 1e-5
        assert out.max() <= 1.0 + 1e-5

    def test_preprocess_dicom_ct(self):
        ds = generate_synthetic_dicom(modality="CT", seed=0)
        out = preprocess_dicom(ds)
        assert out.shape == (1, 224, 224)
        assert out.dtype == np.float32

    def test_preprocess_dicom_cr(self):
        ds = generate_synthetic_dicom(modality="CR", seed=0)
        out = preprocess_dicom(ds)
        assert out.shape == (1, 224, 224)
