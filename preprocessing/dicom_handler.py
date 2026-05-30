"""
DICOM file simulation and preprocessing utilities.

Simulates DICOM metadata in pydicom-compatible format and implements:
  - Synthetic DICOM-like metadata generation (PatientID, StudyDate, Modality, etc.)
  - DICOM windowing (window/level transform) for both CT (HU) and X-ray modalities
  - Pixel array → normalised numpy array suitable for model input

No real DICOM files are required; all data is synthetically generated so that
the pipeline can be tested end-to-end without patient data.

Supported modalities:
  CT    — Hounsfield Unit range (-1000 to +3000 HU)
  CR/DX — Digital X-ray (0–255 or 0–4095 pixel values)

Usage::

    from preprocessing.dicom_handler import (
        generate_synthetic_dicom,
        apply_windowing,
        dicom_to_model_input,
        DicomMetadata,
    )

    ds = generate_synthetic_dicom(modality="CT", seed=42)
    arr = apply_windowing(ds.pixel_array, ds.WindowCenter, ds.WindowWidth)
    tensor = dicom_to_model_input(arr)
"""

from __future__ import annotations

import datetime
import random
from dataclasses import dataclass, field

import numpy as np


# ---------------------------------------------------------------------------
# Metadata dataclass (mirrors pydicom Dataset attribute names)
# ---------------------------------------------------------------------------

@dataclass
class DicomMetadata:
    """
    Minimal DICOM-like metadata container.

    Attribute names intentionally mirror pydicom Dataset so that real and
    synthetic objects can be handled uniformly.
    """

    PatientID:       str
    PatientName:     str
    StudyDate:       str   # YYYYMMDD
    StudyTime:       str   # HHMMSS.ffffff
    Modality:        str   # CT | CR | DX
    SOPInstanceUID:  str
    SeriesNumber:    int
    InstanceNumber:  int
    Rows:            int
    Columns:         int
    WindowCenter:    float
    WindowWidth:     float
    RescaleIntercept: float  # For CT: typically -1024 (maps stored→HU)
    RescaleSlope:    float   # For CT: typically 1.0
    BitsAllocated:   int
    PixelSpacing:    list[float]  # [row_spacing_mm, col_spacing_mm]
    pixel_array:     np.ndarray   # Raw stored pixel values
    _modality_meta: dict = field(default_factory=dict, repr=False)

    # ------------------------------------------------------------------ #
    # Convenience properties
    # ------------------------------------------------------------------ #

    @property
    def hu_array(self) -> np.ndarray:
        """
        Return CT pixels in Hounsfield Units.

        HU = pixel_value * RescaleSlope + RescaleIntercept
        For non-CT modalities returns pixel_array unchanged.
        """
        if self.Modality == "CT":
            return (
                self.pixel_array.astype(np.float32) * self.RescaleSlope
                + self.RescaleIntercept
            )
        return self.pixel_array.astype(np.float32)

    @property
    def image_size(self) -> tuple[int, int]:
        return (self.Rows, self.Columns)


# ---------------------------------------------------------------------------
# Synthetic DICOM generator
# ---------------------------------------------------------------------------

# Typical CT window presets  (center, width)
_CT_WINDOW_PRESETS: dict[str, tuple[float, float]] = {
    "lung":       (-600.0, 1500.0),
    "mediastinum": (50.0,  350.0),
    "bone":       (400.0, 1800.0),
    "abdomen":     (60.0,  400.0),
}

# Typical X-ray window (stored 0-255 range)
_XRAY_WINDOW = (128.0, 256.0)


def generate_synthetic_dicom(
    modality: str = "CR",
    size: int = 256,
    preset: str | None = None,
    seed: int | None = None,
) -> DicomMetadata:
    """
    Generate a synthetic DICOM-like object with realistic metadata and pixel data.

    Parameters
    ----------
    modality : str
        "CT", "CR" (computed radiography / chest X-ray), or "DX" (digital X-ray).
    size : int
        Pixel array size (size × size).
    preset : str | None
        CT window preset name (e.g. "lung"). Ignored for X-ray modalities.
        If None, defaults to "lung" for CT, standard window for X-ray.
    seed : int | None
        Random seed for reproducibility.

    Returns
    -------
    DicomMetadata
    """
    rng = np.random.default_rng(seed)
    py_rng = random.Random(seed)

    # ---- Metadata --------------------------------------------------------
    patient_id   = f"SYN{py_rng.randint(100000, 999999)}"
    patient_name = f"Synthetic^Patient^{py_rng.randint(1, 9999)}"

    base_date = datetime.date(2024, 1, 1)
    delta = datetime.timedelta(days=py_rng.randint(0, 365))
    study_date = (base_date + delta).strftime("%Y%m%d")
    study_time = f"{py_rng.randint(0, 23):02d}{py_rng.randint(0, 59):02d}{py_rng.randint(0, 59):02d}.000000"

    sop_uid = (
        f"1.2.840.10008.5.1.4.1.1.{py_rng.randint(1, 9)}"
        f".{py_rng.randint(1000000, 9999999)}"
        f".{py_rng.randint(1000000, 9999999)}"
    )

    # ---- Modality-specific pixel generation ------------------------------
    modality = modality.upper()

    if modality == "CT":
        # Stored values: 0–4095 (12-bit), representing HU after rescale
        # RescaleIntercept = -1024, so HU = stored - 1024
        rescale_intercept = -1024.0
        rescale_slope = 1.0
        bits = 16

        # Simulate a chest CT slice: air, soft tissue, bone
        pixel_array = _generate_ct_slice(rng, size)

        window_preset = preset or "lung"
        wc, ww = _CT_WINDOW_PRESETS.get(window_preset, _CT_WINDOW_PRESETS["lung"])
        pixel_spacing = [float(rng.uniform(0.6, 0.8))] * 2

    else:  # CR / DX
        rescale_intercept = 0.0
        rescale_slope = 1.0
        bits = 8
        pixel_array = _generate_xray_slice(rng, size)
        wc, ww = _XRAY_WINDOW
        pixel_spacing = [float(rng.uniform(0.14, 0.20))] * 2

    return DicomMetadata(
        PatientID=patient_id,
        PatientName=patient_name,
        StudyDate=study_date,
        StudyTime=study_time,
        Modality=modality,
        SOPInstanceUID=sop_uid,
        SeriesNumber=py_rng.randint(1, 5),
        InstanceNumber=py_rng.randint(1, 50),
        Rows=size,
        Columns=size,
        WindowCenter=wc,
        WindowWidth=ww,
        RescaleIntercept=rescale_intercept,
        RescaleSlope=rescale_slope,
        BitsAllocated=bits,
        PixelSpacing=pixel_spacing,
        pixel_array=pixel_array,
    )


# ---------------------------------------------------------------------------
# Internal pixel generators
# ---------------------------------------------------------------------------

def _generate_ct_slice(rng: np.random.Generator, size: int) -> np.ndarray:
    """
    Simulate a CT chest slice.

    Stored values ≈ HU + 1024 (RescaleIntercept = -1024).
      air body → ~0 HU stored as ~1024
      soft tissue → ~50 HU stored as ~1074
      bone → ~700 HU stored as ~1724
    """
    # Air background
    img = np.full((size, size), 24, dtype=np.int16)  # ~air -1000 HU

    y_g, x_g = np.ogrid[:size, :size]
    cx, cy = size // 2, size // 2

    # Body oval (soft tissue, ~50 HU stored ≈ 1074)
    body = ((x_g - cx) ** 2 / (size * 0.45) ** 2 + (y_g - cy) ** 2 / (size * 0.48) ** 2) <= 1.0
    img[body] = 1074 + rng.integers(-30, 30, img[body].shape)

    # Lung fields (air-filled, ~-600 HU stored ≈ 424)
    left_lung  = ((x_g - cx + size // 6) ** 2 / (size // 7) ** 2 +
                  (y_g - cy) ** 2 / (size // 5) ** 2) <= 1.0
    right_lung = ((x_g - cx - size // 6) ** 2 / (size // 7) ** 2 +
                  (y_g - cy) ** 2 / (size // 5) ** 2) <= 1.0
    for lung_mask in [left_lung, right_lung]:
        img[lung_mask] = 424 + rng.integers(-50, 50, img[lung_mask].shape)

    # Heart (~40 HU stored ≈ 1064)
    heart = ((x_g - cx + size // 10) ** 2 / (size // 12) ** 2 +
             (y_g - cy + size // 10) ** 2 / (size // 10) ** 2) <= 1.0
    img[heart] = 1064 + rng.integers(-20, 20, img[heart].shape)

    # Ribs (bone, ~700 HU stored ≈ 1724) — thin arcs
    for i, row in enumerate(range(size // 5, size - size // 5, size // 8)):
        arc = (
            (abs(y_g - row) <= 2) &
            (x_g > size // 8) & (x_g < size - size // 8) &
            body
        )
        img[arc] = 1724 + rng.integers(-50, 50, img[arc].shape)

    return np.clip(img, 0, 4095).astype(np.uint16)


def _generate_xray_slice(rng: np.random.Generator, size: int) -> np.ndarray:
    """Simulate a chest X-ray pixel array (0–255 stored values)."""
    img = rng.normal(80, 15, (size, size)).astype(np.float32)

    y_g, x_g = np.ogrid[:size, :size]
    cx, cy = size // 2, size // 2

    # Lung ovals
    left_lung  = ((x_g - (cx - size // 5)) ** 2 / (size // 7) ** 2 +
                  (y_g - cy) ** 2 / (size // 5) ** 2) <= 1.0
    right_lung = ((x_g - (cx + size // 5)) ** 2 / (size // 7) ** 2 +
                  (y_g - cy) ** 2 / (size // 5) ** 2) <= 1.0
    img[left_lung]  -= rng.normal(20, 5, img[left_lung].shape)
    img[right_lung] -= rng.normal(20, 5, img[right_lung].shape)

    # Cardiac silhouette
    heart = ((x_g - (cx - size // 12)) ** 2 / (size // 10) ** 2 +
             (y_g - (cy + size // 10)) ** 2 / (size // 8) ** 2) <= 1.0
    img[heart] += rng.normal(40, 6, img[heart].shape)

    return np.clip(img, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# DICOM windowing
# ---------------------------------------------------------------------------

def apply_windowing(
    pixel_array: np.ndarray,
    window_center: float,
    window_width: float,
) -> np.ndarray:
    """
    Apply DICOM window/level transform to a pixel array.

    Maps values in [WC - WW/2, WC + WW/2] to [0, 255] with linear clipping.

    Parameters
    ----------
    pixel_array : np.ndarray
        Raw stored pixel values (int or float).
    window_center : float
        Centre of the display window (level).
    window_width : float
        Total width of the display window.

    Returns
    -------
    np.ndarray
        uint8 array with values 0–255.
    """
    low  = window_center - window_width / 2.0
    high = window_center + window_width / 2.0

    arr = pixel_array.astype(np.float32)
    out = np.clip((arr - low) / (high - low), 0.0, 1.0) * 255.0
    return out.astype(np.uint8)


def apply_hu_windowing(
    hu_array: np.ndarray,
    preset: str = "lung",
) -> np.ndarray:
    """
    Apply a named CT window preset to a Hounsfield Unit array.

    Parameters
    ----------
    hu_array : np.ndarray
        CT values in Hounsfield Units (float32).
    preset : str
        One of: "lung", "mediastinum", "bone", "abdomen".

    Returns
    -------
    np.ndarray  uint8 (0–255)
    """
    wc, ww = _CT_WINDOW_PRESETS.get(preset, _CT_WINDOW_PRESETS["lung"])
    return apply_windowing(hu_array, wc, ww)


# ---------------------------------------------------------------------------
# Model input preparation
# ---------------------------------------------------------------------------

def dicom_to_model_input(
    windowed_array: np.ndarray,
    target_size: int = 224,
) -> np.ndarray:
    """
    Convert a windowed uint8 pixel array to a normalised float32 numpy array
    suitable for model input.

    Steps:
      1. Resize to (target_size, target_size) via nearest-neighbour interpolation.
      2. Normalise from [0, 255] to [-1, 1] (mean=0.5, std=0.5 — matches model training).

    Parameters
    ----------
    windowed_array : np.ndarray
        uint8 array (H, W) produced by ``apply_windowing``.
    target_size : int
        Spatial dimension for model input.

    Returns
    -------
    np.ndarray
        float32 array of shape (1, target_size, target_size), range [-1, 1].
    """
    from PIL import Image

    img = Image.fromarray(windowed_array, mode="L").resize(
        (target_size, target_size), Image.NEAREST
    )
    arr = np.array(img, dtype=np.float32) / 255.0   # [0, 1]
    arr = (arr - 0.5) / 0.5                         # [-1, 1]
    return arr[np.newaxis, :]                        # (1, H, W)


def preprocess_dicom(
    ds: DicomMetadata,
    target_size: int = 224,
) -> np.ndarray:
    """
    Full preprocessing pipeline for a DicomMetadata object.

    For CT: applies HU conversion then lung windowing.
    For X-ray: applies standard window/level.

    Returns
    -------
    np.ndarray
        float32 (1, target_size, target_size) ready for model inference.
    """
    if ds.Modality == "CT":
        hu = ds.hu_array
        windowed = apply_hu_windowing(hu, preset="lung")
    else:
        windowed = apply_windowing(ds.pixel_array, ds.WindowCenter, ds.WindowWidth)

    return dicom_to_model_input(windowed, target_size=target_size)


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    for mod in ["CT", "CR", "DX"]:
        ds = generate_synthetic_dicom(modality=mod, size=256, seed=0)
        arr = preprocess_dicom(ds)
        assert arr.shape == (1, 224, 224), f"Wrong shape: {arr.shape}"
        assert arr.dtype == np.float32, "Wrong dtype"
        assert arr.min() >= -1.0 - 1e-6 and arr.max() <= 1.0 + 1e-6, "Values out of range"
        print(
            f"  {mod:3s}  PatientID={ds.PatientID}  "
            f"StudyDate={ds.StudyDate}  WC={ds.WindowCenter}  WW={ds.WindowWidth}  "
            f"array={arr.shape}  range=[{arr.min():.3f}, {arr.max():.3f}]"
        )
    print("DICOM handler smoke test passed.")
