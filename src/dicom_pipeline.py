"""
DICOM Input Pipeline — V2

Parse .dcm files using pydicom, apply window/level normalization,
and convert to a (1, H, W) grayscale tensor matching the XRayCNN
input format.

Usage::

    from src.dicom_pipeline import load_dicom_image, dicom_to_tensor

    dcm = load_dicom_image("scan.dcm")
    tensor = dicom_to_tensor(dcm)   # shape (1, 224, 224)
"""

from __future__ import annotations

import warnings
from typing import Optional, Tuple

import numpy as np
import pydicom
import pydicom.dataset
import torch
import torch.nn.functional as F
from pydicom.dataset import Dataset, FileDataset
from pydicom.uid import ExplicitVRLittleEndian


# ---------------------------------------------------------------------------
# Default window/level fallbacks (chest CT / X-ray)
# ---------------------------------------------------------------------------
DEFAULT_WINDOW_WIDTH: float = 1500.0
DEFAULT_WINDOW_LEVEL: float = -600.0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_dicom_image(path: str) -> FileDataset:
    """
    Load a DICOM file from disk.

    Parameters
    ----------
    path : str
        Filesystem path to a .dcm file.

    Returns
    -------
    pydicom.FileDataset
        Parsed DICOM dataset object.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    pydicom.errors.InvalidDicomError
        If the file is not valid DICOM.
    """
    dcm = pydicom.dcmread(path, force=True)
    return dcm


def dicom_to_tensor(
    dcm: Dataset,
    target_size: Tuple[int, int] = (224, 224),
    window_width: Optional[float] = None,
    window_level: Optional[float] = None,
) -> torch.Tensor:
    """
    Convert a pydicom Dataset to a normalised (1, H, W) float32 tensor.

    Window/level values are read from DICOM tags
    ``WindowWidth`` (0028,1051) and ``WindowCenter`` (0028,1050).
    If missing, falls back to ``DEFAULT_WINDOW_WIDTH`` / ``DEFAULT_WINDOW_LEVEL``.

    Parameters
    ----------
    dcm : pydicom.Dataset
        Loaded DICOM dataset containing ``pixel_array``.
    target_size : (int, int)
        Spatial size to resize to (H, W).  Default (224, 224).
    window_width : float | None
        Override window width.  Uses DICOM tag or default if None.
    window_level : float | None
        Override window level (centre).  Uses DICOM tag or default if None.

    Returns
    -------
    torch.Tensor
        Shape (1, H, W), dtype float32, values normalised to [-1, 1].
    """
    pixel_array = _extract_pixel_array(dcm)
    ww, wl = _get_window_params(dcm, window_width, window_level)
    normalised = _apply_window_level(pixel_array, ww, wl)  # float32 in [-1, 1]

    tensor = torch.from_numpy(normalised).unsqueeze(0)  # (1, H, W)
    tensor = _resize_tensor(tensor, target_size)
    return tensor


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _extract_pixel_array(dcm: Dataset) -> np.ndarray:
    """Return the pixel array as float32, handling missing/corrupt data."""
    try:
        pixels = dcm.pixel_array.astype(np.float32)
    except Exception as exc:  # noqa: BLE001
        warnings.warn(
            f"Could not read pixel_array from DICOM: {exc}. "
            "Using zero array (512x512).",
            stacklevel=3,
        )
        pixels = np.zeros((512, 512), dtype=np.float32)

    # Collapse multi-frame or RGB to 2-D grayscale
    if pixels.ndim == 3 and pixels.shape[0] < pixels.shape[1]:
        # (frames, H, W) — take first frame
        pixels = pixels[0].astype(np.float32)
    elif pixels.ndim == 3 and pixels.shape[2] in (3, 4):
        # (H, W, C) — luminance conversion
        pixels = (
            0.2989 * pixels[:, :, 0]
            + 0.5870 * pixels[:, :, 1]
            + 0.1140 * pixels[:, :, 2]
        ).astype(np.float32)

    return pixels


def _get_window_params(
    dcm: Dataset,
    override_ww: Optional[float],
    override_wl: Optional[float],
) -> Tuple[float, float]:
    """Read window/level from DICOM tags or fall back to defaults."""
    # Window width
    if override_ww is not None:
        ww = float(override_ww)
    elif hasattr(dcm, "WindowWidth"):
        raw = dcm.WindowWidth
        # Tag may be a pydicom DSfloat, list, or DSfloat multi-value
        ww = float(raw[0]) if hasattr(raw, "__iter__") and not isinstance(raw, str) else float(raw)
    else:
        warnings.warn("DICOM tag WindowWidth missing; using default 1500.", stacklevel=3)
        ww = DEFAULT_WINDOW_WIDTH

    # Window level / centre
    if override_wl is not None:
        wl = float(override_wl)
    elif hasattr(dcm, "WindowCenter"):
        raw = dcm.WindowCenter
        wl = float(raw[0]) if hasattr(raw, "__iter__") and not isinstance(raw, str) else float(raw)
    else:
        warnings.warn("DICOM tag WindowCenter missing; using default -600.", stacklevel=3)
        wl = DEFAULT_WINDOW_LEVEL

    if ww <= 0:
        warnings.warn(f"Invalid WindowWidth={ww}; clamping to 1.", stacklevel=3)
        ww = 1.0

    return ww, wl


def _apply_window_level(
    pixels: np.ndarray,
    window_width: float,
    window_level: float,
) -> np.ndarray:
    """
    Apply window/level transformation and normalise to [-1, 1].

    Values below (level - width/2) map to -1.
    Values above (level + width/2) map to +1.
    Linear between.
    """
    low = window_level - window_width / 2.0
    high = window_level + window_width / 2.0

    clipped = np.clip(pixels, low, high)
    # Scale to [0, 1]
    normalised_01 = (clipped - low) / (high - low)
    # Scale to [-1, 1] to match existing XRayCNN training normalisation
    normalised = normalised_01 * 2.0 - 1.0
    return normalised.astype(np.float32)


def _resize_tensor(tensor: torch.Tensor, target_size: Tuple[int, int]) -> torch.Tensor:
    """Bilinearly resize (1, H, W) → (1, H', W')."""
    h, w = tensor.shape[1], tensor.shape[2]
    th, tw = target_size
    if h == th and w == tw:
        return tensor
    return F.interpolate(
        tensor.unsqueeze(0),  # (1, 1, H, W)
        size=target_size,
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)  # back to (1, H', W')


# ---------------------------------------------------------------------------
# Synthetic DICOM fixture (for tests — no real .dcm files needed)
# ---------------------------------------------------------------------------

def make_synthetic_dicom(
    rows: int = 256,
    cols: int = 256,
    window_width: float = 1500.0,
    window_level: float = -600.0,
    seed: int = 42,
) -> Dataset:
    """
    Create an in-memory DICOM Dataset with synthetic pixel data.

    This is intended for unit tests only — no file I/O required.

    Parameters
    ----------
    rows, cols : int
        Image dimensions.
    window_width, window_level : float
        DICOM window tags to embed.
    seed : int
        Random seed for reproducible pixels.

    Returns
    -------
    pydicom.Dataset
    """
    rng = np.random.default_rng(seed)

    ds = Dataset()
    ds.file_meta = Dataset()
    ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds.file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.2"
    ds.file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
    ds.is_implicit_VR = False
    ds.is_little_endian = True

    ds.Rows = rows
    ds.Columns = cols
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 1  # signed
    ds.WindowWidth = str(window_width)
    ds.WindowCenter = str(window_level)

    # Synthetic HU-like values in [-1000, 500]
    pixels = rng.integers(-1000, 500, size=(rows, cols), dtype=np.int16)
    ds.PixelData = pixels.tobytes()
    ds["PixelData"].VR = "OB"

    return ds
