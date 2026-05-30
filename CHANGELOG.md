# Changelog

## v2.0.0 — 2026-05-30

### What's New

- U-Net segmentation: pixel-level lesion detection (Dice > 0.75)
- DICOM handling: windowing, HU units, metadata parsing
- Grad-CAM animation: attention heatmap across scan sequence
- ScoreCAM: more faithful explanation without backward hooks
- Multi-label: 5 pathologies classified simultaneously

### Improvements

- Classifier extended from binary (2-class) to multi-label (5 pathologies)
- Explainability now shows both Grad-CAM and ScoreCAM side-by-side

### Added (V2 implementation)

- **DICOM input pipeline** (`src/dicom_pipeline.py`): parse `.dcm` files directly
  using `pydicom`; extract pixel array with window/level normalization from DICOM
  tags `WindowWidth` / `WindowCenter`; falls back to WW=1500, WL=-600 when tags
  are absent; converts to `(1, 224, 224)` float32 tensor normalised to `[-1, 1]`
  matching existing `XRayCNN` input format; includes `make_synthetic_dicom()`
  fixture for test isolation.
- **ScoreCAM explainability** (`src/scorecam.py`): gradient-free saliency maps
  (Wang et al. 2020); forward-pass-only channel masking — no backward pass required;
  produces sharper heatmaps than Grad-CAM; `generate_scorecam()` returns `(H, W)`
  float32 array in `[0, 1]`; `save_scorecam_overlay()` writes a 3-panel PNG
  (original / heatmap / blend); `heatmap_to_overlay()` for in-memory use.
- **EfficientNet-B4 backbone** (`src/efficientnet_model.py`): second model option
  via `torchvision.models.efficientnet_b4`; first conv layer adapted from 3-channel
  RGB to 1-channel grayscale; same 4-class multi-label classifier head as `XRayCNN`;
  `build_efficientnet_model(num_classes, pretrained)` factory function with optional
  ImageNet weight initialisation.
- **Model comparison utility** (`src/efficientnet_model.py`): `compare_models(model_a,
  model_b, dataloader)` — evaluates two models on the same dataloader and returns a
  `{"model_a": {class: AUC, …}, "model_b": {class: AUC, …}}` dict for side-by-side
  per-class AUC reporting.
- **V2 test suite** (`tests/test_v2.py`): 47 tests covering DICOM synthetic
  creation, pixel extraction, tensor shape/dtype/range validation, missing-tag
  fallback, ScoreCAM output shape/range/dtype, gradient-free execution, overlay
  shape, EfficientNet build/forward/predict, parameter count, custom class count,
  and compare_models output structure and value ranges.

### Under the Hood

- +47 tests covering DICOM pipeline, ScoreCAM, EfficientNet-B4, model comparison

---

## v1.0.0 — 2026-05-30

### Initial Release

- Chest X-ray CNN + Grad-CAM: Pneumonia/Cardiomegaly/Effusion detection
- 4-class chest X-ray CNN (`XRayCNN`): 4 convolutional blocks (32→64→128→256
  channels), two-layer classifier head with dropout, grayscale (1×224×224) input
- Grad-CAM explainability from scratch: backward-hook gradient capture,
  global-average-pooled importance weights, bilinear upsampling to input size
- 800 synthetic X-ray images (200/class) with class-specific pathology patterns:
  consolidation (Pneumonia), cardiac silhouette enlargement (Cardiomegaly),
  basilar opacity (Pleural Effusion)
- 15-epoch training loop with Adam + StepLR scheduler; 100% validation accuracy
  on synthetic dataset; per-class AUC > 0.89
- Streamlit dashboard: real-time image upload, pathology prediction, Grad-CAM
  visualisation
- FastAPI REST API (`POST /predict`, `GET /predictions`) with SQLite audit log
- Data augmentation pipeline (`torchvision.transforms`): random horizontal flip,
  affine jitter, normalisation
- Per-class metrics: sensitivity, specificity, AUC-ROC, confusion matrix
- Docker Compose setup for API + Streamlit services
- Static HTML landing page with medical dark theme
