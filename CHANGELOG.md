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

### Under the Hood

- +35 tests covering U-Net shapes, DICOM parsing, multi-label metrics

## v1.0.0 — 2026-05-30

- Chest X-ray CNN + Grad-CAM: Pneumonia/Cardiomegaly/Effusion detection
