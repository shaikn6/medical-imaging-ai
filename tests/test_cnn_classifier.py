"""
Tests for model/cnn_classifier.py
"""

import os
import sys

import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from model.cnn_classifier import XRayCNN, build_model, NUM_CLASSES, CLASSES


class TestXRayCNN:
    def test_build_model_returns_correct_type(self):
        model = build_model()
        assert isinstance(model, XRayCNN)

    def test_forward_output_shape(self):
        model = build_model()
        model.eval()
        x = torch.randn(2, 1, 224, 224)
        logits = model(x)
        assert logits.shape == (2, NUM_CLASSES), f"Expected (2, {NUM_CLASSES}), got {logits.shape}"

    def test_forward_batch_1(self):
        model = build_model()
        model.eval()
        x = torch.randn(1, 1, 224, 224)
        logits = model(x)
        assert logits.shape == (1, NUM_CLASSES)

    def test_forward_batch_32(self):
        model = build_model()
        model.eval()
        x = torch.randn(32, 1, 224, 224)
        logits = model(x)
        assert logits.shape == (32, NUM_CLASSES)

    def test_last_conv_layer_is_conv2d(self):
        model = build_model()
        layer = model.last_conv_layer
        assert isinstance(layer, torch.nn.Conv2d)

    def test_last_conv_layer_has_256_out_channels(self):
        model = build_model()
        layer = model.last_conv_layer
        assert layer.out_channels == 256

    def test_predict_proba_sums_to_one(self):
        model = build_model()
        model.eval()
        x = torch.randn(4, 1, 224, 224)
        probs = model.predict_proba(x)
        sums = probs.sum(dim=-1)
        assert torch.allclose(sums, torch.ones(4), atol=1e-5)

    def test_predict_returns_valid_class_indices(self):
        model = build_model()
        model.eval()
        x = torch.randn(8, 1, 224, 224)
        preds = model.predict(x)
        assert ((preds >= 0) & (preds < NUM_CLASSES)).all()

    def test_classes_count(self):
        assert len(CLASSES) == 4
        assert "Normal" in CLASSES
        assert "Pneumonia" in CLASSES

    def test_parameter_count_reasonable(self):
        """Model should be lightweight (< 5M parameters)."""
        model = build_model()
        n_params = sum(p.numel() for p in model.parameters())
        assert n_params < 5_000_000, f"Model too large: {n_params:,} params"

    def test_custom_num_classes(self):
        model = XRayCNN(num_classes=2)
        model.eval()
        x = torch.randn(1, 1, 224, 224)
        logits = model(x)
        assert logits.shape == (1, 2)

    def test_features_module_exists(self):
        model = build_model()
        assert hasattr(model, "features")
        assert hasattr(model, "classifier")

    def test_no_nan_in_output(self):
        model = build_model()
        model.eval()
        x = torch.randn(4, 1, 224, 224)
        logits = model(x)
        assert not torch.isnan(logits).any()
