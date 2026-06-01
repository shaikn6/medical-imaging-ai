"""
Tests for models/multilabel_classifier.py
"""

import os
import sys

import pytest
import torch
from torch.utils.data import DataLoader

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from models.multilabel_classifier import (
    MultiLabelXRayCNN,
    build_multilabel_model,
    SyntheticMultiLabelDataset,
    evaluate_multilabel,
    train_multilabel_demo,
    MULTILABEL_CLASSES,
    NUM_LABELS,
)


# ---------------------------------------------------------------------------
# Model architecture
# ---------------------------------------------------------------------------

class TestMultiLabelXRayCNN:
    @pytest.fixture(scope="class")
    def model(self):
        m = build_multilabel_model()
        m.eval()
        return m

    def test_build_returns_correct_type(self):
        assert isinstance(build_multilabel_model(), MultiLabelXRayCNN)

    def test_output_shape(self, model):
        x = torch.randn(4, 1, 224, 224)
        out = model(x)
        assert out.shape == (4, NUM_LABELS), f"Expected (4, {NUM_LABELS}), got {out.shape}"

    def test_num_labels_five(self):
        assert NUM_LABELS == 5

    def test_class_names(self):
        assert "Pneumonia"    in MULTILABEL_CLASSES
        assert "Cardiomegaly" in MULTILABEL_CLASSES
        assert "Effusion"     in MULTILABEL_CLASSES
        assert "Atelectasis"  in MULTILABEL_CLASSES
        assert "Nodule"       in MULTILABEL_CLASSES

    def test_predict_proba_range(self, model):
        x = torch.randn(4, 1, 224, 224)
        probs = model.predict_proba(x)
        assert probs.min() >= 0.0 - 1e-6
        assert probs.max() <= 1.0 + 1e-6

    def test_predict_proba_shape(self, model):
        x = torch.randn(2, 1, 224, 224)
        probs = model.predict_proba(x)
        assert probs.shape == (2, NUM_LABELS)

    def test_predict_binary(self, model):
        x = torch.randn(3, 1, 224, 224)
        preds = model.predict(x)
        unique = torch.unique(preds)
        assert set(unique.tolist()).issubset({0.0, 1.0})

    def test_labels_independent_not_sum_to_one(self, model):
        """Multi-label sigmoid outputs do NOT sum to 1 (unlike softmax)."""
        x = torch.randn(4, 1, 224, 224)
        probs = model.predict_proba(x)
        sums = probs.sum(dim=-1)
        # At least one sample should have sum != 1.0
        assert not torch.allclose(sums, torch.ones(4), atol=0.01)

    def test_last_conv_layer_is_conv2d(self, model):
        layer = model.last_conv_layer
        assert isinstance(layer, torch.nn.Conv2d)

    def test_no_nan_in_output(self, model):
        x = torch.randn(4, 1, 224, 224)
        out = model(x)
        assert not torch.isnan(out).any()


# ---------------------------------------------------------------------------
# Synthetic multi-label dataset
# ---------------------------------------------------------------------------

class TestSyntheticMultiLabelDataset:
    def test_len(self):
        ds = SyntheticMultiLabelDataset(n_samples=20, seed=0)
        assert len(ds) == 20

    def test_image_shape(self):
        ds = SyntheticMultiLabelDataset(n_samples=4, seed=0)
        img, label = ds[0]
        assert img.shape == (1, 224, 224)

    def test_label_shape(self):
        ds = SyntheticMultiLabelDataset(n_samples=4, seed=0)
        _, label = ds[0]
        assert label.shape == (NUM_LABELS,)

    def test_label_binary(self):
        ds = SyntheticMultiLabelDataset(n_samples=10, seed=0)
        for i in range(10):
            _, label = ds[i]
            unique = torch.unique(label)
            assert set(unique.tolist()).issubset({0.0, 1.0})

    def test_at_least_one_label_per_sample(self):
        ds = SyntheticMultiLabelDataset(n_samples=20, seed=0)
        for i in range(20):
            _, label = ds[i]
            assert label.sum() >= 1, f"Sample {i} has zero labels"


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

class TestEvaluateMultilabel:
    @pytest.fixture(scope="class")
    def dataloader(self):
        ds = SyntheticMultiLabelDataset(n_samples=40, seed=99)
        return DataLoader(ds, batch_size=8)

    def test_evaluate_returns_metrics(self, dataloader):
        model = build_multilabel_model()
        model.eval()
        metrics = evaluate_multilabel(model, dataloader)
        assert metrics is not None

    def test_macro_auc_in_range(self, dataloader):
        model = build_multilabel_model()
        model.eval()
        metrics = evaluate_multilabel(model, dataloader)
        assert 0.0 <= metrics.macro_auc <= 1.0

    def test_hamming_loss_in_range(self, dataloader):
        model = build_multilabel_model()
        model.eval()
        metrics = evaluate_multilabel(model, dataloader)
        assert 0.0 <= metrics.hamming_loss <= 1.0

    def test_per_class_auc_count(self, dataloader):
        model = build_multilabel_model()
        model.eval()
        metrics = evaluate_multilabel(model, dataloader)
        assert len(metrics.per_class_auc) == NUM_LABELS

    def test_per_class_names_count(self, dataloader):
        model = build_multilabel_model()
        model.eval()
        metrics = evaluate_multilabel(model, dataloader)
        assert len(metrics.per_class_names) == NUM_LABELS


# ---------------------------------------------------------------------------
# Training demo
# ---------------------------------------------------------------------------

class TestTrainMultilabelDemo:
    def test_demo_runs(self):
        result = train_multilabel_demo(n_samples=40, epochs=2, batch_size=8, seed=0)
        assert "metrics" in result and "model" in result

    def test_demo_returns_multilabel_model(self):
        result = train_multilabel_demo(n_samples=20, epochs=1, batch_size=8, seed=0)
        assert isinstance(result["model"], MultiLabelXRayCNN)
