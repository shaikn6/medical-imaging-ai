"""
Per-class evaluation metrics for the 4-class X-ray classifier.

Provides sensitivity (recall), specificity, and AUC-ROC for each class
plus a full confusion matrix.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
)


CLASSES = ["Normal", "Pneumonia", "Cardiomegaly", "Pleural Effusion"]


@dataclass
class ClassMetrics:
    class_name: str
    sensitivity: float   # TP / (TP + FN)
    specificity: float   # TN / (TN + FP)
    auc_roc: float
    tp: int = 0
    tn: int = 0
    fp: int = 0
    fn: int = 0


@dataclass
class MetricsReport:
    per_class: list[ClassMetrics] = field(default_factory=list)
    confusion_matrix: np.ndarray = field(default_factory=lambda: np.zeros((4, 4), dtype=int))
    macro_auc: float = 0.0

    def summary(self) -> str:
        lines = ["=" * 60, "Chest X-Ray Classification Metrics", "=" * 60]
        for m in self.per_class:
            lines.append(
                f"  {m.class_name:<20}  "
                f"Sens={m.sensitivity:.3f}  "
                f"Spec={m.specificity:.3f}  "
                f"AUC={m.auc_roc:.3f}"
            )
        lines += [
            "-" * 60,
            f"  Macro AUC: {self.macro_auc:.3f}",
            "=" * 60,
        ]
        return "\n".join(lines)


def per_class_metrics(
    y_true: np.ndarray | list,
    y_pred: np.ndarray | list,
    y_scores: np.ndarray | None = None,
    class_names: list[str] | None = None,
) -> MetricsReport:
    """
    Compute per-class sensitivity, specificity, and AUC-ROC.

    Parameters
    ----------
    y_true : array-like of int
        Ground-truth class indices.
    y_pred : array-like of int
        Predicted class indices.
    y_scores : np.ndarray, shape (N, num_classes) | None
        Softmax probabilities. Required for AUC-ROC. If None, AUC is set to NaN.
    class_names : list[str] | None
        Override default CLASSES list.

    Returns
    -------
    MetricsReport
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    names = class_names or CLASSES
    num_classes = len(names)

    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    per_class: list[ClassMetrics] = []

    for k, name in enumerate(names):
        tp = int(cm[k, k])
        fn = int(cm[k, :].sum() - tp)
        fp = int(cm[:, k].sum() - tp)
        tn = int(cm.sum() - tp - fn - fp)

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        if y_scores is not None:
            binary_true = (y_true == k).astype(int)
            try:
                auc = float(roc_auc_score(binary_true, y_scores[:, k]))
            except ValueError:
                auc = float("nan")
        else:
            auc = float("nan")

        per_class.append(
            ClassMetrics(
                class_name=name,
                sensitivity=sensitivity,
                specificity=specificity,
                auc_roc=auc,
                tp=tp, tn=tn, fp=fp, fn=fn,
            )
        )

    valid_aucs = [m.auc_roc for m in per_class if not np.isnan(m.auc_roc)]
    macro_auc = float(np.mean(valid_aucs)) if valid_aucs else float("nan")

    return MetricsReport(per_class=per_class, confusion_matrix=cm, macro_auc=macro_auc)
