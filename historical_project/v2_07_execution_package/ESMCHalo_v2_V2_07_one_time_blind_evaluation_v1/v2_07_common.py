#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    roc_auc_score,
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def sigmoid(values: np.ndarray) -> np.ndarray:
    x = np.clip(np.asarray(values, dtype=np.float64), -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-x))


def expected_calibration_error(
    labels: np.ndarray,
    probability: np.ndarray,
    n_bins: int = 10,
) -> float:
    labels = np.asarray(labels, dtype=np.int64)
    probability = np.asarray(probability, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    result = 0.0
    for index in range(n_bins):
        lower, upper = edges[index], edges[index + 1]
        if index == n_bins - 1:
            mask = (probability >= lower) & (probability <= upper)
        else:
            mask = (probability >= lower) & (probability < upper)
        if mask.any():
            result += float(mask.mean()) * abs(
                float(labels[mask].mean()) - float(probability[mask].mean())
            )
    return float(result)


def metric_dict(
    labels: np.ndarray,
    probability: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    labels = np.asarray(labels, dtype=np.int64)
    probability = np.asarray(probability, dtype=np.float64)
    prediction = (probability >= threshold).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(labels, prediction, labels=[0, 1]).ravel()
    return {
        "n": int(len(labels)),
        "positives": int(labels.sum()),
        "negatives": int((labels == 0).sum()),
        "threshold": float(threshold),
        "ap": float(average_precision_score(labels, probability)),
        "auroc": float(roc_auc_score(labels, probability)),
        "mcc": float(matthews_corrcoef(labels, prediction)),
        "accuracy": float(accuracy_score(labels, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, prediction)),
        "sensitivity": float(tp / (tp + fn)) if tp + fn else float("nan"),
        "specificity": float(tn / (tn + fp)) if tn + fp else float("nan"),
        "ppv": float(precision_score(labels, prediction, zero_division=0)),
        "f1": float(f1_score(labels, prediction, zero_division=0)),
        "brier": float(brier_score_loss(labels, probability)),
        "ece_10_bins": expected_calibration_error(labels, probability, 10),
        "log_loss": float(log_loss(labels, probability, labels=[0, 1])),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }
