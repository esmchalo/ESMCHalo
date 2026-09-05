#!/usr/bin/env python3
"""Frozen ESMCHalo final-v3 contract and downstream scoring utilities.

This module contains no training or threshold-selection code.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import joblib
import numpy as np


RELEASE_VERSION = "ESMCHalo_BMC_Q1_reproducibility_release_v2"
MODEL_FILENAME = "M3_prospective_final_v3.joblib"
MODEL_SHA256 = "3dcc63996eec47ecef3509bd2714527eef124cfb449ab308320fc340dfed9d5b"
MODEL_NAME = "ESMCHalo_prospective_final_v3"
ESMC_REPOSITORY = "biohub/ESMC-600M"
ESMC_REVISION = "a7e82012c83126b9eedb055fea9fa84b6c02f094"
EMBEDDING_LAYER = 36
EMBEDDING_WIDTH = 1152
LONG_SEQUENCE_WINDOW_SIZE = 2046
LONG_SEQUENCE_OVERLAP = 256
LONG_SEQUENCE_STRIDE = LONG_SEQUENCE_WINDOW_SIZE - LONG_SEQUENCE_OVERLAP
POOLING = (
    "mean over non-special residue tokens; for proteins longer than 2046 residues, "
    "overlapping window states are first averaged per residue and the resolved "
    "residue states are then averaged over the full protein"
)
FIXED_THRESHOLD = 0.47
POSITIVE_CLASS = 1
PROSPECTIVE_LR_C = 0.01
TRAINING_HOMOLOGY_IDENTITY = 0.4
ALLOWED_AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWYBXZJUO")

SCOPE_WARNING = (
    "ESMCHalo is a protein-level ranking/classification model, not a mutation-effect "
    "predictor. External evidence supports transferable ranking only unevenly; the "
    "Platt probabilities and fixed threshold 0.47 are not universally portable. "
    "Validate calibration and operating thresholds in the intended target domain."
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sequence_sha256(sequence: str) -> str:
    return hashlib.sha256(sequence.encode("utf-8")).hexdigest()


def stable_expit(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    output = np.empty_like(array)
    positive = array >= 0
    output[positive] = 1.0 / (1.0 + np.exp(-array[positive]))
    exponential = np.exp(array[~positive])
    output[~positive] = exponential / (1.0 + exponential)
    return output


def validate_artifact(artifact: Any) -> dict[str, Any]:
    if not isinstance(artifact, dict):
        raise TypeError("The locked final-v3 artifact must be a dictionary")
    required = {
        "estimator",
        "platt_calibrator",
        "calibrated_probability_threshold",
        "threshold_selection",
        "training_ids",
        "training_homology_identity",
        "old_fixed_test_used_for_selection_or_evaluation",
        "seed",
    }
    missing = sorted(required - set(artifact))
    if missing:
        raise ValueError(f"Locked model artifact is missing keys: {missing}")

    threshold = float(artifact["calibrated_probability_threshold"])
    if not math.isclose(threshold, FIXED_THRESHOLD, abs_tol=1e-12, rel_tol=0):
        raise ValueError(f"Locked threshold changed: {threshold} != {FIXED_THRESHOLD}")
    if bool(artifact["old_fixed_test_used_for_selection_or_evaluation"]):
        raise ValueError("Artifact contract says the old fixed test was accessed")
    if not math.isclose(
        float(artifact["training_homology_identity"]),
        TRAINING_HOMOLOGY_IDENTITY,
        abs_tol=1e-12,
        rel_tol=0,
    ):
        raise ValueError("Training homology identity differs from the locked 0.4 contract")

    estimator = artifact["estimator"]
    calibrator = artifact["platt_calibrator"]
    if not hasattr(estimator, "decision_function") or not hasattr(estimator, "predict_proba"):
        raise TypeError("Locked estimator lacks the required scoring methods")
    if not hasattr(calibrator, "predict_proba"):
        raise TypeError("Locked Platt calibrator lacks predict_proba")
    n_features = int(getattr(estimator, "n_features_in_", -1))
    if n_features != EMBEDDING_WIDTH:
        raise ValueError(f"Estimator width mismatch: {n_features} != {EMBEDDING_WIDTH}")
    named_steps = getattr(estimator, "named_steps", {})
    lr = named_steps.get("logistic_regression") if hasattr(named_steps, "get") else None
    if lr is None:
        raise ValueError("Locked estimator lacks the logistic_regression pipeline step")
    if not math.isclose(float(lr.C), PROSPECTIVE_LR_C, abs_tol=1e-15, rel_tol=0):
        raise ValueError(f"Logistic-regression C changed: {lr.C}")
    if np.asarray(lr.coef_).shape != (1, EMBEDDING_WIDTH):
        raise ValueError(f"Unexpected LR coefficient shape: {np.asarray(lr.coef_).shape}")
    if np.asarray(calibrator.coef_).shape != (1, 1):
        raise ValueError("Unexpected Platt calibrator coefficient shape")
    return artifact


def load_locked_artifact(model_path: Path) -> dict[str, Any]:
    model_path = Path(model_path).resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"Locked model artifact is absent: {model_path}")
    observed = sha256_file(model_path)
    if observed != MODEL_SHA256:
        raise ValueError(
            "Locked final-v3 SHA256 mismatch: "
            f"observed {observed}; expected {MODEL_SHA256}"
        )
    return validate_artifact(joblib.load(model_path))


def score_embeddings(
    embeddings: np.ndarray,
    artifact: dict[str, Any],
) -> dict[str, np.ndarray]:
    matrix = np.asarray(embeddings, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != EMBEDDING_WIDTH:
        raise ValueError(
            f"Expected embedding matrix [n,{EMBEDDING_WIDTH}], observed {matrix.shape}"
        )
    estimator = artifact["estimator"]
    calibrator = artifact["platt_calibrator"]
    raw_logit = np.asarray(estimator.decision_function(matrix), dtype=np.float64).reshape(-1)
    raw_probability = np.asarray(
        estimator.predict_proba(matrix)[:, POSITIVE_CLASS], dtype=np.float64
    )
    expected_raw = stable_expit(raw_logit)
    if not np.allclose(raw_probability, expected_raw, atol=2e-12, rtol=0):
        raise ValueError("Estimator probability is inconsistent with its raw logit")
    calibrated_probability = np.asarray(
        calibrator.predict_proba(raw_logit.reshape(-1, 1))[:, POSITIVE_CLASS],
        dtype=np.float64,
    )
    prediction = (calibrated_probability >= FIXED_THRESHOLD).astype(np.int64)
    return {
        "raw_logit": raw_logit,
        "raw_probability": raw_probability,
        "calibrated_probability": calibrated_probability,
        "prediction": prediction,
    }


def model_contract(artifact: dict[str, Any]) -> dict[str, Any]:
    calibrator = artifact["platt_calibrator"]
    return {
        "release_version": RELEASE_VERSION,
        "model_name": MODEL_NAME,
        "model_sha256": MODEL_SHA256,
        "representation": "frozen ESMC-600M layer36 residue representation",
        "repository": ESMC_REPOSITORY,
        "repository_revision": ESMC_REVISION,
        "embedding_layer": EMBEDDING_LAYER,
        "embedding_width": EMBEDDING_WIDTH,
        "pooling": POOLING,
        "long_sequence_contract": {
            "window_size_residues": LONG_SEQUENCE_WINDOW_SIZE,
            "nominal_overlap_residues": LONG_SEQUENCE_OVERLAP,
            "stride_residues": LONG_SEQUENCE_STRIDE,
            "final_window": "end-aligned",
            "overlap_resolution": "arithmetic mean across windows containing each residue",
            "protein_pooling": "arithmetic mean across resolved residue states",
        },
        "readout": "StandardScaler + LogisticRegression(C=0.01)",
        "platt_slope": float(np.asarray(calibrator.coef_).reshape(-1)[0]),
        "platt_intercept": float(np.asarray(calibrator.intercept_).reshape(-1)[0]),
        "fixed_threshold": FIXED_THRESHOLD,
        "training_records": len(artifact["training_ids"]),
        "training_homology_identity": float(artifact["training_homology_identity"]),
        "old_fixed_test_used_for_final_v3_selection_or_evaluation": bool(
            artifact["old_fixed_test_used_for_selection_or_evaluation"]
        ),
        "scope_warning": SCOPE_WARNING,
    }


def write_json_atomic(path: Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)
