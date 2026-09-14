#!/usr/bin/env python3
"""V2-04 DeepSaltPro group-OOF knowledge distillation.

The program intentionally operates only on frozen development artifacts. It
contains path guards against blind, test200, and external-challenge inputs and
does not import or instantiate ESMC or DeepSaltPro.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import random
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import sklearn
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


EXPECTED_SHA256 = {
    "folds": "68119ccd8a9c03244c2b23356b52bc9c20a4c1ba6319970917a8bca53863205d",
    "registry": "d02f0cb31d7b599d1de4780f61d3bda6b26390671dc42053ef1ba1c8dc2b0ee5",
    "canonical_rows": "48e032b4ef7cfcaf2a8b03037111fb0f1d75aec1c9db51a832cca9f54502d39f",
    "canonical_matrix": "04dd4e5a0e0d6738de5ad97a962b9fbd1bc514e2ce04333db3f5875e2ffc837e",
    "teacher_oof": "e3715f16f7f32fb0bd28f48250c5c0bf748f8666762ba52e2a5d853d508915b2",
}

FORBIDDEN_PATH_TOKENS = ("blind", "test200", "external", "challenge")
ID_COLUMNS = ("clean_id", "id", "protein_id", "sample_id", "record_id")
LABEL_COLUMNS = ("label", "y", "target", "class")
FOLD_COLUMNS = ("fold", "outer_fold", "fold_id", "oof_fold")
GROUP_COLUMNS = (
    "homology_group",
    "strict40_group",
    "cluster_id",
    "group",
    "mmseqs40_cluster",
)
STRICT25_COLUMNS = (
    "strict25",
    "is_strict25",
    "strict25_member",
    "strict25_subset",
    "strict25_eval",
    "strict25_diagnostic",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, obj: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def guarded_input(path: Path, role: str, access_log: list[dict[str, Any]]) -> Path:
    resolved = path.expanduser().resolve()
    lowered = str(resolved).lower()
    hits = [token for token in FORBIDDEN_PATH_TOKENS if token in lowered]
    if hits:
        raise RuntimeError(f"Forbidden input path for {role}: tokens={hits}, path={resolved}")
    if not resolved.is_file():
        raise FileNotFoundError(f"Missing input for {role}: {resolved}")
    access_log.append({"role": role, "path": str(resolved), "opened_at_utc": utc_now()})
    return resolved


def pick_column(frame: pd.DataFrame, candidates: Iterable[str], role: str) -> str:
    lower = {str(c).lower(): str(c) for c in frame.columns}
    for candidate in candidates:
        if candidate.lower() in lower:
            return lower[candidate.lower()]
    raise ValueError(f"Cannot resolve {role} column; columns={list(frame.columns)}")


def optional_column(frame: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    lower = {str(c).lower(): str(c) for c in frame.columns}
    for candidate in candidates:
        if candidate.lower() in lower:
            return lower[candidate.lower()]
    return None


def read_tsv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", low_memory=False)


def coerce_binary(series: pd.Series, role: str) -> np.ndarray:
    mapping = {
        "0": 0,
        "1": 1,
        "false": 0,
        "true": 1,
        "no": 0,
        "yes": 1,
        "n": 0,
        "y": 1,
    }
    values: list[int] = []
    for value in series:
        if pd.isna(value):
            raise ValueError(f"Missing value in {role}")
        key = str(value).strip().lower()
        if key in mapping:
            values.append(mapping[key])
        else:
            try:
                numeric = int(float(key))
            except ValueError as exc:
                raise ValueError(f"Non-binary value in {role}: {value!r}") from exc
            if numeric not in (0, 1):
                raise ValueError(f"Non-binary value in {role}: {value!r}")
            values.append(numeric)
    return np.asarray(values, dtype=np.int64)


def metric_dict(y: np.ndarray, probability: np.ndarray) -> dict[str, float | int]:
    y = np.asarray(y, dtype=np.int64)
    probability = np.asarray(probability, dtype=np.float64)
    prediction = (probability >= 0.5).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    return {
        "n": int(len(y)),
        "accuracy": float(accuracy_score(y, prediction)),
        "mcc": float(matthews_corrcoef(y, prediction)),
        "auroc": float(roc_auc_score(y, probability)),
        "ap": float(average_precision_score(y, probability)),
        "sensitivity": float(recall_score(y, prediction, pos_label=1, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else float("nan"),
        "ppv": float(precision_score(y, prediction, pos_label=1, zero_division=0)),
        "brier_uncalibrated": float(brier_score_loss(y, probability)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


class StudentMLP(nn.Module):
    def __init__(self, width: int, dropout: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(width, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(-1)


@dataclass(frozen=True)
class Candidate:
    candidate: str
    alpha: float
    beta: float
    gamma: float
    temperature: float


def rank_loss(student: torch.Tensor, teacher: torch.Tensor, pair_cap: int) -> torch.Tensor:
    n = int(student.numel())
    if n < 2:
        return student.sum() * 0.0
    # Pair adjacent elements after the DataLoader's deterministic shuffle. This
    # avoids quadratic memory and never consults labels.
    left = torch.arange(0, n - 1, 2, device=student.device)
    right = left + 1
    if left.numel() > pair_cap:
        left = left[:pair_cap]
        right = right[:pair_cap]
    teacher_delta = teacher[left] - teacher[right]
    keep = teacher_delta.abs() > 1e-12
    if not bool(keep.any()):
        return student.sum() * 0.0
    sign = teacher_delta[keep].sign()
    student_delta = student[left[keep]] - student[right[keep]]
    return F.softplus(-sign * student_delta).mean()


def combined_loss(
    student_logit: torch.Tensor,
    hard_label: torch.Tensor,
    teacher_logit: torch.Tensor,
    candidate: Candidate,
    pair_cap: int,
) -> torch.Tensor:
    hard = F.binary_cross_entropy_with_logits(student_logit, hard_label)
    total = candidate.alpha * hard
    if candidate.beta:
        temperature = candidate.temperature
        soft_target = torch.sigmoid(teacher_logit / temperature)
        kd = F.binary_cross_entropy_with_logits(student_logit / temperature, soft_target)
        total = total + candidate.beta * (temperature**2) * kd
    if candidate.gamma:
        total = total + candidate.gamma * rank_loss(student_logit, teacher_logit, pair_cap)
    return total


def data_loader(
    x: np.ndarray,
    y: np.ndarray,
    teacher: np.ndarray,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(np.asarray(x, dtype=np.float32)),
        torch.from_numpy(np.asarray(y, dtype=np.float32)),
        torch.from_numpy(np.asarray(teacher, dtype=np.float32)),
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


@torch.no_grad()
def predict(model: nn.Module, x: np.ndarray, device: torch.device, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    logits: list[np.ndarray] = []
    tensor = torch.from_numpy(np.asarray(x, dtype=np.float32))
    loader = DataLoader(tensor, batch_size=batch_size, shuffle=False, num_workers=0)
    for batch in loader:
        values = model(batch.to(device, non_blocking=True)).float().cpu().numpy()
        logits.append(values)
    raw = np.concatenate(logits).astype(np.float64)
    probability = 1.0 / (1.0 + np.exp(-np.clip(raw, -50.0, 50.0)))
    return raw, probability


def fit_for_epochs(
    x: np.ndarray,
    y: np.ndarray,
    teacher: np.ndarray,
    candidate: Candidate,
    protocol: dict[str, Any],
    device: torch.device,
    epochs: int,
    seed: int,
) -> StudentMLP:
    spec = protocol["student"]
    seed_everything(seed)
    model = StudentMLP(x.shape[1], float(spec["dropout"])).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(spec["learning_rate"]),
        weight_decay=float(spec["weight_decay"]),
    )
    loader = data_loader(x, y, teacher, int(spec["batch_size"]), True, seed)
    for _ in range(epochs):
        model.train()
        for batch_x, batch_y, batch_teacher in loader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            batch_teacher = batch_teacher.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_x)
            loss = combined_loss(
                logits,
                batch_y,
                batch_teacher,
                candidate,
                int(protocol["loss"]["rank_pair_cap_per_minibatch"]),
            )
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"Non-finite training loss for {candidate.candidate}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(spec["gradient_clip_norm"]))
            optimizer.step()
    return model


def choose_epoch(
    x_train: np.ndarray,
    y_train: np.ndarray,
    teacher_train: np.ndarray,
    x_inner: np.ndarray,
    y_inner: np.ndarray,
    candidate: Candidate,
    protocol: dict[str, Any],
    device: torch.device,
    seed: int,
) -> tuple[int, float, list[dict[str, float | int]]]:
    spec = protocol["student"]
    seed_everything(seed)
    model = StudentMLP(x_train.shape[1], float(spec["dropout"])).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(spec["learning_rate"]),
        weight_decay=float(spec["weight_decay"]),
    )
    loader = data_loader(x_train, y_train, teacher_train, int(spec["batch_size"]), True, seed)
    best_ap = -math.inf
    best_epoch = 1
    stale = 0
    history: list[dict[str, float | int]] = []
    for epoch in range(1, int(spec["maximum_epochs"]) + 1):
        model.train()
        total_loss = 0.0
        batches = 0
        for batch_x, batch_y, batch_teacher in loader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            batch_teacher = batch_teacher.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_x)
            loss = combined_loss(
                logits,
                batch_y,
                batch_teacher,
                candidate,
                int(protocol["loss"]["rank_pair_cap_per_minibatch"]),
            )
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"Non-finite selection loss for {candidate.candidate}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(spec["gradient_clip_norm"]))
            optimizer.step()
            total_loss += float(loss.detach().cpu())
            batches += 1
        _, probability = predict(model, x_inner, device, int(spec["batch_size"]) * 2)
        inner_ap = float(average_precision_score(y_inner, probability))
        history.append({"epoch": epoch, "mean_training_loss": total_loss / max(batches, 1), "inner_ap": inner_ap})
        if inner_ap > best_ap + 1e-6:
            best_ap = inner_ap
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
        if stale >= int(spec["early_stopping_patience"]):
            break
    return best_epoch, best_ap, history


def teacher_logits(frame: pd.DataFrame) -> tuple[np.ndarray, dict[str, Any]]:
    logit_col = optional_column(frame, ("teacher_logit", "raw_logit", "logit", "deepsaltpro_logit"))
    if logit_col is not None:
        values = pd.to_numeric(frame[logit_col], errors="raise").to_numpy(dtype=np.float64)
        mode = "raw_logit"
        source_col = logit_col
    else:
        raw_col = optional_column(frame, ("raw_score", "score_uncalibrated", "uncalibrated_score"))
        prob_col = optional_column(
            frame,
            (
                "probability_uncalibrated",
                "uncalibrated_probability",
                "raw_probability",
                "teacher_probability",
                "deepsaltpro_probability",
                "oof_probability",
                "predicted_probability",
                "probability",
                "prob",
                "y_score",
            ),
        )
        source_col = raw_col or prob_col
        if source_col is None:
            raise ValueError(
                "Teacher OOF lacks raw logit or explicitly uncalibrated score/probability; "
                f"columns={list(frame.columns)}"
            )
        source = pd.to_numeric(frame[source_col], errors="raise").to_numpy(dtype=np.float64)
        if np.all(np.isfinite(source)) and float(source.min()) >= 0.0 and float(source.max()) <= 1.0:
            clipped = np.clip(source, 1e-7, 1.0 - 1e-7)
            values = np.log(clipped / (1.0 - clipped))
            mode = "logit_recovered_from_uncalibrated_sigmoid_probability"
        else:
            values = source
            mode = "raw_uncalibrated_score_treated_as_logit"
    if not np.all(np.isfinite(values)):
        raise ValueError("Teacher logits contain non-finite values")
    return values, {
        "source_column": source_col,
        "conversion": mode,
        "minimum": float(values.min()),
        "maximum": float(values.max()),
        "mean": float(values.mean()),
        "std": float(values.std(ddof=0)),
    }


def strict25_membership(
    master: pd.DataFrame,
    folds: pd.DataFrame,
    registry: pd.DataFrame,
    explicit_path: Path | None,
    access_log: list[dict[str, Any]],
) -> tuple[np.ndarray, dict[str, Any]]:
    for source_name, source in (("folds", folds), ("registry", registry)):
        column = optional_column(source, STRICT25_COLUMNS)
        if column is None:
            continue
        source_id = pick_column(source, ID_COLUMNS, f"{source_name} ID")
        table = pd.DataFrame({"clean_id": source[source_id].astype(str), "member": coerce_binary(source[column], column)})
        if table.clean_id.duplicated().any():
            raise ValueError(f"Duplicate IDs in {source_name} strict25 column")
        mapping = table.set_index("clean_id")["member"]
        if not set(master.clean_id).issubset(mapping.index):
            raise ValueError(f"{source_name} strict25 column does not cover all development IDs")
        mask = master.clean_id.map(mapping).to_numpy(dtype=np.int64).astype(bool)
        return mask, {"source": source_name, "column": column, "mode": "boolean_column"}

    if explicit_path is None:
        raise RuntimeError(
            "Strict25 membership is required for the pre-registered gate but was not found "
            "in folds/registry. Re-run with --strict25-membership pointing to a frozen "
            "development-only TSV. No model training was started."
        )
    path = guarded_input(explicit_path, "strict25_membership", access_log)
    table = read_tsv(path)
    id_col = pick_column(table, ID_COLUMNS, "strict25 ID")
    if table[id_col].astype(str).duplicated().any():
        raise ValueError("Duplicate IDs in strict25 membership file")
    member_col = optional_column(table, STRICT25_COLUMNS)
    if member_col is None:
        member_ids = set(table[id_col].astype(str))
        unknown = member_ids - set(master.clean_id)
        if unknown:
            raise ValueError(f"Strict25 membership has {len(unknown)} unknown IDs")
        mask = master.clean_id.isin(member_ids).to_numpy()
        mode = "listed_ids_are_members"
    else:
        mapping = pd.Series(coerce_binary(table[member_col], member_col), index=table[id_col].astype(str))
        if not set(master.clean_id).issubset(mapping.index):
            raise ValueError("Boolean strict25 membership file does not cover all development IDs")
        mask = master.clean_id.map(mapping).to_numpy(dtype=np.int64).astype(bool)
        mode = "boolean_column"
    return mask, {"source": str(path), "column": member_col, "mode": mode}


def validate_v203b2_gate(root: Path, access_log: list[dict[str, Any]]) -> dict[str, Any]:
    directory = root / "ESMCHalo_v2_optimization/V2_03_head_layer_pooling_screen/V2_03B2_streamed_pooling_v1"
    done_path = guarded_input(directory / "V2_03B2_DONE.json", "v203b2_done", access_log)
    selection_path = guarded_input(directory / "selected_pooling_representations.json", "v203b2_selection", access_log)
    feature_manifest_path = guarded_input(directory / "FEATURE_SHA256.json", "v203b2_feature_manifest", access_log)
    done = json.loads(done_path.read_text(encoding="utf-8"))
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    manifest = json.loads(feature_manifest_path.read_text(encoding="utf-8"))
    if done.get("status") != "PASS_POOLING_SCREEN_COMPLETE":
        raise RuntimeError(f"V2-03B2 completion gate failed: {done.get('status')}")
    if done.get("records") != 10567 or done.get("blind_labels_read") is not False:
        raise RuntimeError("V2-03B2 record/blind gate failed")
    if selection.get("decision") != "KEEP_CANONICAL_POOLING":
        raise RuntimeError(f"Unexpected pooling decision: {selection}")
    checks: dict[str, Any] = {}
    for key in ("mean", "max", "std", "completed", "window_counts", "rows"):
        item = manifest.get(key)
        if not isinstance(item, dict):
            raise RuntimeError(f"Missing V2-03B2 feature manifest item: {key}")
        path = guarded_input(Path(item["path"]), f"v203b2_feature_{key}", access_log)
        actual = sha256(path)
        if actual != item["sha256"]:
            raise RuntimeError(f"V2-03B2 feature hash mismatch: {key}")
        checks[key] = {"path": str(path), "sha256": actual, "status": "PASS"}
    completed_path = Path(manifest["completed"]["path"])
    completed = np.load(completed_path, mmap_mode="r", allow_pickle=False)
    if completed.shape != (10567,) or int(np.asarray(completed).sum()) != 10567:
        raise RuntimeError("V2-03B2 completed.npy does not show 10567 completed records")
    return {
        "status": "PASS",
        "decision": selection["decision"],
        "records": done["records"],
        "feature_hash_checks": checks,
    }


def load_bind_inputs(args: argparse.Namespace, protocol: dict[str, Any], access_log: list[dict[str, Any]]) -> tuple[np.ndarray, pd.DataFrame, np.ndarray, dict[str, Any]]:
    root = args.root.resolve()
    paths = {
        "folds": root / "ESMCHalo_v2_optimization/V2_01_baseline_and_teacher_oof/group_folds_v2.tsv",
        "registry": root / "ESMCHalo_v2_optimization/V2_02_label_audit/V2_02B_audit_lock_v1/V2_02_LOCKED_TRAINING_REGISTRY.tsv",
        "canonical_rows": root / "data/embeddings/esmc600m_canonical_b1/rows.tsv",
        "canonical_matrix": root / "data/embeddings/esmc600m_canonical_b1/embeddings.npy",
        "teacher_oof": root / "ESMCHalo_v2_optimization/V2_01_baseline_and_teacher_oof/deepsaltpro_nested_oof_full_v1/deepsaltpro_oof_predictions.tsv",
    }
    actual_hashes: dict[str, dict[str, str]] = {}
    safe_paths: dict[str, Path] = {}
    for role, path in paths.items():
        safe = guarded_input(path, role, access_log)
        actual = sha256(safe)
        expected = EXPECTED_SHA256[role]
        if actual != expected:
            raise RuntimeError(f"SHA256 mismatch for {role}: expected={expected}, actual={actual}")
        actual_hashes[role] = {"path": str(safe), "sha256": actual, "status": "PASS"}
        safe_paths[role] = safe

    folds = read_tsv(safe_paths["folds"])
    registry = read_tsv(safe_paths["registry"])
    rows = read_tsv(safe_paths["canonical_rows"])
    teacher = read_tsv(safe_paths["teacher_oof"])
    matrix = np.load(safe_paths["canonical_matrix"], mmap_mode="r", allow_pickle=False)

    fold_id = pick_column(folds, ID_COLUMNS, "fold ID")
    fold_col = pick_column(folds, FOLD_COLUMNS, "fold")
    group_col = pick_column(folds, GROUP_COLUMNS, "strict40 group")
    registry_id = pick_column(registry, ID_COLUMNS, "registry ID")
    label_col = pick_column(registry, LABEL_COLUMNS, "registry label")
    row_id = pick_column(rows, ID_COLUMNS, "canonical row ID")
    embedding_row_col = pick_column(rows, ("embedding_row", "row", "row_index", "index"), "embedding row")
    teacher_id = pick_column(teacher, ID_COLUMNS, "teacher ID")

    for name, frame, column in (
        ("folds", folds, fold_id),
        ("registry", registry, registry_id),
        ("canonical_rows", rows, row_id),
        ("teacher", teacher, teacher_id),
    ):
        if frame[column].astype(str).duplicated().any():
            raise ValueError(f"Duplicate IDs in {name}")

    master = pd.DataFrame(
        {
            "clean_id": folds[fold_id].astype(str),
            "fold": pd.to_numeric(folds[fold_col], errors="raise").astype(int),
            "homology_group": folds[group_col].astype(str),
        }
    )
    label_map = pd.Series(coerce_binary(registry[label_col], "registry label"), index=registry[registry_id].astype(str))
    row_map = pd.Series(pd.to_numeric(rows[embedding_row_col], errors="raise").astype(int).to_numpy(), index=rows[row_id].astype(str))
    teacher_frame = teacher.copy()
    teacher_frame[teacher_id] = teacher_frame[teacher_id].astype(str)
    teacher_value, teacher_meta = teacher_logits(teacher_frame)
    teacher_map = pd.Series(teacher_value, index=teacher_frame[teacher_id])

    expected_ids = set(master.clean_id)
    # The canonical embedding store is a project-level superset.  It may contain
    # records that are not members of the locked 10,567-row development split.
    # Only development IDs are indexed below; extra rows are never assigned a
    # label or used for fitting. Registry and teacher OOF, in contrast, must be
    # exact development-only bindings.
    for name, index in (("registry", label_map.index), ("teacher", teacher_map.index)):
        if set(index) != expected_ids:
            raise ValueError(
                f"ID set mismatch for {name}: "
                f"missing={len(expected_ids-set(index))}, extra={len(set(index)-expected_ids)}"
            )
    canonical_ids = set(row_map.index)
    canonical_missing = expected_ids - canonical_ids
    if canonical_missing:
        raise ValueError(
            "Canonical rows do not cover the locked development set: "
            f"missing={len(canonical_missing)}, extra={len(canonical_ids-expected_ids)}"
        )

    master["label"] = master.clean_id.map(label_map).astype(int)
    master["embedding_row"] = master.clean_id.map(row_map).astype(int)
    master["teacher_logit"] = master.clean_id.map(teacher_map).astype(float)
    if len(master) != int(protocol["records_expected"]) or not master.clean_id.is_unique:
        raise ValueError("Development binding is not exactly 10567 unique records")
    if set(master.fold) != set(protocol["folds_expected"]):
        raise ValueError(f"Unexpected folds: {sorted(master.fold.unique())}")
    if master.groupby("homology_group").fold.nunique().max() != 1:
        raise ValueError("A strict40 homology group crosses outer folds")
    if matrix.ndim != 2 or matrix.shape[1] != 1152 or matrix.shape[0] != len(rows):
        raise ValueError(
            "Canonical matrix/rows shape mismatch: "
            f"matrix={matrix.shape}, rows={len(rows)}, expected_width=1152"
        )
    selected_rows = master.embedding_row.to_numpy(dtype=np.int64)
    if len(np.unique(selected_rows)) != len(master):
        raise ValueError("Canonical embedding rows are not unique for development IDs")
    if int(selected_rows.min()) < 0 or int(selected_rows.max()) >= matrix.shape[0]:
        raise ValueError(
            "A development embedding row is outside the canonical matrix: "
            f"min={int(selected_rows.min())}, max={int(selected_rows.max())}, "
            f"matrix_rows={matrix.shape[0]}"
        )

    teacher_label_col = optional_column(teacher_frame, LABEL_COLUMNS)
    if teacher_label_col is not None:
        teacher_labels = pd.Series(coerce_binary(teacher_frame[teacher_label_col], "teacher label"), index=teacher_frame[teacher_id])
        if not np.array_equal(master.label.to_numpy(), master.clean_id.map(teacher_labels).to_numpy(dtype=int)):
            raise ValueError("Teacher OOF labels disagree with locked registry")
    teacher_fold_col = optional_column(teacher_frame, FOLD_COLUMNS)
    if teacher_fold_col is not None:
        teacher_folds = pd.Series(pd.to_numeric(teacher_frame[teacher_fold_col], errors="raise").astype(int).to_numpy(), index=teacher_frame[teacher_id])
        if not np.array_equal(master.fold.to_numpy(), master.clean_id.map(teacher_folds).to_numpy(dtype=int)):
            raise ValueError("Teacher OOF folds disagree with locked strict40 folds")

    strict_mask, strict_meta = strict25_membership(
        master,
        folds,
        registry,
        args.strict25_membership,
        access_log,
    )
    if int(strict_mask.sum()) < 20:
        raise ValueError(f"Strict25 diagnostic contains too few records: {int(strict_mask.sum())}")
    if len(np.unique(master.loc[strict_mask, "label"])) != 2:
        raise ValueError("Strict25 diagnostic must contain both classes")

    # Advanced indexing materializes only the locked development rows. Extra
    # canonical rows remain unused and no label source is opened for them.
    x = np.asarray(matrix[selected_rows], dtype=np.float32)
    if not np.all(np.isfinite(x)):
        raise ValueError("Canonical feature matrix contains non-finite values")
    binding = {
        "status": "PASS",
        "records": int(len(master)),
        "positive": int(master.label.sum()),
        "negative": int((1 - master.label).sum()),
        "unique_ids": int(master.clean_id.nunique()),
        "unique_strict40_groups": int(master.homology_group.nunique()),
        "fold_sizes": {str(k): int(v) for k, v in master.fold.value_counts().sort_index().items()},
        "maximum_folds_per_group": int(master.groupby("homology_group").fold.nunique().max()),
        "canonical_store_shape": list(matrix.shape),
        "canonical_store_records": int(len(rows)),
        "canonical_unused_records": int(len(canonical_ids - expected_ids)),
        "canonical_selected_development_shape": list(x.shape),
        "teacher": teacher_meta,
        "strict25": {**strict_meta, "records": int(strict_mask.sum())},
        "input_hashes": actual_hashes,
        "labels_changed": 0,
        "sample_weights_applied": False,
        "blind_test200_external_labels_read": False,
    }
    return x, master, strict_mask, binding


def append_progress(path: Path, row: dict[str, Any]) -> None:
    exists = path.exists()
    pd.DataFrame([row]).to_csv(path, sep="\t", index=False, mode="a", header=not exists)


def run_candidate_fold(
    candidate: Candidate,
    outer_fold: int,
    x: np.ndarray,
    master: pd.DataFrame,
    protocol: dict[str, Any],
    device: torch.device,
    output: Path,
    progress_path: Path,
    base_seed: int,
) -> pd.DataFrame:
    partial_dir = output / "partial" / candidate.candidate
    partial_dir.mkdir(parents=True, exist_ok=True)
    result_path = partial_dir / f"fold_{outer_fold}.tsv"
    meta_path = partial_dir / f"fold_{outer_fold}.json"
    checkpoint_path = partial_dir / f"fold_{outer_fold}.pt"
    if result_path.is_file() and meta_path.is_file() and checkpoint_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("status") == "PASS" and meta.get("candidate") == candidate.candidate and meta.get("outer_fold") == outer_fold:
            frame = read_tsv(result_path)
            if len(frame) == int((master.fold == outer_fold).sum()) and frame.clean_id.is_unique:
                append_progress(progress_path, {"utc": utc_now(), "candidate": candidate.candidate, "fold": outer_fold, "state": "REUSED_VALID_CHECKPOINT"})
                return frame

    inner_fold = (outer_fold + 1) % 5
    selection_train = (master.fold != outer_fold) & (master.fold != inner_fold)
    inner_valid = master.fold == inner_fold
    outer_train = master.fold != outer_fold
    outer_valid = master.fold == outer_fold
    if set(master.loc[selection_train, "homology_group"]) & set(master.loc[inner_valid, "homology_group"]):
        raise RuntimeError("Inner selection group leakage")
    if set(master.loc[outer_train, "homology_group"]) & set(master.loc[outer_valid, "homology_group"]):
        raise RuntimeError("Outer group leakage")

    scaler_selection = StandardScaler()
    x_selection = scaler_selection.fit_transform(x[selection_train]).astype(np.float32)
    x_inner = scaler_selection.transform(x[inner_valid]).astype(np.float32)
    seed = base_seed + outer_fold * 100 + int(candidate.candidate[1:])
    selected_epochs, best_inner_ap, history = choose_epoch(
        x_selection,
        master.loc[selection_train, "label"].to_numpy(dtype=np.float32),
        master.loc[selection_train, "teacher_logit"].to_numpy(dtype=np.float32),
        x_inner,
        master.loc[inner_valid, "label"].to_numpy(dtype=np.int64),
        candidate,
        protocol,
        device,
        seed,
    )

    scaler_final = StandardScaler()
    x_outer_train = scaler_final.fit_transform(x[outer_train]).astype(np.float32)
    x_outer_valid = scaler_final.transform(x[outer_valid]).astype(np.float32)
    final_seed = seed + 50000
    model = fit_for_epochs(
        x_outer_train,
        master.loc[outer_train, "label"].to_numpy(dtype=np.float32),
        master.loc[outer_train, "teacher_logit"].to_numpy(dtype=np.float32),
        candidate,
        protocol,
        device,
        selected_epochs,
        final_seed,
    )
    raw, probability = predict(model, x_outer_valid, device, int(protocol["student"]["batch_size"]) * 2)
    valid = master.loc[outer_valid, ["clean_id", "label", "homology_group", "fold", "teacher_logit"]].copy()
    valid.insert(0, "candidate", candidate.candidate)
    valid["student_raw_logit"] = raw
    valid["probability_uncalibrated"] = probability
    valid["prediction_at_0_5"] = (probability >= 0.5).astype(int)
    valid.to_csv(result_path, sep="\t", index=False)

    checkpoint = {
        "module": "V2-04",
        "candidate": asdict(candidate),
        "outer_fold": outer_fold,
        "inner_fold": inner_fold,
        "selected_epochs": selected_epochs,
        "model_width": int(x.shape[1]),
        "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "scaler_mean": scaler_final.mean_.astype(np.float64),
        "scaler_scale": scaler_final.scale_.astype(np.float64),
        "canonical_mean_only": True,
        "esmc_frozen": True,
    }
    torch.save(checkpoint, checkpoint_path)
    meta = {
        "status": "PASS",
        "candidate": candidate.candidate,
        "outer_fold": outer_fold,
        "inner_fold": inner_fold,
        "outer_train_records": int(outer_train.sum()),
        "outer_validation_records": int(outer_valid.sum()),
        "inner_selection_train_records": int(selection_train.sum()),
        "inner_validation_records": int(inner_valid.sum()),
        "outer_shared_groups": 0,
        "inner_shared_groups": 0,
        "selected_epochs": selected_epochs,
        "best_inner_ap": best_inner_ap,
        "epoch_history": history,
        "result_sha256": sha256(result_path),
        "checkpoint_sha256": sha256(checkpoint_path),
    }
    atomic_json(meta_path, meta)
    append_progress(progress_path, {"utc": utc_now(), "candidate": candidate.candidate, "fold": outer_fold, "state": "PASS", "selected_epochs": selected_epochs, "best_inner_ap": best_inner_ap})
    return valid


def fold_metrics(oof: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for candidate, candidate_frame in oof.groupby("candidate", sort=True):
        for fold, fold_frame in candidate_frame.groupby("fold", sort=True):
            metrics = metric_dict(fold_frame.label.to_numpy(), fold_frame.probability_uncalibrated.to_numpy())
            rows.append({"candidate": candidate, "fold": int(fold), **metrics, "shared_strict40_groups": 0})
    return pd.DataFrame(rows)


def overall_metrics(oof: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for candidate, frame in oof.groupby("candidate", sort=True):
        rows.append({"candidate": candidate, **metric_dict(frame.label.to_numpy(), frame.probability_uncalibrated.to_numpy())})
    return pd.DataFrame(rows)


def build_gates(overall: pd.DataFrame, folds: pd.DataFrame, strict25: pd.DataFrame) -> pd.DataFrame:
    base = overall.set_index("candidate").loc["K0"]
    strict_base = strict25.set_index("candidate").loc["K0"]
    fold_sd = folds.groupby("candidate")[["ap", "mcc", "auroc"]].std(ddof=1)
    rows: list[dict[str, Any]] = []
    for _, current in overall.sort_values("candidate").iterrows():
        candidate = str(current.candidate)
        delta_ap = float(current.ap - base.ap)
        delta_mcc = float(current.mcc - base.mcc)
        delta_auroc = float(current.auroc - base.auroc)
        magnitude = delta_ap >= 0.005 or delta_auroc >= 0.005 or delta_mcc >= 0.02
        stability_parts: list[bool] = []
        stability_detail: dict[str, float] = {}
        for metric in ("ap", "mcc", "auroc"):
            k0_sd = float(fold_sd.loc["K0", metric])
            candidate_sd = float(fold_sd.loc[candidate, metric])
            limit = max(k0_sd + 0.005, 1.25 * k0_sd)
            stability_parts.append(candidate_sd <= limit + 1e-12)
            stability_detail[f"fold_sd_{metric}"] = candidate_sd
            stability_detail[f"fold_sd_limit_{metric}"] = limit
        strict_current = strict25.set_index("candidate").loc[candidate]
        strict_delta_ap = float(strict_current.ap - strict_base.ap)
        strict_delta_mcc = float(strict_current.mcc - strict_base.mcc)
        strict_delta_auroc = float(strict_current.auroc - strict_base.auroc)
        strict_gate = strict_delta_ap >= -0.005 and strict_delta_auroc >= -0.005 and strict_delta_mcc >= -0.02
        final_gate = candidate != "K0" and magnitude and all(stability_parts) and strict_gate
        rows.append(
            {
                "candidate": candidate,
                "delta_ap": delta_ap,
                "delta_mcc": delta_mcc,
                "delta_auroc": delta_auroc,
                "magnitude_gate": magnitude,
                **stability_detail,
                "fold_stability_gate": all(stability_parts),
                "strict25_delta_ap": strict_delta_ap,
                "strict25_delta_mcc": strict_delta_mcc,
                "strict25_delta_auroc": strict_delta_auroc,
                "strict25_gate": strict_gate,
                "passes_all_kd_gates": final_gate,
            }
        )
    return pd.DataFrame(rows)


def group_bootstrap(selected: pd.DataFrame, base: pd.DataFrame, replicates: int, seed: int) -> pd.DataFrame:
    merged = base[["clean_id", "label", "homology_group", "probability_uncalibrated"]].merge(
        selected[["clean_id", "probability_uncalibrated"]],
        on="clean_id",
        suffixes=("_k0", "_selected"),
        validate="one_to_one",
    )
    groups = list(merged.groupby("homology_group", sort=False).indices.values())
    rng = np.random.default_rng(seed)
    deltas: list[dict[str, float | int]] = []
    for replicate in range(replicates):
        chosen = rng.integers(0, len(groups), size=len(groups))
        indices = np.concatenate([groups[i] for i in chosen])
        sample = merged.iloc[indices]
        if sample.label.nunique() < 2:
            continue
        k0 = metric_dict(sample.label.to_numpy(), sample.probability_uncalibrated_k0.to_numpy())
        kd = metric_dict(sample.label.to_numpy(), sample.probability_uncalibrated_selected.to_numpy())
        deltas.append({"replicate": replicate, "delta_ap": kd["ap"] - k0["ap"], "delta_mcc": kd["mcc"] - k0["mcc"], "delta_auroc": kd["auroc"] - k0["auroc"]})
    frame = pd.DataFrame(deltas)
    summary: list[dict[str, Any]] = []
    for metric in ("delta_ap", "delta_mcc", "delta_auroc"):
        values = frame[metric].to_numpy(dtype=float)
        summary.append(
            {
                "metric": metric,
                "replicates": int(len(values)),
                "mean": float(values.mean()),
                "ci_2_5": float(np.quantile(values, 0.025)),
                "median": float(np.quantile(values, 0.5)),
                "ci_97_5": float(np.quantile(values, 0.975)),
                "fraction_gt_zero": float((values > 0).mean()),
            }
        )
    return pd.DataFrame(summary)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--strict25-membership", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--threads", type=int, default=16)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    lock_handle = (output / ".V2_04.lock").open("a+", encoding="utf-8")
    try:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print(f"Another V2-04 process holds the lock: {output / '.V2_04.lock'}", file=sys.stderr)
        return 3
    lock_handle.seek(0)
    lock_handle.truncate()
    lock_handle.write(f"pid={os.getpid()} started_at_utc={utc_now()}\n")
    lock_handle.flush()
    status_path = output / "STATUS.json"
    access_log: list[dict[str, Any]] = []
    started = time.time()
    atomic_json(status_path, {"module": "V2-04", "status": "STARTED", "started_at_utc": utc_now(), "blind_labels_read": False, "lora": False})
    try:
        for role, path in (("root", args.root), ("output", output), ("protocol", args.protocol)):
            lowered = str(path.expanduser().resolve()).lower()
            hits = [token for token in FORBIDDEN_PATH_TOKENS if token in lowered]
            if hits:
                raise RuntimeError(f"Forbidden {role} path tokens={hits}: {path}")
        protocol_path = guarded_input(args.protocol, "protocol", access_log)
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        if protocol.get("protocol_status") != "LOCKED_BEFORE_EXECUTION":
            raise RuntimeError("Protocol is not locked")
        if args.threads < 1:
            raise ValueError("--threads must be positive")
        torch.set_num_threads(args.threads)
        if args.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable; do not silently change backend")
        device = torch.device(args.device)
        v203b2_gate = validate_v203b2_gate(args.root.resolve(), access_log)
        atomic_json(status_path, {"module": "V2-04", "status": "PREFLIGHT_BINDING", "updated_at_utc": utc_now(), "blind_labels_read": False, "lora": False})
        x, master, strict_mask, binding = load_bind_inputs(args, protocol, access_log)
        binding["v203b2_gate"] = v203b2_gate
        atomic_json(output / "V2_04_BINDING_AUDIT.json", binding)
        atomic_json(
            output / "INPUT_SHA256.json",
            {
                "protocol": {"path": str(protocol_path), "sha256": sha256(protocol_path)},
                **binding["input_hashes"],
            },
        )
        atomic_json(
            output / "RUNTIME.json",
            {
                "python": sys.version,
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "sklearn": sklearn.__version__,
                "torch": torch.__version__,
                "device": str(device),
                "cuda_device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                "threads": args.threads,
            },
        )

        candidates = [Candidate(candidate=name, **values) for name, values in protocol["candidates"].items()]
        atomic_json(status_path, {"module": "V2-04", "status": "TRAINING", "updated_at_utc": utc_now(), "blind_labels_read": False, "lora": False})
        frames: list[pd.DataFrame] = []
        progress_path = output / "PROGRESS.tsv"
        for candidate in candidates:
            for outer_fold in protocol["folds_expected"]:
                frames.append(
                    run_candidate_fold(
                        candidate,
                        int(outer_fold),
                        x,
                        master,
                        protocol,
                        device,
                        output,
                        progress_path,
                        int(protocol["seed"]),
                    )
                )
        oof = pd.concat(frames, ignore_index=True)
        expected_rows = len(master) * len(candidates)
        if len(oof) != expected_rows or oof.duplicated(["candidate", "clean_id"]).any():
            raise RuntimeError("OOF assembly is incomplete or duplicated")
        if oof.groupby(["candidate", "homology_group"]).fold.nunique().max() != 1:
            raise RuntimeError("OOF output violates group-fold binding")
        oof = oof.sort_values(["candidate", "clean_id"], kind="mergesort").reset_index(drop=True)
        oof.to_csv(output / "all_candidate_oof_predictions.tsv", sep="\t", index=False)

        overall = overall_metrics(oof)
        folds = fold_metrics(oof)
        overall.to_csv(output / "candidate_overall_metrics.tsv", sep="\t", index=False)
        folds.to_csv(output / "candidate_fold_metrics.tsv", sep="\t", index=False)
        strict_ids = set(master.loc[strict_mask, "clean_id"])
        strict_oof = oof[oof.clean_id.isin(strict_ids)]
        strict_metrics = overall_metrics(strict_oof)
        strict_metrics.to_csv(output / "strict25_metrics.tsv", sep="\t", index=False)
        gates = build_gates(overall, folds, strict_metrics)
        gates.to_csv(output / "candidate_gain_gates.tsv", sep="\t", index=False)

        passed = gates[gates.passes_all_kd_gates.astype(bool)].candidate.tolist()
        if passed:
            ranked = overall[overall.candidate.isin(passed)].sort_values(
                ["ap", "mcc", "auroc", "candidate"],
                ascending=[False, False, False, True],
                kind="mergesort",
            )
            selected_name = str(ranked.iloc[0].candidate)
            decision = "KEEP_KD"
            status = "PASS_KD_GAIN"
        else:
            selected_name = "K0"
            decision = "STOP_KD_NO_STABLE_GAIN"
            status = "PASS_SCREEN_COMPLETE_NO_KD_GAIN"
        selected = {
            "status": status,
            "decision": decision,
            "selected_candidate": selected_name,
            "eligible_kd_candidates": passed,
            "selection_priority": protocol["selection_priority"],
            "canonical_mean_only": True,
            "deepSaltPro_required_at_inference": False,
            "blind_labels_read": False,
            "lora": False,
            "next_gate": "V2-05 only after V2-04 review",
        }
        atomic_json(output / "selected_kd_candidate.json", selected)
        bootstrap = group_bootstrap(
            oof[oof.candidate == selected_name],
            oof[oof.candidate == "K0"],
            int(protocol["bootstrap"]["replicates"]),
            int(protocol["seed"]) + 900000,
        )
        bootstrap.to_csv(output / "selected_vs_k0_group_bootstrap.tsv", sep="\t", index=False)

        access_audit = {
            "status": "PASS_ALLOWED_INPUTS_ONLY",
            "accessed_inputs": access_log,
            "forbidden_tokens": list(FORBIDDEN_PATH_TOKENS),
            "blind_labels_read": False,
            "test200_labels_read": False,
            "external_challenge_labels_read": False,
            "esmc_loaded_or_updated": False,
            "lora": False,
        }
        atomic_json(output / "DATA_ACCESS_AUDIT.json", access_audit)
        finished = time.time()
        done = {
            "module": "V2-04",
            "status": status,
            "decision": decision,
            "selected_candidate": selected_name,
            "started_at_utc": datetime.fromtimestamp(started, timezone.utc).isoformat(),
            "finished_at_utc": utc_now(),
            "elapsed_seconds": round(finished - started, 3),
            "records": int(len(master)),
            "candidates": [c.candidate for c in candidates],
            "outer_folds_complete": 5,
            "strict40_shared_groups": 0,
            "strict25_records": int(strict_mask.sum()),
            "canonical_mean_only": True,
            "labels_changed": 0,
            "weights_applied": False,
            "calibration_applied": False,
            "threshold_optimized": False,
            "hard_negative_training": False,
            "lora": False,
            "blind_labels_read": False,
            "test200_labels_read": False,
            "external_challenge_labels_read": False,
            "next_gate": "V2-05 only after review",
        }
        atomic_json(output / "V2_04_DONE.json", done)
        atomic_json(status_path, done)
        return 0
    except Exception as exc:
        failure = {
            "module": "V2-04",
            "status": "FAILED",
            "updated_at_utc": utc_now(),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "blind_labels_read": False,
            "lora": False,
        }
        atomic_json(status_path, failure)
        atomic_json(
            output / "DATA_ACCESS_AUDIT.json",
            {
                "status": "FAILED_BEFORE_OR_DURING_ALLOWED_INPUT_PROCESSING",
                "accessed_inputs": access_log,
                "blind_labels_read": False,
                "test200_labels_read": False,
                "external_challenge_labels_read": False,
                "lora": False,
            },
        )
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
