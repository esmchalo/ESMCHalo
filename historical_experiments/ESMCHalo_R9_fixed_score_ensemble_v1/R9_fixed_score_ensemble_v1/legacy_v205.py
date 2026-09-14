#!/usr/bin/env python3
"""V2-05A development-only hard-sample weighting ablation."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import random
import re
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import sklearn
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (accuracy_score, average_precision_score,
                             brier_score_loss, confusion_matrix,
                             matthews_corrcoef, precision_score, recall_score,
                             roc_auc_score)
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

EXPECTED_SHA256 = {
    "folds": "68119ccd8a9c03244c2b23356b52bc9c20a4c1ba6319970917a8bca53863205d",
    "registry": "d02f0cb31d7b599d1de4780f61d3bda6b26390671dc42053ef1ba1c8dc2b0ee5",
    "canonical_rows": "48e032b4ef7cfcaf2a8b03037111fb0f1d75aec1c9db51a832cca9f54502d39f",
    "canonical_matrix": "04dd4e5a0e0d6738de5ad97a962b9fbd1bc514e2ce04333db3f5875e2ffc837e",
    "teacher_oof": "e3715f16f7f32fb0bd28f48250c5c0bf748f8666762ba52e2a5d853d508915b2",
    "v204_done": "e2403dbda79a4e4ce3b512d214653184cf7bb8b00accd5588b81508ddd565027",
    "v204_selection": "520d372900467f67aa12d96d9c02c1e11be0ad877e67719543627e16a6535c4e",
    "v204_oof": "1cc10ccaff97df186f784fdbabe0b8a7f13f8ddbe5693d7505549e1b76152494",
    "v204_binding": "e18a3b95b03fac800c66ca29ea0241cb72865d6ee482ece83db29724af78b06a",
}
FORBIDDEN = ("blind", "test200", "external", "challenge")
IDS = ("clean_id", "id", "protein_id", "sample_id", "record_id")
LABELS = ("label", "y", "target", "class")
FOLDS = ("fold", "outer_fold", "fold_id", "oof_fold")
GROUPS = ("homology_group", "strict40_group", "cluster_id", "group", "mmseqs40_cluster")
STRICT25 = ("strict25", "is_strict25", "strict25_member", "strict25_subset", "strict25_eval", "strict25_diagnostic")
TIERS = ("audit_tier", "quality_tier", "tier", "audit_category", "audit_level", "training_tier")
ARMS = ("H0", "H1", "H2", "H3")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, obj: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def guarded(path: Path, role: str, log: list[dict[str, Any]]) -> Path:
    p = path.expanduser().resolve()
    hits = [x for x in FORBIDDEN if x in str(p).lower()]
    if hits:
        raise RuntimeError(f"Forbidden input path for {role}: {hits}: {p}")
    if not p.is_file():
        raise FileNotFoundError(f"Missing input for {role}: {p}")
    log.append({"role": role, "path": str(p), "opened_at_utc": now()})
    return p


def pick(df: pd.DataFrame, names: Iterable[str], role: str) -> str:
    lower = {str(c).lower(): str(c) for c in df.columns}
    for name in names:
        if name.lower() in lower:
            return lower[name.lower()]
    raise ValueError(f"Cannot resolve {role}; columns={list(df.columns)}")


def optional(df: pd.DataFrame, names: Iterable[str]) -> str | None:
    lower = {str(c).lower(): str(c) for c in df.columns}
    return next((lower[n.lower()] for n in names if n.lower() in lower), None)


def read_tsv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", low_memory=False)


def binary(series: pd.Series, role: str) -> np.ndarray:
    m = {"0": 0, "1": 1, "false": 0, "true": 1, "no": 0, "yes": 1, "n": 0, "y": 1}
    out = []
    for value in series:
        key = str(value).strip().lower()
        if key in m:
            out.append(m[key])
        else:
            try:
                v = int(float(key))
            except Exception as exc:
                raise ValueError(f"Non-binary {role}: {value!r}") from exc
            if v not in (0, 1):
                raise ValueError(f"Non-binary {role}: {value!r}")
            out.append(v)
    return np.asarray(out, dtype=np.int64)


def metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float | int]:
    y = np.asarray(y, dtype=np.int64)
    p = np.asarray(p, dtype=np.float64)
    pred = (p >= 0.5).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "n": int(len(y)), "accuracy": float(accuracy_score(y, pred)),
        "mcc": float(matthews_corrcoef(y, pred)), "auroc": float(roc_auc_score(y, p)),
        "ap": float(average_precision_score(y, p)),
        "sensitivity": float(recall_score(y, pred, pos_label=1, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if tn + fp else float("nan"),
        "ppv": float(precision_score(y, pred, pos_label=1, zero_division=0)),
        "brier_uncalibrated": float(brier_score_loss(y, p)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def seed_all(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


class Student(nn.Module):
    def __init__(self, width: int, dropout: float):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(width, 256), nn.GELU(), nn.Dropout(dropout),
                                     nn.Linear(256, 64), nn.GELU(), nn.Dropout(dropout), nn.Linear(64, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(-1)


def loss_value(student: torch.Tensor, y: torch.Tensor, teacher: torch.Tensor,
               weights: torch.Tensor, arm: str) -> torch.Tensor:
    t = 2.0
    if arm == "H0":
        hard = F.binary_cross_entropy_with_logits(student, y)
        soft = F.binary_cross_entropy_with_logits(student / t, torch.sigmoid(teacher / t))
        return 0.5 * hard + 0.5 * (t ** 2) * soft
    hard = F.binary_cross_entropy_with_logits(student, y, reduction="none")
    soft = F.binary_cross_entropy_with_logits(student / t, torch.sigmoid(teacher / t), reduction="none")
    per_row = 0.5 * hard + 0.5 * (t ** 2) * soft
    return (weights * per_row).sum() / weights.sum()


def loader(x: np.ndarray, y: np.ndarray, teacher: np.ndarray, weights: np.ndarray,
           batch: int, shuffle: bool, seed: int) -> DataLoader:
    ds = TensorDataset(*(torch.from_numpy(np.asarray(a, dtype=np.float32))
                         for a in (x, y, teacher, weights)))
    gen = torch.Generator(); gen.manual_seed(seed)
    return DataLoader(ds, batch_size=batch, shuffle=shuffle, generator=gen,
                      num_workers=0, pin_memory=torch.cuda.is_available(), drop_last=False)


@torch.no_grad()
def predict(model: nn.Module, x: np.ndarray, device: torch.device, batch: int) -> tuple[np.ndarray, np.ndarray]:
    model.eval(); parts = []
    for bx in DataLoader(torch.from_numpy(np.asarray(x, dtype=np.float32)), batch_size=batch, shuffle=False):
        parts.append(model(bx.to(device, non_blocking=True)).float().cpu().numpy())
    raw = np.concatenate(parts).astype(np.float64)
    return raw, 1.0 / (1.0 + np.exp(-np.clip(raw, -50, 50)))


def optimizer(model: nn.Module, spec: dict[str, Any]) -> torch.optim.Optimizer:
    return torch.optim.AdamW(model.parameters(), lr=float(spec["learning_rate"]),
                             weight_decay=float(spec["weight_decay"]))


def choose_epoch(x: np.ndarray, y: np.ndarray, teacher: np.ndarray, weights: np.ndarray,
                 x_valid: np.ndarray, y_valid: np.ndarray, arm: str, protocol: dict[str, Any],
                 device: torch.device, seed: int) -> tuple[int, float, list[dict[str, Any]]]:
    spec = protocol["student"]; seed_all(seed)
    model = Student(x.shape[1], float(spec["dropout"])).to(device); opt = optimizer(model, spec)
    dl = loader(x, y, teacher, weights, int(spec["batch_size"]), True, seed)
    best, best_epoch, stale, history = -math.inf, 1, 0, []
    for epoch in range(1, int(spec["maximum_epochs"]) + 1):
        model.train(); total = 0.0; batches = 0
        for bx, by, bt, bw in dl:
            bx, by, bt, bw = (v.to(device, non_blocking=True) for v in (bx, by, bt, bw))
            opt.zero_grad(set_to_none=True); logits = model(bx)
            loss = loss_value(logits, by, bt, bw, arm)
            if not bool(torch.isfinite(loss)): raise FloatingPointError(f"Non-finite loss {arm}")
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), float(spec["gradient_clip_norm"])); opt.step()
            total += float(loss.detach().cpu()); batches += 1
        _, p = predict(model, x_valid, device, int(spec["batch_size"]) * 2)
        ap = float(average_precision_score(y_valid, p))
        history.append({"epoch": epoch, "mean_training_loss": total / max(batches, 1), "inner_ap": ap})
        if ap > best + 1e-6: best, best_epoch, stale = ap, epoch, 0
        else: stale += 1
        if stale >= int(spec["early_stopping_patience"]): break
    return best_epoch, best, history


def fit_epochs(x: np.ndarray, y: np.ndarray, teacher: np.ndarray, weights: np.ndarray,
               arm: str, protocol: dict[str, Any], device: torch.device, epochs: int, seed: int) -> Student:
    spec = protocol["student"]; seed_all(seed)
    model = Student(x.shape[1], float(spec["dropout"])).to(device); opt = optimizer(model, spec)
    dl = loader(x, y, teacher, weights, int(spec["batch_size"]), True, seed)
    for _ in range(epochs):
        model.train()
        for bx, by, bt, bw in dl:
            bx, by, bt, bw = (v.to(device, non_blocking=True) for v in (bx, by, bt, bw))
            opt.zero_grad(set_to_none=True); logits = model(bx); loss = loss_value(logits, by, bt, bw, arm)
            if not bool(torch.isfinite(loss)): raise FloatingPointError(f"Non-finite loss {arm}")
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), float(spec["gradient_clip_norm"])); opt.step()
    return model


def teacher_logits(df: pd.DataFrame) -> tuple[np.ndarray, dict[str, Any]]:
    col = optional(df, ("teacher_logit", "raw_logit", "logit", "deepsaltpro_logit"))
    if col is None:
        raise ValueError("Teacher OOF must contain raw uncalibrated logits")
    v = pd.to_numeric(df[col], errors="raise").to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(v)): raise ValueError("Non-finite teacher logits")
    return v, {"source_column": col, "conversion": "raw_logit", "minimum": float(v.min()), "maximum": float(v.max())}


def tier_masks(registry: pd.DataFrame, id_col: str, master: pd.DataFrame,
               protocol: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    col = pick(registry, TIERS, "V2-02 audit tier")
    table = pd.Series(registry[col].astype(str).to_numpy(), index=registry[id_col].astype(str))
    raw = master.clean_id.map(table)
    if raw.isna().any(): raise ValueError("Audit tier does not cover development IDs")
    norm = raw.map(lambda s: re.sub(r"[^a-z0-9]+", "_", str(s).strip().lower()).strip("_"))
    clean = norm.map(lambda s: s in {"a", "a_clean", "a_clean_core"} or (s.startswith("a_") and "clean" in s)).to_numpy(bool)
    hard = norm.map(lambda s: s in {"d", "d_hard", "d_hard_sample"} or (s.startswith("d_") and "hard" in s)).to_numpy(bool)
    expected = protocol["expected_audit_counts"]
    if int(clean.sum()) != int(expected["A_clean_core"]) or int(hard.sum()) != int(expected["D_hard"]):
        raise ValueError(f"Audit tier counts mismatch: column={col}, values={norm.value_counts().to_dict()}, A={clean.sum()}, D={hard.sum()}")
    return clean, hard, {"column": col, "normalized_counts": {str(k): int(v) for k, v in norm.value_counts().items()},
                         "A_clean_core": int(clean.sum()), "D_hard": int(hard.sum())}


def load_inputs(args: argparse.Namespace, protocol: dict[str, Any], log: list[dict[str, Any]]):
    root = args.root.resolve()
    v204 = root / "ESMCHalo_v2_optimization/V2_04_knowledge_distillation/V2_04_deepsaltpro_oof_kd_v1"
    paths = {
        "folds": root / "ESMCHalo_v2_optimization/V2_01_baseline_and_teacher_oof/group_folds_v2.tsv",
        "registry": root / "ESMCHalo_v2_optimization/V2_02_label_audit/V2_02B_audit_lock_v1/V2_02_LOCKED_TRAINING_REGISTRY.tsv",
        "canonical_rows": root / "data/embeddings/esmc600m_canonical_b1/rows.tsv",
        "canonical_matrix": root / "data/embeddings/esmc600m_canonical_b1/embeddings.npy",
        "teacher_oof": root / "ESMCHalo_v2_optimization/V2_01_baseline_and_teacher_oof/deepsaltpro_nested_oof_full_v1/deepsaltpro_oof_predictions.tsv",
        "v204_done": v204 / "V2_04_DONE.json", "v204_selection": v204 / "selected_kd_candidate.json",
        "v204_oof": v204 / "all_candidate_oof_predictions.tsv", "v204_binding": v204 / "V2_04_BINDING_AUDIT.json",
    }
    safe, hashes = {}, {}
    for role, path in paths.items():
        safe[role] = guarded(path, role, log); actual = sha256(safe[role])
        if actual != EXPECTED_SHA256[role]: raise RuntimeError(f"SHA256 mismatch {role}: expected={EXPECTED_SHA256[role]} actual={actual}")
        hashes[role] = {"path": str(safe[role]), "sha256": actual, "status": "PASS"}
    done = json.loads(safe["v204_done"].read_text()); selected = json.loads(safe["v204_selection"].read_text())
    if done.get("status") != "PASS_KD_GAIN" or done.get("selected_candidate") != "K2": raise RuntimeError("V2-04 DONE gate failed")
    if selected.get("decision") != "KEEP_KD" or selected.get("selected_candidate") != "K2": raise RuntimeError("Frozen K2 selection gate failed")
    if any(done.get(k) is not False for k in ("blind_labels_read", "test200_labels_read", "external_challenge_labels_read", "lora")):
        raise RuntimeError("V2-04 boundary audit failed")

    folds, registry, rows, teacher = (read_tsv(safe[x]) for x in ("folds", "registry", "canonical_rows", "teacher_oof"))
    matrix = np.load(safe["canonical_matrix"], mmap_mode="r", allow_pickle=False)
    fid, fcol, gcol = pick(folds, IDS, "fold ID"), pick(folds, FOLDS, "fold"), pick(folds, GROUPS, "group")
    rid, lcol = pick(registry, IDS, "registry ID"), pick(registry, LABELS, "label")
    xid, xrow = pick(rows, IDS, "row ID"), pick(rows, ("embedding_row", "row", "row_index", "index"), "embedding row")
    tid = pick(teacher, IDS, "teacher ID")
    for name, frame, col in (("folds", folds, fid), ("registry", registry, rid), ("rows", rows, xid), ("teacher", teacher, tid)):
        if frame[col].astype(str).duplicated().any(): raise ValueError(f"Duplicate IDs in {name}")
    master = pd.DataFrame({"clean_id": folds[fid].astype(str), "fold": pd.to_numeric(folds[fcol]).astype(int),
                           "homology_group": folds[gcol].astype(str)})
    label_map = pd.Series(binary(registry[lcol], "label"), index=registry[rid].astype(str))
    row_map = pd.Series(pd.to_numeric(rows[xrow]).astype(int).to_numpy(), index=rows[xid].astype(str))
    tv, tmeta = teacher_logits(teacher); teacher_map = pd.Series(tv, index=teacher[tid].astype(str))
    expected = set(master.clean_id)
    for name, idx in (("registry", label_map.index), ("teacher", teacher_map.index)):
        if set(idx) != expected: raise ValueError(f"Exact ID mismatch for {name}")
    if not expected.issubset(set(row_map.index)): raise ValueError("Canonical store misses development IDs")
    master["label"] = master.clean_id.map(label_map).astype(int); master["embedding_row"] = master.clean_id.map(row_map).astype(int)
    master["teacher_logit"] = master.clean_id.map(teacher_map).astype(float)
    if len(master) != int(protocol["records_expected"]) or set(master.fold) != set(protocol["folds_expected"]): raise ValueError("Development/fold gate failed")
    if master.groupby("homology_group").fold.nunique().max() != 1: raise ValueError("strict40 group crosses folds")
    if matrix.ndim != 2 or matrix.shape[1] != 1152 or matrix.shape[0] != len(rows): raise ValueError(f"Canonical shape mismatch {matrix.shape}")
    idx = master.embedding_row.to_numpy(np.int64); x = np.asarray(matrix[idx], dtype=np.float32)
    if not np.all(np.isfinite(x)) or len(np.unique(idx)) != len(idx): raise ValueError("Invalid canonical development matrix")

    scol = pick(folds, STRICT25, "strict25 membership")
    smap = pd.Series(binary(folds[scol], "strict25"), index=folds[fid].astype(str))
    strict = master.clean_id.map(smap).to_numpy(np.int64).astype(bool)
    if int(strict.sum()) != 10458: raise ValueError(f"strict25 count mismatch: {strict.sum()}")
    clean, dhard, tiermeta = tier_masks(registry, rid, master, protocol)

    frozen = read_tsv(safe["v204_oof"]); k2 = frozen[frozen["candidate"].astype(str) == "K2"].copy()
    required = {"clean_id", "label", "homology_group", "fold", "probability_uncalibrated"}
    if not required.issubset(k2.columns) or len(k2) != len(master) or not k2.clean_id.is_unique: raise ValueError("Frozen K2 OOF structure failed")
    k2 = master[["clean_id", "label", "homology_group", "fold"]].merge(
        k2[["clean_id", "label", "homology_group", "fold", "probability_uncalibrated"]], on="clean_id", suffixes=("", "_frozen"), validate="one_to_one")
    for col in ("label", "fold", "homology_group"):
        if not np.array_equal(k2[col].astype(str).to_numpy(), k2[f"{col}_frozen"].astype(str).to_numpy()): raise ValueError(f"Frozen K2 {col} mismatch")
    fp = ((k2.label == 0) & (pd.to_numeric(k2.probability_uncalibrated) >= 0.5)).to_numpy(bool)
    if int(fp.sum()) != int(protocol["expected_audit_counts"]["K2_oof_false_positive"]): raise ValueError(f"K2 OOF FP count mismatch: {fp.sum()}")
    flags = pd.DataFrame({"clean_id": master.clean_id, "D_hard": dhard, "K2_oof_false_positive": fp,
                          "H3_union": dhard | fp})
    flag_counts = {"D_hard": int(dhard.sum()), "K2_oof_false_positive": int(fp.sum()),
                   "intersection": int((dhard & fp).sum()), "union": int((dhard | fp).sum())}
    bind = {"status": "PASS", "records": len(master), "positive": int(master.label.sum()), "negative": int((1-master.label).sum()),
            "unique_strict40_groups": int(master.homology_group.nunique()), "maximum_folds_per_group": 1,
            "fold_sizes": {str(k): int(v) for k, v in master.fold.value_counts().sort_index().items()},
            "canonical_store_shape": list(matrix.shape), "canonical_selected_shape": list(x.shape),
            "strict25": {"column": scol, "records": int(strict.sum())}, "audit_tiers": tiermeta,
            "hard_flags": flag_counts, "teacher": tmeta, "input_hashes": hashes,
            "labels_changed": 0, "blind_test200_external_labels_read": False, "lora": False,
            "candidate_negative_pool_used": False}
    k2ref = k2[["clean_id", "label", "homology_group", "fold", "probability_uncalibrated"]].copy()
    return x, master, strict, clean, flags, k2ref, bind


def arm_weights(flags: pd.DataFrame, arm: str) -> np.ndarray:
    selected = {"H0": np.zeros(len(flags), bool), "H1": flags.D_hard.to_numpy(bool),
                "H2": flags.K2_oof_false_positive.to_numpy(bool), "H3": flags.H3_union.to_numpy(bool)}[arm]
    return np.where(selected, 1.5, 1.0).astype(np.float32)


def progress(path: Path, row: dict[str, Any]) -> None:
    pd.DataFrame([row]).to_csv(path, sep="\t", index=False, mode="a", header=not path.exists())


def run_fold(arm: str, fold: int, x: np.ndarray, master: pd.DataFrame, flags: pd.DataFrame,
             protocol: dict[str, Any], device: torch.device, out: Path) -> pd.DataFrame:
    directory = out / "partial" / arm; directory.mkdir(parents=True, exist_ok=True)
    tsv, meta, ckpt = directory/f"fold_{fold}.tsv", directory/f"fold_{fold}.json", directory/f"fold_{fold}.pt"
    if tsv.is_file() and meta.is_file() and ckpt.is_file():
        m = json.loads(meta.read_text())
        if m.get("status") == "PASS" and m.get("result_sha256") == sha256(tsv) and m.get("checkpoint_sha256") == sha256(ckpt):
            reused = read_tsv(tsv)
            if len(reused) == int((master.fold == fold).sum()) and reused.clean_id.is_unique:
                progress(out/"PROGRESS.tsv", {"utc": now(), "arm": arm, "fold": fold, "state": "REUSED_VALID_CHECKPOINT"}); return reused
    inner = (fold + 1) % 5; select_train = (master.fold != fold) & (master.fold != inner)
    inner_valid, outer_train, outer_valid = master.fold == inner, master.fold != fold, master.fold == fold
    if set(master.loc[outer_train, "homology_group"]) & set(master.loc[outer_valid, "homology_group"]): raise RuntimeError("Outer leakage")
    weights = arm_weights(flags, arm)
    ss = StandardScaler(); xs = ss.fit_transform(x[select_train]).astype(np.float32); xv = ss.transform(x[inner_valid]).astype(np.float32)
    seed = int(protocol["seed"]) + fold * 100 + int(protocol["k2_seed_offset"])
    epochs, best, history = choose_epoch(xs, master.loc[select_train,"label"].to_numpy(np.float32),
        master.loc[select_train,"teacher_logit"].to_numpy(np.float32), weights[select_train], xv,
        master.loc[inner_valid,"label"].to_numpy(np.int64), arm, protocol, device, seed)
    sf = StandardScaler(); xt = sf.fit_transform(x[outer_train]).astype(np.float32); xo = sf.transform(x[outer_valid]).astype(np.float32)
    model = fit_epochs(xt, master.loc[outer_train,"label"].to_numpy(np.float32),
        master.loc[outer_train,"teacher_logit"].to_numpy(np.float32), weights[outer_train], arm,
        protocol, device, epochs, seed + 50000)
    raw, p = predict(model, xo, device, int(protocol["student"]["batch_size"]) * 2)
    result = master.loc[outer_valid, ["clean_id","label","homology_group","fold","teacher_logit"]].copy()
    result.insert(0, "arm", arm); result["student_raw_logit"] = raw; result["probability_uncalibrated"] = p
    result["prediction_at_0_5"] = (p >= 0.5).astype(int); result.to_csv(tsv, sep="\t", index=False)
    torch.save({"module":"V2-05A","arm":arm,"outer_fold":fold,"inner_fold":inner,"selected_epochs":epochs,
                "model_state_dict":{k:v.detach().cpu() for k,v in model.state_dict().items()},
                "scaler_mean":sf.mean_.astype(np.float64),"scaler_scale":sf.scale_.astype(np.float64),
                "canonical_mean_only":True,"k2_fixed":True,"lora":False,"upweight":1.0 if arm=="H0" else 1.5}, ckpt)
    atomic_json(meta, {"status":"PASS","arm":arm,"outer_fold":fold,"inner_fold":inner,
        "outer_train_records":int(outer_train.sum()),"outer_validation_records":int(outer_valid.sum()),
        "weighted_outer_train_records":int((weights[outer_train] > 1).sum()),"selected_epochs":epochs,
        "best_inner_ap":best,"epoch_history":history,"result_sha256":sha256(tsv),"checkpoint_sha256":sha256(ckpt),
        "outer_shared_groups":0,"validation_flags_used_for_training":False})
    progress(out/"PROGRESS.tsv", {"utc":now(),"arm":arm,"fold":fold,"state":"PASS","selected_epochs":epochs,"best_inner_ap":best})
    return result


def metric_tables(oof: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    overall, folds = [], []
    for arm, af in oof.groupby("arm", sort=True):
        overall.append({"arm":arm, **metrics(af.label, af.probability_uncalibrated)})
        for fold, ff in af.groupby("fold", sort=True):
            folds.append({"arm":arm,"fold":int(fold),**metrics(ff.label,ff.probability_uncalibrated),"shared_strict40_groups":0})
    return pd.DataFrame(overall), pd.DataFrame(folds)


def subset_metrics(oof: pd.DataFrame, ids: set[str]) -> pd.DataFrame:
    rows=[]
    for arm, frame in oof[oof.clean_id.isin(ids)].groupby("arm",sort=True): rows.append({"arm":arm,**metrics(frame.label,frame.probability_uncalibrated)})
    return pd.DataFrame(rows)


def h0_reproduction(oof: pd.DataFrame, ref: pd.DataFrame, protocol: dict[str, Any]) -> dict[str, Any]:
    h0 = oof[oof.arm == "H0"][["clean_id","label","homology_group","fold","probability_uncalibrated"]]
    m = ref.merge(h0, on="clean_id", suffixes=("_v204","_h0"), validate="one_to_one")
    identity = all(np.array_equal(m[f"{c}_v204"].astype(str),m[f"{c}_h0"].astype(str)) for c in ("label","homology_group","fold"))
    max_abs = float(np.max(np.abs(m.probability_uncalibrated_v204-m.probability_uncalibrated_h0)))
    a, b = metrics(m.label_v204.to_numpy(int),m.probability_uncalibrated_v204), metrics(m.label_h0.to_numpy(int),m.probability_uncalibrated_h0)
    diffs={k:abs(float(a[k])-float(b[k])) for k in ("ap","mcc","auroc","accuracy")}
    passed = identity and max_abs <= float(protocol["h0_reproduction"]["probability_max_abs_tolerance"]) and max(diffs.values()) <= float(protocol["h0_reproduction"]["metric_abs_tolerance"])
    return {"status":"PASS_H0_REPRODUCTION" if passed else "FAIL_H0_REPRODUCTION","passed":passed,"identity_columns_exact":identity,
            "records":len(m),"probability_max_abs_difference":max_abs,"metric_abs_differences":diffs,
            "probability_tolerance":protocol["h0_reproduction"]["probability_max_abs_tolerance"],"metric_tolerance":protocol["h0_reproduction"]["metric_abs_tolerance"]}


def gates(overall: pd.DataFrame, folds: pd.DataFrame, strict: pd.DataFrame, clean: pd.DataFrame) -> pd.DataFrame:
    oi, si, ci = overall.set_index("arm"), strict.set_index("arm"), clean.set_index("arm")
    sd = folds.groupby("arm")[["ap","mcc","auroc"]].std(ddof=1); rows=[]
    for arm in ARMS:
        da, dm, du = float(oi.loc[arm,"ap"]-oi.loc["H0","ap"]), float(oi.loc[arm,"mcc"]-oi.loc["H0","mcc"]), float(oi.loc[arm,"auroc"]-oi.loc["H0","auroc"])
        magnitude = da>=.005 or du>=.005 or dm>=.02; detail={}; stable=[]
        for metric in ("ap","mcc","auroc"):
            limit=max(float(sd.loc["H0",metric])+.005,1.25*float(sd.loc["H0",metric])); value=float(sd.loc[arm,metric])
            detail[f"fold_sd_{metric}"]=value; detail[f"fold_sd_limit_{metric}"]=limit; stable.append(value<=limit+1e-12)
        sda=float(si.loc[arm,"ap"]-si.loc["H0","ap"]); sdm=float(si.loc[arm,"mcc"]-si.loc["H0","mcc"]); sdu=float(si.loc[arm,"auroc"]-si.loc["H0","auroc"])
        cda=float(ci.loc[arm,"ap"]-ci.loc["H0","ap"]); cdu=float(ci.loc[arm,"auroc"]-ci.loc["H0","auroc"])
        sens=float(oi.loc[arm,"sensitivity"]-oi.loc["H0","sensitivity"])
        sg=sda>=-.005 and sdu>=-.005 and sdm>=-.02; cg=cda>=-.005 and cdu>=-.005; seg=sens>=-.01
        passed=arm!="H0" and magnitude and all(stable) and sg and cg and seg
        rows.append({"arm":arm,"delta_ap":da,"delta_mcc":dm,"delta_auroc":du,"magnitude_gate":magnitude,**detail,
            "fold_stability_gate":all(stable),"strict25_delta_ap":sda,"strict25_delta_mcc":sdm,"strict25_delta_auroc":sdu,"strict25_gate":sg,
            "clean_core_delta_ap":cda,"clean_core_delta_auroc":cdu,"clean_core_gate":cg,"sensitivity_delta":sens,"sensitivity_gate":seg,
            "passes_all_hard_sample_gates":passed})
    return pd.DataFrame(rows)


def bootstrap(selected: pd.DataFrame, base: pd.DataFrame, reps: int, seed: int) -> pd.DataFrame:
    m=base[["clean_id","label","homology_group","probability_uncalibrated"]].merge(selected[["clean_id","probability_uncalibrated"]],on="clean_id",suffixes=("_h0","_selected"),validate="one_to_one")
    groups=list(m.groupby("homology_group",sort=False).indices.values()); rng=np.random.default_rng(seed); rows=[]
    for rep in range(reps):
        ix=np.concatenate([groups[i] for i in rng.integers(0,len(groups),size=len(groups))]); s=m.iloc[ix]
        if s.label.nunique()<2: continue
        b=metrics(s.label,s.probability_uncalibrated_h0); a=metrics(s.label,s.probability_uncalibrated_selected)
        rows.append({"replicate":rep,"delta_ap":a["ap"]-b["ap"],"delta_mcc":a["mcc"]-b["mcc"],"delta_auroc":a["auroc"]-b["auroc"]})
    d=pd.DataFrame(rows); summary=[]
    for metric in ("delta_ap","delta_mcc","delta_auroc"):
        v=d[metric].to_numpy(float); summary.append({"metric":metric,"replicates":len(v),"mean":float(v.mean()),"ci_2_5":float(np.quantile(v,.025)),"median":float(np.quantile(v,.5)),"ci_97_5":float(np.quantile(v,.975)),"fraction_gt_zero":float((v>0).mean())})
    return pd.DataFrame(summary)


def args() -> argparse.Namespace:
    p=argparse.ArgumentParser(); p.add_argument("--root",type=Path,required=True); p.add_argument("--output",type=Path,required=True)
    p.add_argument("--protocol",type=Path,required=True); p.add_argument("--device",default="cuda:0"); p.add_argument("--threads",type=int,default=16)
    return p.parse_args()


def main() -> int:
    a=args(); out=a.output.expanduser().resolve(); out.mkdir(parents=True,exist_ok=True)
    lock=(out/".V2_05A.lock").open("a+",encoding="utf-8")
    try: fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError: print("Another V2-05A process holds the lock",file=sys.stderr); return 3
    lock.seek(0);lock.truncate();lock.write(f"pid={os.getpid()} started={now()}\n");lock.flush()
    log=[]; started=time.time(); atomic_json(out/"STATUS.json",{"module":"V2-05A","status":"STARTED","blind_labels_read":False,"lora":False})
    try:
        for role,path in (("root",a.root),("output",out),("protocol",a.protocol)):
            if any(x in str(path.resolve()).lower() for x in FORBIDDEN): raise RuntimeError(f"Forbidden {role} path: {path}")
        pp=guarded(a.protocol,"protocol",log); protocol=json.loads(pp.read_text())
        if protocol.get("protocol_status")!="LOCKED_BEFORE_EXECUTION" or protocol.get("module")!="V2-05A": raise RuntimeError("Protocol gate failed")
        atomic_json(out/"V2_05A_PROTOCOL_LOCKED.json",protocol); torch.set_num_threads(a.threads)
        if a.device.startswith("cuda") and not torch.cuda.is_available(): raise RuntimeError("CUDA requested but unavailable")
        device=torch.device(a.device); atomic_json(out/"STATUS.json",{"module":"V2-05A","status":"PREFLIGHT_BINDING","blind_labels_read":False,"lora":False})
        x,master,strict,clean,flags,ref,binding=load_inputs(a,protocol,log)
        atomic_json(out/"V2_05A_BINDING_AUDIT.json",binding); atomic_json(out/"INPUT_SHA256.json",{"protocol":{"path":str(pp),"sha256":sha256(pp)},**binding["input_hashes"]})
        atomic_json(out/"RUNTIME.json",{"python":sys.version,"numpy":np.__version__,"pandas":pd.__version__,"sklearn":sklearn.__version__,"torch":torch.__version__,"device":str(device),"cuda_device_name":torch.cuda.get_device_name(device) if device.type=="cuda" else None,"threads":a.threads})
        atomic_json(out/"STATUS.json",{"module":"V2-05A","status":"TRAINING","blind_labels_read":False,"lora":False})
        frames=[]
        for arm in ARMS:
            for fold in protocol["folds_expected"]: frames.append(run_fold(arm,int(fold),x,master,flags,protocol,device,out))
        oof=pd.concat(frames,ignore_index=True).sort_values(["arm","clean_id"],kind="mergesort").reset_index(drop=True)
        if len(oof)!=len(master)*4 or oof.duplicated(["arm","clean_id"]).any(): raise RuntimeError("OOF assembly failed")
        oof.to_csv(out/"all_candidate_oof_predictions.tsv",sep="\t",index=False)
        overall,folds=metric_tables(oof); strictm=subset_metrics(oof,set(master.loc[strict,"clean_id"])); cleanm=subset_metrics(oof,set(master.loc[clean,"clean_id"]))
        overall.to_csv(out/"candidate_overall_metrics.tsv",sep="\t",index=False); folds.to_csv(out/"candidate_fold_metrics.tsv",sep="\t",index=False)
        strictm.to_csv(out/"strict25_metrics.tsv",sep="\t",index=False); cleanm.to_csv(out/"clean_core_metrics.tsv",sep="\t",index=False)
        repro=h0_reproduction(oof,ref,protocol); atomic_json(out/"H0_REPRODUCTION_AUDIT.json",repro)
        if not repro["passed"]: raise RuntimeError(f"H0 reproduction gate failed: {repro}")
        gate=gates(overall,folds,strictm,cleanm); gate.to_csv(out/"candidate_gain_gates.tsv",sep="\t",index=False)
        passed=gate[gate.passes_all_hard_sample_gates.astype(bool)].arm.tolist()
        if passed:
            rank={"H1":1,"H2":2,"H3":3}; candidates=overall[overall.arm.isin(passed)].copy(); candidates["simplicity"]=candidates.arm.map(rank)
            chosen=str(candidates.sort_values(["ap","mcc","auroc","simplicity"],ascending=[False,False,False,True],kind="mergesort").iloc[0].arm)
            decision,status="KEEP_HARD_SAMPLE_STRATEGY","PASS_HARD_SAMPLE_STABLE_GAIN"
        else: chosen="H0";decision,status="STOP_HARD_SAMPLE_NO_STABLE_GAIN","PASS_SCREEN_COMPLETE_NO_STABLE_GAIN"
        atomic_json(out/"selected_hard_sample_strategy.json",{"status":status,"decision":decision,"selected_arm":chosen,"eligible_arms":passed,
            "fallback":"KEEP_K2_UNWEIGHTED" if not passed else None,"selection_priority":protocol["selection_priority"],"canonical_mean_only":True,"k2_fixed":True,
            "blind_labels_read":False,"test200_labels_read":False,"external_challenge_labels_read":False,"lora":False,"next_gate":"V2-06 only after V2-05A review"})
        bootstrap(oof[oof.arm==chosen],oof[oof.arm=="H0"],int(protocol["bootstrap"]["replicates"]),int(protocol["seed"])+950000).to_csv(out/"selected_vs_h0_group_bootstrap.tsv",sep="\t",index=False)
        atomic_json(out/"DATA_ACCESS_AUDIT.json",{"status":"PASS_ALLOWED_INPUTS_ONLY","accessed_inputs":log,"forbidden_tokens":list(FORBIDDEN),
            "blind_labels_read":False,"test200_labels_read":False,"external_challenge_labels_read":False,"candidate_negative_pool_used":False,
            "esmc_loaded_or_updated":False,"lora_or_peft":False,"calibration":False,"threshold_optimized":False})
        done={"module":"V2-05A","status":status,"decision":decision,"selected_arm":chosen,"started_at_utc":datetime.fromtimestamp(started,timezone.utc).isoformat(),
            "finished_at_utc":now(),"elapsed_seconds":round(time.time()-started,3),"records":len(master),"arms":list(ARMS),"arm_folds_complete":20,
            "strict40_shared_groups":0,"strict25_records":int(strict.sum()),"clean_core_records":int(clean.sum()),"canonical_mean_only":True,"k2_fixed":True,
            "labels_changed":0,"calibration_applied":False,"threshold_optimized":False,"candidate_negative_pool_used":False,"lora":False,
            "blind_labels_read":False,"test200_labels_read":False,"external_challenge_labels_read":False,"next_gate":"V2-06 only after review"}
        atomic_json(out/"V2_05A_DONE.json",done);atomic_json(out/"STATUS.json",done);return 0
    except Exception as exc:
        failure={"module":"V2-05A","status":"FAILED","updated_at_utc":now(),"error_type":type(exc).__name__,"error":str(exc),"blind_labels_read":False,"lora":False}
        atomic_json(out/"STATUS.json",failure);atomic_json(out/"DATA_ACCESS_AUDIT.json",{"status":"FAILED_DURING_ALLOWED_INPUT_PROCESSING","accessed_inputs":log,
            "blind_labels_read":False,"test200_labels_read":False,"external_challenge_labels_read":False,"candidate_negative_pool_used":False,"lora_or_peft":False})
        traceback.print_exc();return 1


if __name__ == "__main__": raise SystemExit(main())
