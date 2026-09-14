#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import gc
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    roc_auc_score,
)
from sklearn.preprocessing import MinMaxScaler
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset

from KANLayer import KANLayer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fair DeepSaltPro training with fixed group-aware folds, fold-local "
            "MinMaxScaler/PCA, OOF threshold selection, and locked test evaluation."
        )
    )
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--fold-assignments", required=True, type=Path)
    parser.add_argument("--ankh-dir", required=True, type=Path)
    parser.add_argument("--esm2-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--pca-components", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=4e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--min-mcc-improvement", type=float, default=0.005)
    parser.add_argument(
        "--fold-limit",
        type=int,
        default=None,
        help="Smoke test only: train the first N folds.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_feature_dir(
    directory: Path,
    expected_dim: int,
    name: str,
) -> tuple[pd.DataFrame, np.ndarray, dict[str, Any]]:
    rows_path = directory / "rows.tsv"
    embeddings_path = directory / "embeddings.npy"
    summary_path = directory / "summary.json"

    if not rows_path.is_file():
        raise FileNotFoundError(f"{name}: missing {rows_path}")
    if not embeddings_path.is_file():
        raise FileNotFoundError(f"{name}: missing {embeddings_path}")

    rows = pd.read_csv(
        rows_path,
        sep="\t",
        dtype={"clean_id": str, "cluster_id": str},
    )
    embeddings = np.load(
        embeddings_path,
        mmap_mode="r",
        allow_pickle=False,
    )

    if embeddings.ndim != 2:
        raise ValueError(f"{name}: expected 2D embeddings, got {embeddings.shape}")
    if embeddings.shape != (len(rows), expected_dim):
        raise ValueError(
            f"{name}: embeddings shape {embeddings.shape}, "
            f"rows={len(rows)}, expected_dim={expected_dim}"
        )
    if not rows["clean_id"].is_unique:
        raise ValueError(f"{name}: duplicate clean_id")
    if not np.isfinite(embeddings).all():
        raise ValueError(f"{name}: NaN or Inf detected")

    summary: dict[str, Any] = {}
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("status") != "PASS":
            raise ValueError(f"{name}: source summary is not PASS")

    return rows, embeddings, summary


def align_embeddings(
    manifest: pd.DataFrame,
    rows: pd.DataFrame,
    embeddings: np.ndarray,
    name: str,
) -> np.ndarray:
    row_index = {clean_id: i for i, clean_id in enumerate(rows["clean_id"])}
    missing = [x for x in manifest["clean_id"] if x not in row_index]
    if missing:
        raise ValueError(f"{name}: {len(missing)} manifest IDs are missing")

    order = np.asarray(
        [row_index[x] for x in manifest["clean_id"]],
        dtype=np.int64,
    )
    aligned = np.asarray(embeddings[order], dtype=np.float32)
    if not np.isfinite(aligned).all():
        raise ValueError(f"{name}: aligned matrix contains NaN or Inf")
    return aligned


class FocalLossWithLogits(nn.Module):
    def __init__(self, alpha: float = 0.5, gamma: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits,
            targets,
            reduction="none",
        )
        probability = torch.sigmoid(logits)
        pt = torch.where(targets > 0.5, probability, 1.0 - probability)
        alpha_t = torch.where(
            targets > 0.5,
            torch.full_like(targets, self.alpha),
            torch.full_like(targets, 1.0 - self.alpha),
        )
        return (
            alpha_t
            * (1.0 - pt).pow(self.gamma)
            * bce
        ).mean()


class DeepSaltProNet(nn.Module):
    def __init__(self, device: torch.device):
        super().__init__()

        self.esm2_cnn = nn.Sequential(
            nn.Conv1d(512, 256, kernel_size=9, padding="same"),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 128, kernel_size=9, padding="same"),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        self.esm2_gru_projection = nn.Sequential(
            nn.Conv1d(512, 512, kernel_size=9, padding="same"),
            nn.BatchNorm1d(512),
            nn.ReLU(),
        )

        self.ankh_cnn = nn.Sequential(
            nn.Conv1d(512, 256, kernel_size=9, padding="same"),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 128, kernel_size=9, padding="same"),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        self.ankh_gru_projection = nn.Sequential(
            nn.Conv1d(512, 512, kernel_size=9, padding="same"),
            nn.BatchNorm1d(512),
            nn.ReLU(),
        )

        self.bigru = nn.GRU(
            input_size=1024,
            hidden_size=64,
            bidirectional=True,
            batch_first=True,
        )

        self.kan1 = KANLayer(
            in_dim=384,
            out_dim=16,
            num=5,
            k=3,
            grid_eps=0.5,
            grid_range=[-1, 1],
            sp_trainable=True,
            sb_trainable=True,
            noise_scale=0.5,
            device=device,
        )
        self.kan2 = KANLayer(
            in_dim=16,
            out_dim=1,
            num=3,
            k=3,
            grid_eps=0.5,
            grid_range=[-1, 1],
            sp_trainable=True,
            sb_trainable=True,
            noise_scale=0.2,
            device=device,
        )
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Conv1d, nn.Linear)):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        esm2_input: torch.Tensor,
        ankh_input: torch.Tensor,
    ) -> torch.Tensor:
        esm2_cnn = self.esm2_cnn(esm2_input).permute(0, 2, 1)
        ankh_cnn = self.ankh_cnn(ankh_input).permute(0, 2, 1)

        esm2_gru = self.esm2_gru_projection(esm2_input).permute(0, 2, 1)
        ankh_gru = self.ankh_gru_projection(ankh_input).permute(0, 2, 1)

        gru_input = torch.cat([esm2_gru, ankh_gru], dim=-1)
        gru_output, _ = self.bigru(gru_input)

        combined = torch.cat(
            [esm2_cnn, gru_output, ankh_cnn],
            dim=-1,
        ).squeeze(1)

        x, *_ = self.kan1(combined)
        x, *_ = self.kan2(x)
        return x.squeeze(-1)


def predict_probabilities(
    model: nn.Module,
    esm2_matrix: np.ndarray,
    ankh_matrix: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval()

    dataset = TensorDataset(
        torch.from_numpy(
            np.ascontiguousarray(esm2_matrix, dtype=np.float32)
        ),
        torch.from_numpy(
            np.ascontiguousarray(ankh_matrix, dtype=np.float32)
        ),
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    probabilities = []
    with torch.inference_mode():
        for esm2_batch, ankh_batch in loader:
            logits = model(
                esm2_batch.unsqueeze(2).to(device),
                ankh_batch.unsqueeze(2).to(device),
            )
            probabilities.append(
                torch.sigmoid(logits).cpu().numpy()
            )

    return np.concatenate(probabilities).astype(np.float64)


def expected_calibration_error(
    y_true: np.ndarray,
    probability: np.ndarray,
    n_bins: int = 10,
) -> float:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = len(y_true)
    ece = 0.0

    for bin_index in range(n_bins):
        lower = edges[bin_index]
        upper = edges[bin_index + 1]

        if bin_index == n_bins - 1:
            mask = (probability >= lower) & (probability <= upper)
        else:
            mask = (probability >= lower) & (probability < upper)

        if not np.any(mask):
            continue

        confidence = float(np.mean(probability[mask]))
        accuracy = float(np.mean(y_true[mask]))
        ece += (int(mask.sum()) / total) * abs(accuracy - confidence)

    return float(ece)


def metric_dict(
    y_true: np.ndarray,
    probability: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    prediction = (probability >= threshold).astype(np.int64)
    matrix = confusion_matrix(
        y_true,
        prediction,
        labels=[0, 1],
    )
    tn, fp, fn, tp = matrix.ravel()

    specificity = tn / (tn + fp) if (tn + fp) else float("nan")
    sensitivity = tp / (tp + fn) if (tp + fn) else float("nan")

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, prediction)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true, prediction)
        ),
        "specificity": float(specificity),
        "sensitivity": float(sensitivity),
        "precision": float(
            precision_score(y_true, prediction, zero_division=0)
        ),
        "f1": float(
            f1_score(y_true, prediction, zero_division=0)
        ),
        "mcc": float(
            matthews_corrcoef(y_true, prediction)
        ),
        "roc_auc": float(
            roc_auc_score(y_true, probability)
        ),
        "pr_auc": float(
            average_precision_score(y_true, probability)
        ),
        "brier": float(
            brier_score_loss(y_true, probability)
        ),
        "ece_10_bins": expected_calibration_error(
            y_true,
            probability,
            n_bins=10,
        ),
        "confusion_matrix": matrix.tolist(),
    }


def mcc_from_counts(
    tp: int,
    tn: int,
    fp: int,
    fn: int,
) -> float:
    numerator = (tp * tn) - (fp * fn)
    denominator = math.sqrt(
        (tp + fp)
        * (tp + fn)
        * (tn + fp)
        * (tn + fn)
    )
    if denominator == 0:
        return 0.0
    return numerator / denominator


def choose_mcc_threshold(
    y_true: np.ndarray,
    probability: np.ndarray,
) -> float:
    y_true = np.asarray(y_true, dtype=np.int64)
    probability = np.asarray(probability, dtype=np.float64)

    best_threshold = 0.5
    best_mcc = float(
        matthews_corrcoef(
            y_true,
            (probability >= 0.5).astype(np.int64),
        )
    )

    order = np.argsort(-probability, kind="mergesort")
    sorted_probability = probability[order]
    sorted_label = y_true[order]

    total_positive = int(sorted_label.sum())
    total_negative = int(len(sorted_label) - total_positive)

    tp = 0
    fp = 0
    index = 0

    while index < len(sorted_label):
        threshold = float(sorted_probability[index])
        end = index

        while (
            end < len(sorted_label)
            and sorted_probability[end] == threshold
        ):
            if sorted_label[end] == 1:
                tp += 1
            else:
                fp += 1
            end += 1

        fn = total_positive - tp
        tn = total_negative - fp
        mcc = mcc_from_counts(tp, tn, fp, fn)

        if (
            mcc > best_mcc
            or (
                math.isclose(mcc, best_mcc)
                and abs(threshold - 0.5)
                < abs(best_threshold - 0.5)
            )
        ):
            best_mcc = mcc
            best_threshold = threshold

        index = end

    return float(best_threshold)


def fit_platt_calibrator(
    probability: np.ndarray,
    labels: np.ndarray,
    seed: int,
) -> LogisticRegression:
    epsilon = 1e-6
    clipped = np.clip(probability, epsilon, 1.0 - epsilon)
    logits = np.log(clipped / (1.0 - clipped)).reshape(-1, 1)

    calibrator = LogisticRegression(
        solver="lbfgs",
        random_state=seed,
    )
    calibrator.fit(logits, labels)
    return calibrator


def apply_platt_calibrator(
    calibrator: LogisticRegression,
    probability: np.ndarray,
) -> np.ndarray:
    epsilon = 1e-6
    clipped = np.clip(probability, epsilon, 1.0 - epsilon)
    logits = np.log(clipped / (1.0 - clipped)).reshape(-1, 1)
    return calibrator.predict_proba(logits)[:, 1]


def train_one_fold(
    fold_id: int,
    fit_esm2: np.ndarray,
    fit_ankh: np.ndarray,
    fit_labels: np.ndarray,
    validation_esm2: np.ndarray,
    validation_ankh: np.ndarray,
    validation_labels: np.ndarray,
    device: torch.device,
    args: argparse.Namespace,
) -> tuple[nn.Module, pd.DataFrame, float]:
    set_seed(args.seed + fold_id)

    model = DeepSaltProNet(device).to(device)
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    criterion = FocalLossWithLogits(
        alpha=0.5,
        gamma=1.0,
    )
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.learning_rate / 100.0,
    )

    dataset = TensorDataset(
        torch.from_numpy(
            np.ascontiguousarray(fit_esm2, dtype=np.float32)
        ),
        torch.from_numpy(
            np.ascontiguousarray(fit_ankh, dtype=np.float32)
        ),
        torch.from_numpy(
            np.ascontiguousarray(fit_labels, dtype=np.float32)
        ),
    )
    generator = torch.Generator().manual_seed(args.seed + fold_id)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
    )

    best_state = None
    best_mcc = -np.inf
    epochs_without_improvement = 0
    history_rows = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_records = 0
        total_correct = 0

        for esm2_batch, ankh_batch, label_batch in loader:
            esm2_batch = esm2_batch.unsqueeze(2).to(device)
            ankh_batch = ankh_batch.unsqueeze(2).to(device)
            label_batch = label_batch.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(esm2_batch, ankh_batch)
            loss = criterion(logits, label_batch)
            loss.backward()
            optimizer.step()

            batch_size = len(label_batch)
            total_loss += float(loss.item()) * batch_size
            total_records += batch_size
            total_correct += int(
                (
                    (logits >= 0).to(label_batch.dtype)
                    == label_batch
                ).sum().item()
            )

        validation_probability = predict_probabilities(
            model,
            validation_esm2,
            validation_ankh,
            device,
            args.eval_batch_size,
        )
        validation_prediction = (
            validation_probability >= 0.5
        ).astype(np.int64)
        validation_mcc = float(
            matthews_corrcoef(
                validation_labels,
                validation_prediction,
            )
        )

        history_rows.append(
            {
                "fold": fold_id,
                "epoch": epoch,
                "train_loss": total_loss / max(total_records, 1),
                "train_accuracy": total_correct / max(total_records, 1),
                "validation_mcc_at_0_5": validation_mcc,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )

        if validation_mcc > (
            best_mcc + args.min_mcc_improvement
        ):
            best_mcc = validation_mcc
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epoch == 1 or epoch % 5 == 0:
            print(
                f"Fold {fold_id} | epoch {epoch}/{args.epochs} | "
                f"loss={history_rows[-1]['train_loss']:.5f} | "
                f"train_acc={history_rows[-1]['train_accuracy']:.4f} | "
                f"val_mcc={validation_mcc:.4f}",
                flush=True,
            )

        scheduler.step()

        if epochs_without_improvement >= args.patience:
            print(
                f"Fold {fold_id}: early stopping at epoch {epoch}",
                flush=True,
            )
            break

    if best_state is None:
        raise RuntimeError(
            f"Fold {fold_id}: no best model state was captured"
        )

    model.load_state_dict(best_state)
    return model, pd.DataFrame(history_rows), float(best_mcc)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    folds_dir = output_dir / "folds"
    folds_dir.mkdir(exist_ok=True)

    device = torch.device(
        args.device
        if torch.cuda.is_available()
        else "cpu"
    )
    print(f"Using device: {device}", flush=True)

    manifest = pd.read_csv(
        args.manifest.expanduser().resolve(),
        sep="\t",
        dtype={"clean_id": str, "cluster_id": str},
    )

    required_columns = [
        "clean_id",
        "label",
        "split",
        "cluster_id",
    ]
    missing = [
        c for c in required_columns
        if c not in manifest.columns
    ]
    if missing:
        raise ValueError(
            f"Manifest missing columns: {missing}"
        )
    if not manifest["clean_id"].is_unique:
        raise ValueError(
            "Manifest clean_id is not unique"
        )

    ankh_rows, ankh_memmap, ankh_summary = load_feature_dir(
        args.ankh_dir.expanduser().resolve(),
        expected_dim=1536,
        name="Ankh",
    )
    esm2_rows, esm2_memmap, esm2_summary = load_feature_dir(
        args.esm2_dir.expanduser().resolve(),
        expected_dim=2560,
        name="ESM-2",
    )

    ankh_all = align_embeddings(
        manifest,
        ankh_rows,
        ankh_memmap,
        "Ankh",
    )
    esm2_all = align_embeddings(
        manifest,
        esm2_rows,
        esm2_memmap,
        "ESM-2",
    )

    train_mask = manifest["split"].eq("train").to_numpy()
    test_mask = manifest["split"].eq("test").to_numpy()

    train_metadata = manifest.loc[
        train_mask,
        required_columns,
    ].reset_index(drop=True)
    test_metadata = manifest.loc[
        test_mask,
        required_columns,
    ].reset_index(drop=True)

    train_ankh_raw = ankh_all[train_mask]
    test_ankh_raw = ankh_all[test_mask]
    train_esm2_raw = esm2_all[train_mask]
    test_esm2_raw = esm2_all[test_mask]

    train_labels = train_metadata["label"].to_numpy(
        dtype=np.int64
    )
    test_labels = test_metadata["label"].to_numpy(
        dtype=np.int64
    )

    assignments = pd.read_csv(
        args.fold_assignments.expanduser().resolve(),
        sep="\t",
        dtype={"clean_id": str, "cluster_id": str},
    )
    if "fold" not in assignments.columns:
        raise ValueError(
            "Fold assignment file does not contain a fold column"
        )

    assignment_subset = assignments[
        ["clean_id", "fold"]
    ].copy()
    merged = train_metadata.merge(
        assignment_subset,
        on="clean_id",
        how="left",
        validate="one_to_one",
    )

    if merged["fold"].isna().any():
        raise ValueError(
            "Some training IDs have no fold assignment"
        )

    fold_ids = sorted(
        int(x) for x in merged["fold"].unique()
    )
    total_fold_count = len(fold_ids)

    if args.fold_limit is not None:
        fold_ids = fold_ids[: args.fold_limit]

    complete_five_fold_run = (
        len(fold_ids) == total_fold_count
    )

    oof_raw = np.full(
        len(train_metadata),
        np.nan,
        dtype=np.float64,
    )
    test_fold_probabilities = []
    fold_metric_rows = []
    started = time.time()

    config = {
        "manifest": str(args.manifest.expanduser().resolve()),
        "fold_assignments": str(
            args.fold_assignments.expanduser().resolve()
        ),
        "ankh_dir": str(args.ankh_dir.expanduser().resolve()),
        "esm2_dir": str(args.esm2_dir.expanduser().resolve()),
        "device": str(device),
        "seed": args.seed,
        "pca_components": args.pca_components,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "patience": args.patience,
        "min_mcc_improvement": args.min_mcc_improvement,
        "fold_ids": fold_ids,
        "ankh_source_summary": ankh_summary,
        "esm2_source_summary": esm2_summary,
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
    }
    (output_dir / "run_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    for fold_id in fold_ids:
        print(
            f"\n================ Fold {fold_id} ================",
            flush=True,
        )

        fold_dir = folds_dir / f"fold_{fold_id}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        validation_index = np.flatnonzero(
            merged["fold"].to_numpy(dtype=np.int64)
            == fold_id
        )
        fit_index = np.flatnonzero(
            merged["fold"].to_numpy(dtype=np.int64)
            != fold_id
        )

        fit_clusters = set(
            train_metadata.iloc[fit_index]["cluster_id"]
        )
        validation_clusters = set(
            train_metadata.iloc[validation_index]["cluster_id"]
        )
        shared_clusters = fit_clusters.intersection(
            validation_clusters
        )
        if shared_clusters:
            raise RuntimeError(
                f"Fold {fold_id}: cluster leakage detected"
            )

        print(
            f"Fit records: {len(fit_index)} | "
            f"validation records: {len(validation_index)}",
            flush=True,
        )

        scaler_esm2 = MinMaxScaler()
        scaler_ankh = MinMaxScaler()

        pca_esm2 = PCA(
            n_components=args.pca_components,
            svd_solver="randomized",
            random_state=args.seed + fold_id,
        )
        pca_ankh = PCA(
            n_components=args.pca_components,
            svd_solver="randomized",
            random_state=args.seed + fold_id,
        )

        fit_esm2_scaled = scaler_esm2.fit_transform(
            train_esm2_raw[fit_index]
        )
        validation_esm2_scaled = scaler_esm2.transform(
            train_esm2_raw[validation_index]
        )
        test_esm2_scaled = scaler_esm2.transform(
            test_esm2_raw
        )

        fit_ankh_scaled = scaler_ankh.fit_transform(
            train_ankh_raw[fit_index]
        )
        validation_ankh_scaled = scaler_ankh.transform(
            train_ankh_raw[validation_index]
        )
        test_ankh_scaled = scaler_ankh.transform(
            test_ankh_raw
        )

        fit_esm2 = pca_esm2.fit_transform(
            fit_esm2_scaled
        ).astype(np.float32)
        validation_esm2 = pca_esm2.transform(
            validation_esm2_scaled
        ).astype(np.float32)
        test_esm2 = pca_esm2.transform(
            test_esm2_scaled
        ).astype(np.float32)

        fit_ankh = pca_ankh.fit_transform(
            fit_ankh_scaled
        ).astype(np.float32)
        validation_ankh = pca_ankh.transform(
            validation_ankh_scaled
        ).astype(np.float32)
        test_ankh = pca_ankh.transform(
            test_ankh_scaled
        ).astype(np.float32)

        preprocessing = {
            "scaler_esm2": scaler_esm2,
            "scaler_ankh": scaler_ankh,
            "pca_esm2": pca_esm2,
            "pca_ankh": pca_ankh,
        }
        joblib.dump(
            preprocessing,
            fold_dir / "preprocessing.joblib",
        )

        preprocessing_summary = {
            "fold": fold_id,
            "fit_records": int(len(fit_index)),
            "validation_records": int(len(validation_index)),
            "shared_clusters": 0,
            "esm2_variance_retained": float(
                np.sum(pca_esm2.explained_variance_ratio_)
            ),
            "ankh_variance_retained": float(
                np.sum(pca_ankh.explained_variance_ratio_)
            ),
        }
        (fold_dir / "preprocessing_summary.json").write_text(
            json.dumps(
                preprocessing_summary,
                indent=2,
                ensure_ascii=False,
            ) + "\n",
            encoding="utf-8",
        )

        model, history, best_validation_mcc = train_one_fold(
            fold_id=fold_id,
            fit_esm2=fit_esm2,
            fit_ankh=fit_ankh,
            fit_labels=train_labels[fit_index],
            validation_esm2=validation_esm2,
            validation_ankh=validation_ankh,
            validation_labels=train_labels[validation_index],
            device=device,
            args=args,
        )

        history.to_csv(
            fold_dir / "training_history.tsv",
            sep="\t",
            index=False,
        )
        torch.save(
            {
                "state_dict": model.state_dict(),
                "fold": fold_id,
                "seed": args.seed,
                "pca_components": args.pca_components,
            },
            fold_dir / "model.pt",
        )

        validation_probability = predict_probabilities(
            model,
            validation_esm2,
            validation_ankh,
            device,
            args.eval_batch_size,
        )
        test_probability = predict_probabilities(
            model,
            test_esm2,
            test_ankh,
            device,
            args.eval_batch_size,
        )

        oof_raw[validation_index] = validation_probability
        test_fold_probabilities.append(test_probability)

        validation_output = train_metadata.iloc[
            validation_index
        ].copy()
        validation_output["fold"] = fold_id
        validation_output["true_label"] = train_labels[
            validation_index
        ]
        validation_output[
            "probability_raw"
        ] = validation_probability
        validation_output.to_csv(
            fold_dir / "validation_predictions.tsv",
            sep="\t",
            index=False,
        )

        fold_metrics = metric_dict(
            train_labels[validation_index],
            validation_probability,
            threshold=0.5,
        )
        fold_metrics.update(
            {
                "fold": fold_id,
                "fit_records": int(len(fit_index)),
                "validation_records": int(len(validation_index)),
                "best_validation_mcc_during_training": (
                    best_validation_mcc
                ),
                "esm2_variance_retained": (
                    preprocessing_summary[
                        "esm2_variance_retained"
                    ]
                ),
                "ankh_variance_retained": (
                    preprocessing_summary[
                        "ankh_variance_retained"
                    ]
                ),
            }
        )
        fold_metric_rows.append(fold_metrics)

        (fold_dir / "metrics.json").write_text(
            json.dumps(
                fold_metrics,
                indent=2,
                ensure_ascii=False,
            ) + "\n",
            encoding="utf-8",
        )
        print(
            json.dumps(
                fold_metrics,
                indent=2,
                ensure_ascii=False,
            ),
            flush=True,
        )

        del (
            model,
            fit_esm2_scaled,
            validation_esm2_scaled,
            test_esm2_scaled,
            fit_ankh_scaled,
            validation_ankh_scaled,
            test_ankh_scaled,
            fit_esm2,
            validation_esm2,
            test_esm2,
            fit_ankh,
            validation_ankh,
            test_ankh,
        )
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    pd.DataFrame(fold_metric_rows).to_json(
        output_dir / "fold_metrics.json",
        orient="records",
        indent=2,
    )

    if not complete_five_fold_run:
        smoke_summary = {
            "status": "SMOKE_PASS",
            "trained_folds": fold_ids,
            "expected_total_folds": total_fold_count,
            "elapsed_seconds": round(
                time.time() - started,
                3,
            ),
        }
        (output_dir / "run_summary.json").write_text(
            json.dumps(
                smoke_summary,
                indent=2,
                ensure_ascii=False,
            ) + "\n",
            encoding="utf-8",
        )
        print(
            json.dumps(
                smoke_summary,
                indent=2,
                ensure_ascii=False,
            ),
            flush=True,
        )
        return

    if np.isnan(oof_raw).any():
        raise RuntimeError(
            "OOF probabilities are incomplete"
        )

    test_raw = np.mean(
        np.stack(test_fold_probabilities, axis=0),
        axis=0,
    )

    raw_oof_threshold = choose_mcc_threshold(
        train_labels,
        oof_raw,
    )

    calibrator = fit_platt_calibrator(
        oof_raw,
        train_labels,
        seed=args.seed,
    )
    oof_calibrated = apply_platt_calibrator(
        calibrator,
        oof_raw,
    )
    test_calibrated = apply_platt_calibrator(
        calibrator,
        test_raw,
    )
    calibrated_oof_threshold = choose_mcc_threshold(
        train_labels,
        oof_calibrated,
    )

    joblib.dump(
        calibrator,
        output_dir / "platt_calibrator.joblib",
    )

    oof_output = train_metadata.copy()
    oof_output["true_label"] = train_labels
    oof_output["fold"] = merged["fold"].astype(int)
    oof_output["probability_raw"] = oof_raw
    oof_output["probability_calibrated"] = oof_calibrated
    oof_output["prediction_raw_oof_threshold"] = (
        oof_raw >= raw_oof_threshold
    ).astype(np.int64)
    oof_output[
        "prediction_calibrated_oof_threshold"
    ] = (
        oof_calibrated >= calibrated_oof_threshold
    ).astype(np.int64)
    oof_output.to_csv(
        output_dir / "oof_predictions.tsv",
        sep="\t",
        index=False,
    )

    test_output = test_metadata.copy()
    test_output["true_label"] = test_labels
    test_output["probability_raw"] = test_raw
    test_output["probability_calibrated"] = test_calibrated
    test_output["prediction_raw_0_5"] = (
        test_raw >= 0.5
    ).astype(np.int64)
    test_output["prediction_raw_oof_threshold"] = (
        test_raw >= raw_oof_threshold
    ).astype(np.int64)
    test_output[
        "prediction_calibrated_oof_threshold"
    ] = (
        test_calibrated >= calibrated_oof_threshold
    ).astype(np.int64)
    test_output.to_csv(
        output_dir / "test_predictions.tsv",
        sep="\t",
        index=False,
    )

    metrics = {
        "oof_raw_at_0_5": metric_dict(
            train_labels,
            oof_raw,
            threshold=0.5,
        ),
        "oof_raw_at_oof_threshold": metric_dict(
            train_labels,
            oof_raw,
            threshold=raw_oof_threshold,
        ),
        "oof_calibrated_at_oof_threshold": metric_dict(
            train_labels,
            oof_calibrated,
            threshold=calibrated_oof_threshold,
        ),
        "test_raw_at_0_5": metric_dict(
            test_labels,
            test_raw,
            threshold=0.5,
        ),
        "test_raw_at_oof_threshold": metric_dict(
            test_labels,
            test_raw,
            threshold=raw_oof_threshold,
        ),
        "test_calibrated_at_oof_threshold": metric_dict(
            test_labels,
            test_calibrated,
            threshold=calibrated_oof_threshold,
        ),
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(
            metrics,
            indent=2,
            ensure_ascii=False,
        ) + "\n",
        encoding="utf-8",
    )

    run_summary = {
        "status": "PASS",
        "device": str(device),
        "seed": args.seed,
        "folds": fold_ids,
        "train_records": int(len(train_metadata)),
        "test_records": int(len(test_metadata)),
        "raw_oof_mcc_threshold": raw_oof_threshold,
        "calibrated_oof_mcc_threshold": (
            calibrated_oof_threshold
        ),
        "elapsed_seconds": round(
            time.time() - started,
            3,
        ),
        "outputs": {
            "oof_predictions": str(
                output_dir / "oof_predictions.tsv"
            ),
            "test_predictions": str(
                output_dir / "test_predictions.tsv"
            ),
            "metrics": str(
                output_dir / "metrics.json"
            ),
        },
    }
    (output_dir / "run_summary.json").write_text(
        json.dumps(
            run_summary,
            indent=2,
            ensure_ascii=False,
        ) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            run_summary,
            indent=2,
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
