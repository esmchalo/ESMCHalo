#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import MinMaxScaler
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Nested group-OOF DeepSaltPro teacher generation for ESMCHalo-v2."
    )
    p.add_argument("--root", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--outer-folds", default="0", help="Comma-separated folds, or all")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=20260902)
    p.add_argument("--inner-splits", type=int, default=4)
    p.add_argument("--pca-components", type=int, default=512)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--eval-batch-size", type=int, default=1024)
    p.add_argument("--learning-rate", type=float, default=4e-5)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--min-mcc-improvement", type=float, default=0.005)
    return p.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_legacy_module(root: Path):
    code_dir = root / "external/DeepSaltPro/Code"
    runner_path = root / "scripts/deepsaltpro_fair_stage4b/train_deepsaltpro_fair.py"
    require(code_dir.is_dir(), f"Missing DeepSaltPro code directory: {code_dir}")
    require(runner_path.is_file(), f"Missing audited DeepSaltPro runner: {runner_path}")
    sys.path.insert(0, str(code_dir))
    spec = importlib.util.spec_from_file_location("deepsaltpro_fair_stage4b", runner_path)
    require(spec is not None and spec.loader is not None, "Could not load DeepSaltPro runner")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, runner_path


def load_feature_dir(directory: Path, expected_dim: int, name: str):
    rows_path = directory / "rows.tsv"
    embeddings_path = directory / "embeddings.npy"
    summary_path = directory / "summary.json"
    require(rows_path.is_file(), f"{name}: missing rows.tsv")
    require(embeddings_path.is_file(), f"{name}: missing embeddings.npy")
    rows = pd.read_csv(rows_path, sep="\t", dtype={"clean_id": str})
    require("clean_id" in rows.columns and rows["clean_id"].is_unique, f"{name}: invalid clean_id")
    embeddings = np.load(embeddings_path, mmap_mode="r", allow_pickle=False)
    require(embeddings.shape == (len(rows), expected_dim), f"{name}: unexpected shape {embeddings.shape}")
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else {}
    require(summary.get("status") == "PASS", f"{name}: source summary is not PASS")
    return rows, embeddings, {
        "directory": str(directory),
        "rows_sha256": sha256_file(rows_path),
        "embeddings_sha256": sha256_file(embeddings_path),
        "shape": list(embeddings.shape),
        "source_status": summary.get("status"),
    }


def align_features(development: pd.DataFrame, rows: pd.DataFrame, matrix: np.ndarray, name: str) -> np.ndarray:
    lookup = {clean_id: i for i, clean_id in enumerate(rows["clean_id"].astype(str))}
    missing = [x for x in development["clean_id"].astype(str) if x not in lookup]
    require(not missing, f"{name}: {len(missing)} development IDs missing")
    order = np.asarray([lookup[x] for x in development["clean_id"].astype(str)], dtype=np.int64)
    aligned = np.asarray(matrix[order], dtype=np.float32)
    require(np.isfinite(aligned).all(), f"{name}: non-finite aligned features")
    return aligned


def fit_transform_pair(
    esm2_raw: np.ndarray,
    ankh_raw: np.ndarray,
    fit_index: np.ndarray,
    eval_index: np.ndarray,
    components: int,
    seed: int,
):
    esm2_scaler = MinMaxScaler()
    ankh_scaler = MinMaxScaler()
    esm2_pca = PCA(n_components=components, svd_solver="randomized", random_state=seed)
    ankh_pca = PCA(n_components=components, svd_solver="randomized", random_state=seed + 1)

    fit_esm2_scaled = esm2_scaler.fit_transform(esm2_raw[fit_index])
    eval_esm2_scaled = esm2_scaler.transform(esm2_raw[eval_index])
    fit_ankh_scaled = ankh_scaler.fit_transform(ankh_raw[fit_index])
    eval_ankh_scaled = ankh_scaler.transform(ankh_raw[eval_index])

    fit_esm2 = esm2_pca.fit_transform(fit_esm2_scaled).astype(np.float32)
    eval_esm2 = esm2_pca.transform(eval_esm2_scaled).astype(np.float32)
    fit_ankh = ankh_pca.fit_transform(fit_ankh_scaled).astype(np.float32)
    eval_ankh = ankh_pca.transform(eval_ankh_scaled).astype(np.float32)

    preprocessing = {
        "esm2_scaler": esm2_scaler,
        "ankh_scaler": ankh_scaler,
        "esm2_pca": esm2_pca,
        "ankh_pca": ankh_pca,
    }
    summary = {
        "fit_records": int(len(fit_index)),
        "evaluation_records": int(len(eval_index)),
        "esm2_variance_retained": float(np.sum(esm2_pca.explained_variance_ratio_)),
        "ankh_variance_retained": float(np.sum(ankh_pca.explained_variance_ratio_)),
    }
    return fit_esm2, eval_esm2, fit_ankh, eval_ankh, preprocessing, summary


def selected_epoch_from_history(history: pd.DataFrame, minimum_improvement: float) -> int:
    best = -np.inf
    selected = None
    for row in history.itertuples(index=False):
        value = float(row.validation_mcc_at_0_5)
        if value > best + minimum_improvement:
            best = value
            selected = int(row.epoch)
    require(selected is not None, "No inner-selected epoch")
    return selected


def train_fixed_epochs(
    legacy,
    esm2: np.ndarray,
    ankh: np.ndarray,
    labels: np.ndarray,
    device: torch.device,
    args: argparse.Namespace,
    epochs: int,
    seed_offset: int,
):
    legacy.set_seed(args.seed + seed_offset)
    model = legacy.DeepSaltProNet(device).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = legacy.FocalLossWithLogits(alpha=0.5, gamma=1.0)
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=max(args.epochs, 1),
        eta_min=args.learning_rate / 100.0,
    )
    dataset = TensorDataset(
        torch.from_numpy(np.ascontiguousarray(esm2, dtype=np.float32)),
        torch.from_numpy(np.ascontiguousarray(ankh, dtype=np.float32)),
        torch.from_numpy(np.ascontiguousarray(labels, dtype=np.float32)),
    )
    generator = torch.Generator().manual_seed(args.seed + seed_offset)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, generator=generator)
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = 0.0
        correct = 0
        count = 0
        for esm2_batch, ankh_batch, label_batch in loader:
            esm2_batch = esm2_batch.unsqueeze(2).to(device)
            ankh_batch = ankh_batch.unsqueeze(2).to(device)
            label_batch = label_batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(esm2_batch, ankh_batch)
            loss = criterion(logits, label_batch)
            loss.backward()
            optimizer.step()
            n = len(label_batch)
            loss_sum += float(loss.item()) * n
            correct += int(((logits >= 0).to(label_batch.dtype) == label_batch).sum().item())
            count += n
        scheduler.step()
        row = {
            "epoch": epoch,
            "train_loss": loss_sum / max(count, 1),
            "train_accuracy": correct / max(count, 1),
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        print(
            f"Final refit epoch {epoch}/{epochs} | loss={row['train_loss']:.5f} | "
            f"train_acc={row['train_accuracy']:.4f}",
            flush=True,
        )
    return model, pd.DataFrame(history)


def parse_outer_folds(value: str) -> list[int]:
    if value.strip().lower() == "all":
        return [0, 1, 2, 3, 4]
    result = [int(x.strip()) for x in value.split(",") if x.strip()]
    require(result and len(result) == len(set(result)), f"Invalid outer folds: {value}")
    require(set(result) <= {0, 1, 2, 3, 4}, f"Invalid outer folds: {result}")
    return result


def main() -> None:
    args = parse_args()
    root = args.root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat()
    start_time = time.time()

    require(torch.cuda.is_available(), "CUDA is unavailable; do not run the DeepSaltPro smoke test on CPU")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    legacy, runner_path = load_legacy_module(root)

    base = root / "ESMCHalo_v2_optimization/V2_01_baseline_and_teacher_oof"
    development_path = root / "ESMCHalo_v2_optimization/V2_00_protocol_and_data_lock/v2_development.tsv"
    folds_path = base / "group_folds_v2.tsv"
    binding_path = base / "ASSET_BINDING.json"
    require(binding_path.is_file(), "V2-01C ASSET_BINDING.json is missing")
    binding = json.loads(binding_path.read_text(encoding="utf-8"))
    require(binding.get("status") == "PASS_READY_FOR_BASELINE_OOF", "V2-01C binding is not PASS")

    development = pd.read_csv(development_path, sep="\t", dtype=str, keep_default_na=False)
    folds = pd.read_csv(folds_path, sep="\t", dtype=str, keep_default_na=False)
    require(len(development) == 10567, f"Development row count changed: {len(development)}")
    require({"clean_id", "label", "homology_group"} <= set(development.columns), "Development columns changed")
    require({"clean_id", "oof_fold"} <= set(folds.columns), "Locked fold columns changed")
    require(development["clean_id"].is_unique and folds["clean_id"].is_unique, "Duplicate development/fold IDs")

    metadata = development.merge(
        folds[["clean_id", "oof_fold"]], on="clean_id", how="left", validate="one_to_one"
    )
    require(metadata["oof_fold"].ne("").all(), "Missing locked outer-fold assignment")
    metadata["label"] = metadata["label"].astype(np.int64)
    metadata["oof_fold"] = metadata["oof_fold"].astype(np.int64)
    metadata["cluster_id"] = metadata["homology_group"].astype(str)
    require(set(metadata["oof_fold"]) == {0, 1, 2, 3, 4}, "Unexpected outer folds")

    ankh_dir = root / "work/deepsaltpro_fair_v1/features/full_ankh_window"
    esm2_dir = root / "work/deepsaltpro_fair_v1/features/full_esm2_window"
    ankh_rows, ankh_memmap, ankh_binding = load_feature_dir(ankh_dir, 1536, "Ankh")
    esm2_rows, esm2_memmap, esm2_binding = load_feature_dir(esm2_dir, 2560, "ESM-2")
    ankh_raw = align_features(metadata, ankh_rows, ankh_memmap, "Ankh")
    esm2_raw = align_features(metadata, esm2_rows, esm2_memmap, "ESM-2")
    labels = metadata["label"].to_numpy(dtype=np.int64)
    outer_vector = metadata["oof_fold"].to_numpy(dtype=np.int64)
    groups = metadata["cluster_id"].to_numpy(dtype=str)
    requested_folds = parse_outer_folds(args.outer_folds)

    training_args = SimpleNamespace(
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        patience=args.patience,
        min_mcc_improvement=args.min_mcc_improvement,
    )

    run_contract = {
        "module": "V2-01D",
        "purpose": "DeepSaltPro nested group-OOF teacher generation",
        "requested_outer_folds": requested_folds,
        "outer_fold_source": str(folds_path),
        "outer_fold_source_sha256": sha256_file(folds_path),
        "inner_selection": {
            "algorithm": "StratifiedGroupKFold",
            "n_splits": args.inner_splits,
            "chosen_split": 0,
            "group": "homology_group",
            "use": "epoch selection only",
        },
        "outer_validation_use": "prediction and evaluation only; never early stopping",
        "final_refit": "all outer-fit records for the inner-selected epoch count",
        "preprocessing": "fold-local MinMaxScaler and PCA fitted without outer validation",
        "pca_components_per_backbone": args.pca_components,
        "epochs_cap": args.epochs,
        "blind_labels_read": False,
        "calibration_applied": False,
        "threshold_optimized": False,
        "audited_runner": str(runner_path),
        "audited_runner_sha256": sha256_file(runner_path),
        "ankh": ankh_binding,
        "esm2": esm2_binding,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device),
        "seed": args.seed,
    }
    write_json(output / "RUN_CONTRACT.json", run_contract)

    fold_summaries = []
    prediction_frames = []
    for outer_fold in requested_folds:
        print(f"Outer fold {outer_fold}: nested selection started", flush=True)
        fold_dir = output / f"fold_{outer_fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        outer_validation = np.flatnonzero(outer_vector == outer_fold)
        outer_fit = np.flatnonzero(outer_vector != outer_fold)
        require(set(groups[outer_fit]).isdisjoint(set(groups[outer_validation])), f"Outer fold {outer_fold}: group leakage")

        inner_splitter = StratifiedGroupKFold(
            n_splits=args.inner_splits,
            shuffle=True,
            random_state=args.seed + outer_fold,
        )
        inner_train_local, inner_validation_local = next(
            inner_splitter.split(
                np.zeros(len(outer_fit), dtype=np.int8),
                labels[outer_fit],
                groups=groups[outer_fit],
            )
        )
        inner_train = outer_fit[inner_train_local]
        inner_validation = outer_fit[inner_validation_local]
        require(set(groups[inner_train]).isdisjoint(set(groups[inner_validation])), f"Outer fold {outer_fold}: inner group leakage")

        (
            inner_fit_esm2,
            inner_valid_esm2,
            inner_fit_ankh,
            inner_valid_ankh,
            inner_preprocessing,
            inner_preprocessing_summary,
        ) = fit_transform_pair(
            esm2_raw,
            ankh_raw,
            inner_train,
            inner_validation,
            args.pca_components,
            args.seed + outer_fold * 10,
        )
        joblib.dump(inner_preprocessing, fold_dir / "inner_selection_preprocessing.joblib")
        write_json(fold_dir / "inner_preprocessing_summary.json", inner_preprocessing_summary)

        selection_model, selection_history, inner_best_mcc = legacy.train_one_fold(
            fold_id=outer_fold,
            fit_esm2=inner_fit_esm2,
            fit_ankh=inner_fit_ankh,
            fit_labels=labels[inner_train],
            validation_esm2=inner_valid_esm2,
            validation_ankh=inner_valid_ankh,
            validation_labels=labels[inner_validation],
            device=device,
            args=training_args,
        )
        selection_history.to_csv(fold_dir / "inner_selection_history.tsv", sep="\t", index=False)
        selected_epoch = selected_epoch_from_history(selection_history, args.min_mcc_improvement)
        del selection_model, inner_fit_esm2, inner_valid_esm2, inner_fit_ankh, inner_valid_ankh
        gc.collect()
        torch.cuda.empty_cache()

        (
            final_fit_esm2,
            outer_valid_esm2,
            final_fit_ankh,
            outer_valid_ankh,
            final_preprocessing,
            final_preprocessing_summary,
        ) = fit_transform_pair(
            esm2_raw,
            ankh_raw,
            outer_fit,
            outer_validation,
            args.pca_components,
            args.seed + outer_fold * 10 + 2,
        )
        joblib.dump(final_preprocessing, fold_dir / "final_preprocessing.joblib")
        write_json(fold_dir / "final_preprocessing_summary.json", final_preprocessing_summary)

        final_model, final_history = train_fixed_epochs(
            legacy,
            final_fit_esm2,
            final_fit_ankh,
            labels[outer_fit],
            device,
            args,
            selected_epoch,
            seed_offset=100 + outer_fold,
        )
        final_history.to_csv(fold_dir / "final_refit_history.tsv", sep="\t", index=False)
        probability = legacy.predict_probabilities(
            final_model,
            outer_valid_esm2,
            outer_valid_ankh,
            device,
            args.eval_batch_size,
        )
        clipped = np.clip(probability, 1e-7, 1 - 1e-7)
        raw_logit = np.log(clipped / (1 - clipped))
        metrics = legacy.metric_dict(labels[outer_validation], probability, threshold=0.5)
        torch.save(
            {
                "state_dict": final_model.state_dict(),
                "outer_fold": outer_fold,
                "selected_epoch": selected_epoch,
                "seed": args.seed,
            },
            fold_dir / "final_model.pt",
        )
        predictions = metadata.iloc[outer_validation][
            ["clean_id", "label", "cluster_id", "oof_fold"]
        ].copy()
        predictions["raw_logit"] = raw_logit
        predictions["probability_raw"] = probability
        predictions.to_csv(fold_dir / "outer_validation_predictions.tsv", sep="\t", index=False)
        prediction_frames.append(predictions)

        summary = {
            "outer_fold": outer_fold,
            "status": "PASS",
            "outer_fit_records": int(len(outer_fit)),
            "outer_validation_records": int(len(outer_validation)),
            "outer_group_overlap": 0,
            "inner_train_records": int(len(inner_train)),
            "inner_validation_records": int(len(inner_validation)),
            "inner_group_overlap": 0,
            "inner_best_mcc": float(inner_best_mcc),
            "selected_epoch": selected_epoch,
            "outer_metrics_at_0_5": metrics,
            "outer_validation_used_for_early_stopping": False,
        }
        write_json(fold_dir / "fold_summary.json", summary)
        fold_summaries.append(summary)
        print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)

        del final_model, final_fit_esm2, outer_valid_esm2, final_fit_ankh, outer_valid_ankh
        gc.collect()
        torch.cuda.empty_cache()

    combined = pd.concat(prediction_frames, ignore_index=True).sort_values("clean_id")
    combined.to_csv(output / "completed_outer_fold_predictions.tsv", sep="\t", index=False)
    pd.DataFrame(
        [
            {
                "outer_fold": x["outer_fold"],
                "selected_epoch": x["selected_epoch"],
                "outer_validation_records": x["outer_validation_records"],
                "auroc": x["outer_metrics_at_0_5"]["roc_auc"],
                "ap": x["outer_metrics_at_0_5"]["pr_auc"],
                "mcc_at_0_5": x["outer_metrics_at_0_5"]["mcc"],
            }
            for x in fold_summaries
        ]
    ).to_csv(output / "fold_summary.tsv", sep="\t", index=False)

    complete = requested_folds == [0, 1, 2, 3, 4]
    if complete:
        require(len(combined) == len(metadata), "Full OOF row count is incomplete")
        require(combined["clean_id"].is_unique, "Full OOF has duplicate IDs")
        combined.to_csv(output / "deepsaltpro_oof_predictions.tsv", sep="\t", index=False)
        overall = legacy.metric_dict(
            combined["label"].to_numpy(dtype=np.int64),
            combined["probability_raw"].to_numpy(dtype=np.float64),
            threshold=0.5,
        )
        status = "PASS_COMPLETE_OOF"
    else:
        overall = None
        status = "SMOKE_PASS"

    run_summary = {
        "module": "V2-01D",
        "status": status,
        "started_at_utc": started,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round(time.time() - start_time, 3),
        "requested_outer_folds": requested_folds,
        "epochs_cap": args.epochs,
        "completed_records": int(len(combined)),
        "overall_oof_at_0_5": overall,
        "blind_labels_read": False,
        "outer_validation_used_for_early_stopping": False,
        "fold_summaries": fold_summaries,
    }
    write_json(output / "RUN_SUMMARY.json", run_summary)
    print(json.dumps(run_summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"V2-01D FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
