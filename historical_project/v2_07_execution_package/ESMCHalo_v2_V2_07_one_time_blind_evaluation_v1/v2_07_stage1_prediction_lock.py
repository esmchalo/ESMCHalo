#!/usr/bin/env python3
"""Stage 1: label-inaccessible prediction generation and immutable lock."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from v2_07_common import require, sha256_file, sigmoid, write_json


EXPECTED_ESMCHALO = {
    "fold_0.pt": "57a12bba50e19bebc94cb839ba05639276b4b004892283efb867adf5c5480ae4",
    "fold_1.pt": "1e49d4f98d7ba8e2a567e9a7bf1298db43cbde20389d7b7243a41ce753c2c9bf",
    "fold_2.pt": "5685ca6f2ec06e4ac19c44cdbf8d13347d2ab00646a17a995b366e279ec3d023",
    "fold_3.pt": "c40651770ae2f689415cefdd516e7f2a74a60efd5c2f70b26a798c460f1606a2",
    "fold_4.pt": "6c290a6b7f5de70c5ff9052df2565486008f54056dccbb84d521216d545c524a",
}
EXPECTED_DEEPSALTPRO_MODELS = {
    0: "95cdba5d18b01969eb0f63c99eba3b42ab7d85555f83caa5ca1f5679da583675",
    1: "4a4881e803e0e571a6b2f7839912b3ec6c041111adbc532635dafca93372a906",
    2: "e1712dc98f416ebee359594be58015f330a58bca5745e4efc23004eef58488b8",
    3: "33d1d8232faa61bf9ac56f77ad57dee45831dbbee0737421868c7f15777f429f",
    4: "7e589f7ea50b4e79f6cce773a2a949892a8e07a5a2753913ae2f73e8918901f9",
}
EXPECTED_DEEPSALTPRO_PREPROCESSING = {
    0: "a4b5c832d01299926b891604495c37d3edfbfe23846cbdb6c0aaf98277fd3e9b",
    1: "5d4d0d25df0f383054f38c984683d48d0eab622025c01b2455b45a509cef9cc0",
    2: "a852fd149939d8d0ee72ef14fea4998f9c192274cc83cb063b9ed4852e1ce220",
    3: "2057d3b97014c979a79ba03adcdd28bf26d1bfbb3df179951be527d193d63d57",
    4: "c50b30de6f0812e57846680816dbdc34c89d14aa0367110adb2a46caab0780c7",
}
EXPECTED_RUNNER = "8619990b59c092bd563c4d7499ac1acab6da31a37421a6847ef2333a629feb5e"
EXPECTED_ANKH_MATRIX = "39065de47e630627c5a4048872bce1af314bd3c6d1ba4695aa178802af70dfaf"
EXPECTED_ESM2_MATRIX = "9d5e26a22ce41d44bc97d220da89cb8704b5fa1eb4c2fe42dd93ae23dc754103"


class Student(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(1152, 256), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(256, 64), nn.GELU(), nn.Dropout(0.2), nn.Linear(64, 1),
        )

    def forward(self, values):
        return self.network(values).squeeze(-1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/home/lvfang/ESMC_halophile"))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=512)
    return parser.parse_args()


def acquire_lock(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o444)
    except FileExistsError as exc:
        raise RuntimeError(f"Stage 1 lock already exists: {path}") from exc
    with os.fdopen(descriptor, "w") as handle:
        handle.write(f"pid={os.getpid()}\n")


def align_label_free_features(directory: Path, dimension: int, ids: list[str]):
    rows_path = directory / "rows.tsv"
    matrix_path = directory / "embeddings.npy"
    require(rows_path.is_file() and matrix_path.is_file(), f"Missing feature assets: {directory}")
    header = pd.read_csv(rows_path, sep="\t", nrows=0)
    require("clean_id" in header.columns, f"clean_id missing: {rows_path}")
    rows = pd.read_csv(rows_path, sep="\t", dtype=str, usecols=["clean_id"])
    require(rows["clean_id"].is_unique, f"Duplicate IDs: {rows_path}")
    matrix = np.load(matrix_path, mmap_mode="r", allow_pickle=False)
    require(matrix.shape == (len(rows), dimension), f"Unexpected shape: {matrix.shape}")
    lookup = {value: index for index, value in enumerate(rows["clean_id"].astype(str))}
    missing = [value for value in ids if value not in lookup]
    require(not missing, f"{len(missing)} blind IDs missing from {directory}")
    order = np.asarray([lookup[value] for value in ids], dtype=np.int64)
    selected = np.asarray(matrix[order], dtype=np.float32)
    require(np.isfinite(selected).all(), f"Non-finite values: {directory}")
    return selected, rows_path, matrix_path, order


@torch.inference_mode()
def esmchalo_score(checkpoint_path: Path, matrix: np.ndarray, device, batch_size):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    require(checkpoint.get("arm") == "H0", f"Not H0: {checkpoint_path}")
    require(checkpoint.get("canonical_mean_only") is True, "Not canonical mean")
    require(checkpoint.get("lora") is False, "LoRA checkpoint rejected")
    model = Student().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    mean = np.asarray(checkpoint["scaler_mean"], dtype=np.float32)
    scale = np.asarray(checkpoint["scaler_scale"], dtype=np.float32)
    standardized = (matrix - mean) / scale
    result = []
    loader = DataLoader(
        torch.from_numpy(np.ascontiguousarray(standardized, dtype=np.float32)),
        batch_size=batch_size,
        shuffle=False,
    )
    for batch in loader:
        result.append(model(batch.to(device)).float().cpu().numpy())
    return np.concatenate(result).astype(np.float64)


def load_deepsaltpro_runner(root: Path):
    code = root / "external/DeepSaltPro/Code"
    runner = root / "scripts/deepsaltpro_fair_stage4b/train_deepsaltpro_fair.py"
    require(runner.is_file(), f"Missing runner: {runner}")
    require(sha256_file(runner) == EXPECTED_RUNNER, "DeepSaltPro runner hash changed")
    sys.path.insert(0, str(code))
    spec = importlib.util.spec_from_file_location("locked_deepsaltpro_runner", runner)
    require(spec is not None and spec.loader is not None, "Cannot load DeepSaltPro runner")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, runner


def locked_private_label_hash(lock_dir: Path) -> str:
    checksum_path = lock_dir / "SHA256SUMS"
    require(checksum_path.is_file(), "V2-00 SHA256SUMS missing")
    matches = []
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        fields = line.strip().split(maxsplit=1)
        if len(fields) == 2 and fields[1].lstrip("*").endswith("v2_blind_labels_private.tsv"):
            matches.append(fields[0])
    require(len(matches) == 1 and len(matches[0]) == 64, "Frozen private-label hash not found")
    return matches[0]


def main() -> None:
    args = parse_args()
    root = args.root.expanduser().resolve()
    output = args.output or root / (
        "ESMCHalo_v2_optimization/V2_07_blind_evaluation/"
        "V2_07_one_time_blind_evaluation_v1"
    )
    output = output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    require(not (output / "PREDICTION_LOCK.json").exists(), "Predictions already locked")
    require(not (output / "V2_07_DONE.json").exists(), "V2-07 already complete")
    acquire_lock(output / ".STAGE1_RUNNING.lock")
    write_json(output / "STATUS.json", {
        "module": "V2-07", "status": "STAGE1_PREDICTION_RUNNING",
        "blind_labels_read": False, "lora": False,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    })

    device = torch.device(args.device)
    require(device.type != "cuda" or torch.cuda.is_available(), "CUDA unavailable")
    lock_dir = root / "ESMCHalo_v2_optimization/V2_00_protocol_and_data_lock"
    expected_private_label_sha256 = locked_private_label_hash(lock_dir)
    blind_ids_path = lock_dir / "v2_blind_ids.tsv"
    safe_columns = ["clean_id", "sequence_hash", "homology_group", "strict25_member", "v2_role"]
    header = pd.read_csv(blind_ids_path, sep="\t", nrows=0)
    require(set(safe_columns) <= set(header.columns), "Blind ID schema changed")
    blind = pd.read_csv(blind_ids_path, sep="\t", dtype=str, usecols=safe_columns, keep_default_na=False)
    require(len(blind) == 1865 and blind["clean_id"].is_unique, "Blind ID lock failed")
    require(blind["homology_group"].ne("").all(), "Missing homology_group")
    require(blind["v2_role"].eq("blind").all(), "Unexpected v2_role")
    ids = blind["clean_id"].astype(str).tolist()

    canonical = root / "data/embeddings/esmc600m_canonical_b1"
    x_esmc, esmc_rows, esmc_matrix, esmc_order = align_label_free_features(canonical, 1152, ids)
    ankh_dir = root / "work/deepsaltpro_fair_v1/features/full_ankh_window"
    esm2_dir = root / "work/deepsaltpro_fair_v1/features/full_esm2_window"
    x_ankh, ankh_rows, ankh_matrix, ankh_order = align_label_free_features(ankh_dir, 1536, ids)
    x_esm2, esm2_rows, esm2_matrix, esm2_order = align_label_free_features(esm2_dir, 2560, ids)
    require(sha256_file(ankh_matrix) == EXPECTED_ANKH_MATRIX, "Ankh matrix hash changed")
    require(sha256_file(esm2_matrix) == EXPECTED_ESM2_MATRIX, "ESM-2 matrix hash changed")

    freeze = root / (
        "ESMCHalo_v2_optimization/V2_06_ensemble_calibration_freeze/"
        "V2_06_ensemble_calibration_freeze_v1/frozen_ensemble"
    )
    esmc_raw = []
    model_binding = {"esmchalo_v2": [], "deepsaltpro": []}
    for name, expected in EXPECTED_ESMCHALO.items():
        path = freeze / "members" / name
        actual = sha256_file(path)
        require(actual == expected, f"ESMCHalo member hash changed: {name}")
        esmc_raw.append(esmchalo_score(path, x_esmc, device, args.batch_size))
        model_binding["esmchalo_v2"].append({"file": str(path), "sha256": actual})
    esmc_raw_matrix = np.vstack(esmc_raw)
    esmc_mean_raw = esmc_raw_matrix.mean(axis=0)
    calibrator = json.loads((freeze / "final_calibrator.json").read_text())
    threshold = json.loads((freeze / "final_threshold.json").read_text())
    require(calibrator.get("type") == "platt" and calibrator.get("name") == "C1", "Calibrator changed")
    require(float(calibrator["coefficient"]) == 1.65445192210991, "Platt coefficient changed")
    require(float(calibrator["intercept"]) == -0.059280024051954906, "Platt intercept changed")
    require(float(threshold["deployment_threshold"]) == 0.5, "Threshold changed")
    esmc_probability = sigmoid(
        float(calibrator["coefficient"]) * esmc_mean_raw + float(calibrator["intercept"])
    )

    legacy, runner = load_deepsaltpro_runner(root)
    deepsalt_base = root / (
        "ESMCHalo_v2_optimization/V2_01_baseline_and_teacher_oof/"
        "deepsaltpro_nested_oof_full_v1"
    )
    ds_probabilities = []
    for fold in range(5):
        model_path = deepsalt_base / f"fold_{fold}/final_model.pt"
        prep_path = deepsalt_base / f"fold_{fold}/final_preprocessing.joblib"
        model_hash = sha256_file(model_path)
        prep_hash = sha256_file(prep_path)
        require(model_hash == EXPECTED_DEEPSALTPRO_MODELS[fold], f"DeepSaltPro model {fold} changed")
        require(prep_hash == EXPECTED_DEEPSALTPRO_PREPROCESSING[fold], f"DeepSaltPro preprocessing {fold} changed")
        preprocessing = joblib.load(prep_path)
        require(set(preprocessing) == {"esm2_scaler", "ankh_scaler", "esm2_pca", "ankh_pca"}, "Preprocessing schema changed")
        fold_esm2 = preprocessing["esm2_pca"].transform(
            preprocessing["esm2_scaler"].transform(x_esm2)
        ).astype(np.float32)
        fold_ankh = preprocessing["ankh_pca"].transform(
            preprocessing["ankh_scaler"].transform(x_ankh)
        ).astype(np.float32)
        require(fold_esm2.shape == (1865, 512) and fold_ankh.shape == (1865, 512), "PCA shape changed")
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        require(int(checkpoint["outer_fold"]) == fold, "DeepSaltPro fold mismatch")
        model = legacy.DeepSaltProNet(device).to(device)
        model.load_state_dict(checkpoint["state_dict"])
        probability = legacy.predict_probabilities(
            model, fold_esm2, fold_ankh, device, args.batch_size
        )
        require(len(probability) == 1865 and np.isfinite(probability).all(), "Invalid DeepSaltPro probability")
        ds_probabilities.append(probability)
        model_binding["deepsaltpro"].append({
            "fold": fold, "model": str(model_path), "model_sha256": model_hash,
            "preprocessing": str(prep_path), "preprocessing_sha256": prep_hash,
        })
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    ds_probability_matrix = np.vstack(ds_probabilities)
    ds_probability = ds_probability_matrix.mean(axis=0)

    predictions = blind[["clean_id", "sequence_hash", "homology_group", "strict25_member"]].copy()
    for fold in range(5):
        predictions[f"esmchalo_raw_logit_fold_{fold}"] = esmc_raw_matrix[fold]
    predictions["esmchalo_mean_raw_logit"] = esmc_mean_raw
    predictions["esmchalo_probability_calibrated"] = esmc_probability
    predictions["esmchalo_prediction_at_0_5"] = (esmc_probability >= 0.5).astype(np.int64)
    for fold in range(5):
        predictions[f"deepsaltpro_probability_fold_{fold}"] = ds_probability_matrix[fold]
    predictions["deepsaltpro_probability_mean"] = ds_probability
    predictions["deepsaltpro_prediction_at_0_5"] = (ds_probability >= 0.5).astype(np.int64)
    require(predictions["clean_id"].is_unique and len(predictions) == 1865, "Prediction binding failed")
    require(np.isfinite(predictions.select_dtypes(include=[np.number])).all().all(), "Non-finite predictions")
    prediction_path = output / "blind_predictions_LOCKED.tsv"
    predictions.to_csv(prediction_path, sep="\t", index=False)
    prediction_sha = sha256_file(prediction_path)

    safe_binding = pd.DataFrame({
        "clean_id": blind["clean_id"],
        "sequence_hash": blind["sequence_hash"],
        "homology_group": blind["homology_group"],
        "canonical_position": esmc_order,
        "ankh_position": ankh_order,
        "esm2_position": esm2_order,
    })
    binding_payload = safe_binding.to_csv(sep="\t", index=False).encode()
    safe_binding_sha = __import__("hashlib").sha256(binding_payload).hexdigest()
    write_json(output / "FEATURE_AND_MODEL_BINDING.json", {
        "status": "PASS_LABEL_FREE_BINDING",
        "records": 1865,
        "blind_ids_sha256": sha256_file(blind_ids_path),
        "safe_binding_sha256": safe_binding_sha,
        "canonical_matrix": str(esmc_matrix),
        "canonical_matrix_sha256": sha256_file(esmc_matrix),
        "ankh_matrix": str(ankh_matrix),
        "ankh_matrix_sha256": EXPECTED_ANKH_MATRIX,
        "esm2_matrix": str(esm2_matrix),
        "esm2_matrix_sha256": EXPECTED_ESM2_MATRIX,
        "feature_row_files_opened_with_usecols_clean_id_only": [str(esmc_rows), str(ankh_rows), str(esm2_rows)],
        "models": model_binding,
        "deepsaltpro_runner": str(runner),
        "deepsaltpro_runner_sha256": EXPECTED_RUNNER,
        "blind_labels_read": False,
    })
    prediction_lock = {
        "module": "V2-07", "stage": 1, "status": "PASS_PREDICTIONS_LOCKED",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "records": 1865, "prediction_file": str(prediction_path),
        "prediction_sha256": prediction_sha,
        "esmchalo_aggregation": "mean_raw_logit_then_C1_Platt",
        "esmchalo_threshold": 0.5,
        "deepsaltpro_aggregation": "mean_probability",
        "deepsaltpro_threshold": 0.5,
        "bootstrap_group_field_locked": "homology_group",
        "private_label_expected_sha256_from_V2_00_lock": expected_private_label_sha256,
        "blind_labels_read": False, "test200_labels_read": False,
        "external_challenge_labels_read": False, "model_retraining": False,
        "recalibration": False, "threshold_optimization": False, "lora": False,
        "next_gate": "Independent review before one-time Stage 2 reveal",
    }
    write_json(output / "PREDICTION_LOCK.json", prediction_lock)
    os.chmod(prediction_path, 0o444)
    write_json(output / "STATUS.json", prediction_lock)
    os.chmod(output / "PREDICTION_LOCK.json", 0o444)
    print(json.dumps(prediction_lock, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
