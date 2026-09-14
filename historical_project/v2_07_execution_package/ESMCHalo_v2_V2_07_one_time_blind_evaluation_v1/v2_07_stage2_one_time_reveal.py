#!/usr/bin/env python3
"""Stage 2: one-time label reveal after Stage-1 prediction lock review."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from v2_07_common import metric_dict, require, sha256_file, write_json


CONFIRMATION = "I_CONFIRM_ONE_TIME_BLIND_REVEAL_AFTER_PREDICTION_LOCK_REVIEW"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/home/lvfang/ESMC_halophile"))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--confirm", required=True)
    return parser.parse_args()


def acquire_lock(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o444)
    except FileExistsError as exc:
        raise RuntimeError("Stage 2 has already been started; rerun prohibited") from exc
    with os.fdopen(descriptor, "w") as handle:
        handle.write(f"pid={os.getpid()}\n")


def bootstrap_indices(groups: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    unique = np.unique(groups)
    sampled = rng.choice(unique, size=len(unique), replace=True)
    lookup = {group: np.flatnonzero(groups == group) for group in unique}
    return np.concatenate([lookup[group] for group in sampled])


def main() -> None:
    args = parse_args()
    require(args.confirm == CONFIRMATION, "Exact one-time reveal confirmation is required")
    root = args.root.expanduser().resolve()
    output = args.output or root / (
        "ESMCHalo_v2_optimization/V2_07_blind_evaluation/"
        "V2_07_one_time_blind_evaluation_v1"
    )
    output = output.expanduser().resolve()
    require(not (output / "V2_07_DONE.json").exists(), "V2-07 is already complete")
    lock_path = output / "PREDICTION_LOCK.json"
    require(lock_path.is_file(), "Stage-1 PREDICTION_LOCK.json missing")
    lock = json.loads(lock_path.read_text())
    require(lock.get("status") == "PASS_PREDICTIONS_LOCKED", "Prediction lock not accepted")
    prediction_path = Path(lock["prediction_file"])
    require(prediction_path.is_file(), "Locked prediction file missing")
    require(sha256_file(prediction_path) == lock["prediction_sha256"], "Locked predictions changed")
    acquire_lock(output / ".STAGE2_ONE_TIME_REVEAL.lock")

    # This is the only code path that opens the frozen private labels.
    label_path = root / (
        "ESMCHalo_v2_optimization/V2_00_protocol_and_data_lock/"
        "v2_blind_labels_private.tsv"
    )
    # One physical read: the same immutable byte buffer supplies both SHA256 and parsing.
    label_bytes = label_path.read_bytes()
    label_sha_before = hashlib.sha256(label_bytes).hexdigest()
    require(
        label_sha_before == lock["private_label_expected_sha256_from_V2_00_lock"],
        "Private blind labels do not match the V2-00 frozen SHA256",
    )
    labels = pd.read_csv(io.BytesIO(label_bytes), sep="\t", dtype=str, keep_default_na=False)
    require("clean_id" in labels.columns, "clean_id absent from private labels")
    label_column = next(
        (name for name in ("label", "target", "y", "class") if name in labels.columns),
        None,
    )
    require(label_column is not None, "Binary label column not found")
    require(len(labels) == 1865 and labels["clean_id"].is_unique, "Private label lock failed")
    labels[label_column] = labels[label_column].astype(np.int64)
    require(set(labels[label_column]) == {0, 1}, "Labels are not binary")
    predictions = pd.read_csv(prediction_path, sep="\t", dtype={"clean_id": str, "homology_group": str})
    merged = predictions.merge(
        labels[["clean_id", label_column]], on="clean_id", how="left", validate="one_to_one"
    )
    require(len(merged) == 1865 and merged[label_column].notna().all(), "Label/prediction merge failed")
    require(set(labels["clean_id"]) == set(predictions["clean_id"]), "Blind ID sets differ")
    merged = merged.rename(columns={label_column: "label"})
    y = merged["label"].to_numpy(dtype=np.int64)
    esmc = merged["esmchalo_probability_calibrated"].to_numpy(dtype=np.float64)
    ds = merged["deepsaltpro_probability_mean"].to_numpy(dtype=np.float64)
    groups = merged["homology_group"].astype(str).to_numpy()
    require(np.all(groups != "") and len(np.unique(groups)) > 1, "Invalid homology groups")

    metrics = {
        "ESMCHalo_v2": metric_dict(y, esmc, 0.5),
        "DeepSaltPro": metric_dict(y, ds, 0.5),
    }
    metric_names = [
        "ap", "auroc", "mcc", "accuracy", "balanced_accuracy",
        "sensitivity", "specificity", "ppv", "f1", "brier",
        "ece_10_bins", "log_loss",
    ]
    observed = {
        name: metrics["ESMCHalo_v2"][name] - metrics["DeepSaltPro"][name]
        for name in metric_names
    }

    rng = np.random.default_rng(20260903)
    bootstrap_rows = []
    attempts = 0
    while len(bootstrap_rows) < 5000:
        attempts += 1
        require(attempts <= 100000, "Could not obtain 5000 valid group replicates")
        index = bootstrap_indices(groups, rng)
        if len(np.unique(y[index])) != 2:
            continue
        em = metric_dict(y[index], esmc[index], 0.5)
        dm = metric_dict(y[index], ds[index], 0.5)
        row = {"replicate": len(bootstrap_rows) + 1}
        for name in metric_names:
            row[f"esmchalo_{name}"] = em[name]
            row[f"deepsaltpro_{name}"] = dm[name]
            row[f"delta_{name}"] = em[name] - dm[name]
        bootstrap_rows.append(row)
    bootstrap = pd.DataFrame(bootstrap_rows)
    bootstrap.to_csv(output / "paired_group_bootstrap_replicates.tsv", sep="\t", index=False)

    summary_rows = []
    for name in metric_names:
        values = bootstrap[f"delta_{name}"].to_numpy(dtype=np.float64)
        summary_rows.append({
            "metric": name,
            "direction": "higher_is_better" if name not in {"brier", "ece_10_bins", "log_loss"} else "lower_is_better",
            "observed_delta_ESMCHalo_minus_DeepSaltPro": observed[name],
            "bootstrap_mean": float(values.mean()),
            "ci_2_5": float(np.quantile(values, 0.025)),
            "median": float(np.quantile(values, 0.5)),
            "ci_97_5": float(np.quantile(values, 0.975)),
            "fraction_gt_zero": float(np.mean(values > 0)),
        })
    pd.DataFrame(summary_rows).to_csv(
        output / "paired_group_bootstrap_summary.tsv", sep="\t", index=False
    )

    esmc_correct = (esmc >= 0.5).astype(int) == y
    ds_correct = (ds >= 0.5).astype(int) == y
    esmc_only = int(np.sum(esmc_correct & ~ds_correct))
    ds_only = int(np.sum(~esmc_correct & ds_correct))
    discordant = esmc_only + ds_only
    pvalue = 1.0 if discordant == 0 else float(
        binomtest(min(esmc_only, ds_only), discordant, 0.5, alternative="two-sided").pvalue
    )
    mcnemar = {
        "ESMCHalo_correct_DeepSaltPro_wrong": esmc_only,
        "ESMCHalo_wrong_DeepSaltPro_correct": ds_only,
        "discordant": discordant,
        "exact_two_sided_p": pvalue,
    }
    write_json(output / "mcnemar_exact.json", mcnemar)
    metrics_rows = []
    for model, values in metrics.items():
        metrics_rows.append({"model": model, **values})
    pd.DataFrame(metrics_rows).to_csv(output / "blind_metrics.tsv", sep="\t", index=False)
    merged.to_csv(output / "blind_predictions_with_frozen_labels.tsv", sep="\t", index=False)

    done = {
        "module": "V2-07", "status": "PASS_BLIND_EVALUATION_COMPLETE",
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "records": 1865, "groups": int(len(np.unique(groups))),
        "label_file": str(label_path), "label_sha256_verified_against_V2_00_lock": label_sha_before,
        "prediction_sha256_verified": lock["prediction_sha256"],
        "bootstrap_replicates": 5000, "bootstrap_seed": 20260903,
        "metrics": metrics, "observed_deltas_ESMCHalo_minus_DeepSaltPro": observed,
        "mcnemar": mcnemar, "blind_labels_read_once": True,
        "test200_labels_read": False, "external_challenge_labels_read": False,
        "model_retraining": False, "recalibration": False,
        "threshold_optimization": False, "lora": False,
        "post_blind_model_changes_authorized": False,
        "next_gate": "V2-08 reporting and reproducibility update only",
    }
    write_json(output / "V2_07_DONE.json", done)
    write_json(output / "DATA_ACCESS_AUDIT.json", {
        "status": "PASS_ONE_TIME_BLIND_LABEL_ACCESS",
        "stage1_blind_labels_read": False,
        "stage2_private_label_physical_reads": 1,
        "private_label_sha256": label_sha_before,
        "records_joined": 1865,
        "test200_labels_read": False,
        "external_challenge_labels_read": False,
        "model_retraining": False,
        "recalibration": False,
        "threshold_optimization": False,
        "lora": False
    })
    write_json(output / "STATUS.json", done)
    print(json.dumps(done, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
