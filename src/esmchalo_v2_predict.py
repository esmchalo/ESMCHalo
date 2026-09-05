#!/usr/bin/env python3
"""Frozen ESMCHalo-v2 FASTA inference; contains no training or tuning code."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "legacy_extractor"))
sys.path.insert(0, str(HERE))

from esmchalo_predict import ESMCEmbedder, parse_fasta  # noqa: E402
from frozen_ensemble_inference import calibrate, score  # noqa: E402


MEMBER_HASHES = {
    "fold_0.pt": "57a12bba50e19bebc94cb839ba05639276b4b004892283efb867adf5c5480ae4",
    "fold_1.pt": "1e49d4f98d7ba8e2a567e9a7bf1298db43cbde20389d7b7243a41ce753c2c9bf",
    "fold_2.pt": "5685ca6f2ec06e4ac19c44cdbf8d13347d2ab00646a17a995b366e279ec3d023",
    "fold_3.pt": "c40651770ae2f689415cefdd516e7f2a74a60efd5c2f70b26a798c460f1606a2",
    "fold_4.pt": "6c290a6b7f5de70c5ff9052df2565486008f54056dccbb84d521216d545c524a",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    p = argparse.ArgumentParser(description="Frozen ESMCHalo-v2 FASTA inference")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="auto")
    p.add_argument("--dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--student-batch-size", type=int, default=512)
    p.add_argument("--cache-dir", type=Path)
    p.add_argument("--allow-model-download", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    if args.output.exists() and not args.overwrite:
        raise SystemExit(f"Output exists: {args.output}")
    if args.batch_size < 1 or args.student_batch_size < 1:
        raise SystemExit("Batch sizes must be positive")

    release = HERE.parent
    freeze = release / "artifacts" / "frozen_ensemble"
    members = sorted((freeze / "members").glob("fold_*.pt"))
    if [p.name for p in members] != list(MEMBER_HASHES):
        raise RuntimeError("Exactly five frozen members are required")
    for member in members:
        if sha256(member) != MEMBER_HASHES[member.name]:
            raise RuntimeError(f"Frozen member hash mismatch: {member.name}")

    records = parse_fasta(args.input)
    embedder = ESMCEmbedder(
        device_name=args.device,
        dtype_name=args.dtype,
        allow_model_download=args.allow_model_download,
        cache_dir=args.cache_dir,
    )
    matrices = []
    windows = []
    for offset in range(0, len(records), args.batch_size):
        x, nwin = embedder.embed_batch(records[offset:offset + args.batch_size])
        matrices.append(x)
        windows.append(nwin)
    features = np.concatenate(matrices).astype(np.float32)
    nwin = np.concatenate(windows)

    device_name = args.device
    if device_name == "auto":
        device_name = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    raw = np.mean(
        np.vstack([score(m, features, device, args.student_batch_size) for m in members]),
        axis=0,
    )
    calibrator = json.loads((freeze / "final_calibrator.json").read_text())
    threshold_spec = json.loads((freeze / "final_threshold.json").read_text())
    probability = calibrate(raw, calibrator)
    threshold = float(threshold_spec["deployment_threshold"])
    prediction = (probability >= threshold).astype(int)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "input_order", "record_id", "description", "length", "number_of_windows",
            "sequence_sha256", "ensemble_mean_raw_logit", "probability_calibrated",
            "fixed_threshold", "prediction", "prediction_label",
        ], delimiter="\t")
        writer.writeheader()
        for i, record in enumerate(records):
            writer.writerow({
                "input_order": i + 1,
                "record_id": record.record_id,
                "description": record.description.replace("\t", " "),
                "length": len(record.sequence),
                "number_of_windows": int(nwin[i]),
                "sequence_sha256": hashlib.sha256(record.sequence.encode()).hexdigest(),
                "ensemble_mean_raw_logit": f"{raw[i]:.12g}",
                "probability_calibrated": f"{probability[i]:.12g}",
                "fixed_threshold": threshold,
                "prediction": int(prediction[i]),
                "prediction_label": "halophilic_like" if prediction[i] else "non_halophilic_like",
            })

    run = {
        "status": "PASS",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "records": len(records),
        "representation": "ESMC-600M layer36 canonical mean, 1152 dimensions",
        "ensemble": "five fold students; equal mean raw logit",
        "calibration": calibrator,
        "threshold": threshold_spec,
        "model_training": False,
        "threshold_optimization": False,
        "DeepSaltPro_required_at_inference": False,
    }
    args.output.with_suffix(args.output.suffix + ".run.json").write_text(
        json.dumps(run, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
