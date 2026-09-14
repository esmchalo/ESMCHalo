#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from numpy.lib.format import open_memmap
from transformers import AutoConfig, AutoModelForMaskedLM, AutoTokenizer


REQUIRED_COLUMNS = {
    "clean_id",
    "original_id",
    "split",
    "label",
    "length",
    "cluster_id",
    "sequence",
}
STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract frozen ESMC-600M mean-pooled protein embeddings from "
            "strict_manifest.tsv with batching, long-sequence windowing, "
            "checkpointing, and resume support."
        )
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Process only the first N rows; 0 means all rows.",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=2046,
        help="Maximum amino-acid residues per model call.",
    )
    parser.add_argument(
        "--window-overlap",
        type=int,
        default=256,
        help="Residue overlap for sequences longer than window-size.",
    )
    parser.add_argument(
        "--max-batch-size",
        type=int,
        default=16,
        help="Maximum number of short sequences in one batch.",
    )
    parser.add_argument(
        "--max-batch-tokens",
        type=int,
        default=8192,
        help="Approximate padded-token budget per short-sequence batch.",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=100,
        help="Flush progress after this many newly completed proteins.",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Torch device, normally cuda:0.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete existing extraction files in output-dir and restart.",
    )
    return parser.parse_args()


def read_manifest(path: Path, limit: int) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Manifest not found: {path}")

    records: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fieldnames = set(reader.fieldnames or [])
        missing = REQUIRED_COLUMNS - fieldnames
        if missing:
            raise ValueError(
                f"Manifest is missing required columns: {sorted(missing)}"
            )

        for row_number, row in enumerate(reader, start=2):
            if limit > 0 and len(records) >= limit:
                break

            clean_id = row["clean_id"].strip()
            sequence = "".join(row["sequence"].split()).upper()
            declared_length = int(row["length"])

            if not clean_id:
                raise ValueError(f"Empty clean_id at manifest row {row_number}")
            if clean_id in seen_ids:
                raise ValueError(f"Duplicate clean_id: {clean_id}")
            if len(sequence) != declared_length:
                raise ValueError(
                    f"Length mismatch for {clean_id}: "
                    f"manifest={declared_length}, actual={len(sequence)}"
                )

            invalid = sorted(set(sequence) - STANDARD_AA)
            if invalid:
                raise ValueError(
                    f"Non-standard residues in {clean_id}: {invalid}. "
                    "The cleaned manifest should contain only 20 standard residues."
                )

            seen_ids.add(clean_id)
            records.append(
                {
                    "clean_id": clean_id,
                    "original_id": row["original_id"].strip(),
                    "split": row["split"].strip(),
                    "label": int(row["label"]),
                    "length": declared_length,
                    "cluster_id": row["cluster_id"].strip(),
                    "sequence": sequence,
                }
            )

    if not records:
        raise ValueError("No records were loaded from the manifest.")

    return records


def records_digest(records: list[dict[str, Any]]) -> str:
    hasher = hashlib.sha256()
    for record in records:
        hasher.update(record["clean_id"].encode("utf-8"))
        hasher.update(b"\0")
        hasher.update(record["sequence"].encode("ascii"))
        hasher.update(b"\n")
    return hasher.hexdigest()


def make_short_batches(
    indices: Iterable[int],
    records: list[dict[str, Any]],
    max_batch_size: int,
    max_batch_tokens: int,
) -> list[list[int]]:
    ordered = sorted(indices, key=lambda i: records[i]["length"])
    batches: list[list[int]] = []
    current: list[int] = []
    current_max_tokens = 0

    for index in ordered:
        token_length = records[index]["length"] + 2
        proposed_max = max(current_max_tokens, token_length)
        proposed_size = len(current) + 1
        proposed_tokens = proposed_max * proposed_size

        if current and (
            proposed_size > max_batch_size
            or proposed_tokens > max_batch_tokens
        ):
            batches.append(current)
            current = [index]
            current_max_tokens = token_length
        else:
            current.append(index)
            current_max_tokens = proposed_max

    if current:
        batches.append(current)

    return batches


def window_starts(length: int, window_size: int, overlap: int) -> list[int]:
    if length <= window_size:
        return [0]

    step = window_size - overlap
    if step <= 0:
        raise ValueError("window-overlap must be smaller than window-size.")

    starts = list(range(0, length - window_size + 1, step))
    final_start = length - window_size
    if starts[-1] != final_start:
        starts.append(final_start)
    return starts


def get_final_hidden(output: Any) -> torch.Tensor:
    hidden = getattr(output, "last_hidden_state", None)
    if hidden is not None:
        return hidden

    hidden_states = getattr(output, "hidden_states", None)
    if hidden_states is not None:
        return hidden_states[-1]

    fields = list(output.keys()) if hasattr(output, "keys") else []
    raise RuntimeError(
        "Model output contains neither last_hidden_state nor hidden_states. "
        f"Available fields: {fields}"
    )


def tokenize(
    tokenizer: Any,
    sequences: list[str],
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    encoded = tokenizer(
        sequences,
        return_tensors="pt",
        padding=True,
        return_special_tokens_mask=True,
    )

    special_tokens_mask = encoded.pop("special_tokens_mask").bool()
    attention_mask = encoded["attention_mask"].bool()
    model_inputs = {
        key: value.to(device, non_blocking=True)
        for key, value in encoded.items()
    }
    return model_inputs, attention_mask, special_tokens_mask


def extract_short_batch(
    model: Any,
    tokenizer: Any,
    sequences: list[str],
    device: torch.device,
) -> np.ndarray:
    model_inputs, attention_mask, special_tokens_mask = tokenize(
        tokenizer, sequences, device
    )

    with torch.inference_mode():
        output = model(**model_inputs, return_dict=True)

    final_hidden = get_final_hidden(output)
    residue_mask = (
        attention_mask & ~special_tokens_mask
    ).to(device)

    residue_counts = residue_mask.sum(dim=1)
    expected = torch.tensor(
        [len(sequence) for sequence in sequences],
        dtype=residue_counts.dtype,
        device=device,
    )
    if not torch.equal(residue_counts, expected):
        raise RuntimeError(
            "Token/residue count mismatch: "
            f"observed={residue_counts.tolist()}, expected={expected.tolist()}"
        )

    weights = residue_mask.unsqueeze(-1).to(final_hidden.dtype)
    denominator = weights.sum(dim=1).clamp_min(1)
    pooled = (final_hidden * weights).sum(dim=1) / denominator

    array = pooled.float().cpu().numpy()
    if not np.isfinite(array).all():
        raise FloatingPointError("NaN or Inf found in a short-sequence batch.")
    return array


def extract_long_sequence(
    model: Any,
    tokenizer: Any,
    sequence: str,
    embedding_dim: int,
    window_size: int,
    overlap: int,
    device: torch.device,
) -> tuple[np.ndarray, int]:
    starts = window_starts(len(sequence), window_size, overlap)

    residue_sums = np.zeros(
        (len(sequence), embedding_dim),
        dtype=np.float32,
    )
    residue_counts = np.zeros(len(sequence), dtype=np.uint16)

    for start in starts:
        end = min(start + window_size, len(sequence))
        segment = sequence[start:end]

        model_inputs, attention_mask, special_tokens_mask = tokenize(
            tokenizer, [segment], device
        )

        with torch.inference_mode():
            output = model(**model_inputs, return_dict=True)

        final_hidden = get_final_hidden(output)
        residue_mask = (
            attention_mask & ~special_tokens_mask
        ).to(device)

        segment_hidden = final_hidden[0, residue_mask[0]].float().cpu().numpy()
        if segment_hidden.shape != (len(segment), embedding_dim):
            raise RuntimeError(
                "Long-window embedding shape mismatch: "
                f"observed={segment_hidden.shape}, "
                f"expected={(len(segment), embedding_dim)}"
            )

        residue_sums[start:end] += segment_hidden
        residue_counts[start:end] += 1

        del output, final_hidden, segment_hidden
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if np.any(residue_counts == 0):
        missing = int(np.sum(residue_counts == 0))
        raise RuntimeError(f"{missing} residues were not covered by any window.")

    residue_means = residue_sums / residue_counts[:, None]
    pooled = residue_means.mean(axis=0, dtype=np.float64).astype(np.float32)

    if not np.isfinite(pooled).all():
        raise FloatingPointError("NaN or Inf found in a long-sequence embedding.")

    return pooled, len(starts)


def write_rows_tsv(
    path: Path,
    records: list[dict[str, Any]],
    completed: np.ndarray,
    window_counts: np.ndarray,
) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    fields = [
        "embedding_row",
        "clean_id",
        "original_id",
        "split",
        "label",
        "cluster_id",
        "sequence_length",
        "completed",
        "number_of_windows",
    ]

    with tmp_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for index, record in enumerate(records):
            writer.writerow(
                {
                    "embedding_row": index,
                    "clean_id": record["clean_id"],
                    "original_id": record["original_id"],
                    "split": record["split"],
                    "label": record["label"],
                    "cluster_id": record["cluster_id"],
                    "sequence_length": record["length"],
                    "completed": int(completed[index]),
                    "number_of_windows": int(window_counts[index]),
                }
            )

    tmp_path.replace(path)


def append_error(
    path: Path,
    record: dict[str, Any],
    stage: str,
    error: BaseException,
) -> None:
    new_file = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        if new_file:
            writer.writerow(
                [
                    "timestamp",
                    "clean_id",
                    "sequence_length",
                    "stage",
                    "error_type",
                    "error_message",
                    "traceback",
                ]
            )
        writer.writerow(
            [
                datetime.now().isoformat(timespec="seconds"),
                record["clean_id"],
                record["length"],
                stage,
                type(error).__name__,
                str(error).replace("\t", " ").replace("\n", " "),
                traceback.format_exc().replace("\t", " ").replace("\n", " | "),
            ]
        )


def main() -> None:
    args = parse_args()

    manifest = args.manifest.expanduser().resolve()
    model_path = args.model_path.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if args.window_size < 1:
        raise ValueError("window-size must be positive.")
    if not 0 <= args.window_overlap < args.window_size:
        raise ValueError(
            "window-overlap must be >=0 and smaller than window-size."
        )
    if args.max_batch_size < 1 or args.max_batch_tokens < 1:
        raise ValueError("Batch parameters must be positive.")

    output_dir.mkdir(parents=True, exist_ok=True)

    output_files = {
        "embeddings": output_dir / "embeddings.npy",
        "completed": output_dir / "completed.npy",
        "window_counts": output_dir / "window_counts.npy",
        "rows": output_dir / "rows.tsv",
        "errors": output_dir / "errors.tsv",
        "run_config": output_dir / "run_config.json",
        "summary": output_dir / "summary.json",
    }

    if args.overwrite:
        for path in output_files.values():
            path.unlink(missing_ok=True)

    records = read_manifest(manifest, args.limit)
    digest = records_digest(records)

    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False.")

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    torch.set_float32_matmul_precision("high")

    print("=" * 78)
    print("ESMC-600M batch embedding extraction")
    print("=" * 78)
    print(f"Manifest:          {manifest}")
    print(f"Model path:        {model_path}")
    print(f"Output directory:  {output_dir}")
    print(f"Selected records:  {len(records)}")
    print(f"Window size:       {args.window_size}")
    print(f"Window overlap:    {args.window_overlap}")
    print(f"Max batch size:    {args.max_batch_size}")
    print(f"Max batch tokens:  {args.max_batch_tokens}")
    print(f"Device:            {device}")
    if device.type == "cuda":
        print(f"GPU:               {torch.cuda.get_device_name(device)}")

    config = AutoConfig.from_pretrained(
        model_path,
        local_files_only=True,
    )
    embedding_dim = int(getattr(config, "d_model"))

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True,
    )
    model = AutoModelForMaskedLM.from_pretrained(
        model_path,
        local_files_only=True,
        dtype=torch.bfloat16,
    )
    model = model.to(device).eval()

    current_config = {
        "manifest": str(manifest),
        "manifest_records_digest": digest,
        "selected_records": len(records),
        "model_path": str(model_path),
        "model_class": model.__class__.__name__,
        "tokenizer_class": tokenizer.__class__.__name__,
        "embedding_dimension": embedding_dim,
        "window_size": args.window_size,
        "window_overlap": args.window_overlap,
        "max_batch_size": args.max_batch_size,
        "max_batch_tokens": args.max_batch_tokens,
        "dtype": str(next(model.parameters()).dtype),
        "device": str(device),
        "created_or_resumed_at": datetime.now().isoformat(timespec="seconds"),
    }

    if output_files["run_config"].exists():
        previous = json.loads(
            output_files["run_config"].read_text(encoding="utf-8")
        )
        checks = [
            "manifest_records_digest",
            "selected_records",
            "embedding_dimension",
            "window_size",
            "window_overlap",
        ]
        mismatches = {
            key: (previous.get(key), current_config.get(key))
            for key in checks
            if previous.get(key) != current_config.get(key)
        }
        if mismatches:
            raise RuntimeError(
                "Existing output directory is incompatible with this run: "
                f"{mismatches}. Use a new output directory or --overwrite."
            )
    else:
        output_files["run_config"].write_text(
            json.dumps(current_config, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    n_records = len(records)

    if output_files["embeddings"].exists():
        embeddings = open_memmap(
            output_files["embeddings"],
            mode="r+",
        )
        completed = open_memmap(
            output_files["completed"],
            mode="r+",
        )
        window_counts = open_memmap(
            output_files["window_counts"],
            mode="r+",
        )
        if embeddings.shape != (n_records, embedding_dim):
            raise RuntimeError(
                f"Existing embeddings shape is {embeddings.shape}, "
                f"expected {(n_records, embedding_dim)}."
            )
        if completed.shape != (n_records,):
            raise RuntimeError("Existing completed.npy has an incompatible shape.")
        if window_counts.shape != (n_records,):
            raise RuntimeError(
                "Existing window_counts.npy has an incompatible shape."
            )
    else:
        embeddings = open_memmap(
            output_files["embeddings"],
            mode="w+",
            dtype=np.float32,
            shape=(n_records, embedding_dim),
        )
        embeddings[:] = np.nan
        completed = open_memmap(
            output_files["completed"],
            mode="w+",
            dtype=np.bool_,
            shape=(n_records,),
        )
        completed[:] = False
        window_counts = open_memmap(
            output_files["window_counts"],
            mode="w+",
            dtype=np.uint16,
            shape=(n_records,),
        )
        window_counts[:] = 0
        embeddings.flush()
        completed.flush()
        window_counts.flush()

    def checkpoint() -> None:
        embeddings.flush()
        completed.flush()
        window_counts.flush()
        write_rows_tsv(
            output_files["rows"],
            records,
            np.asarray(completed),
            np.asarray(window_counts),
        )

    pending = [i for i in range(n_records) if not bool(completed[i])]
    short_indices = [
        i for i in pending
        if records[i]["length"] <= args.window_size
    ]
    long_indices = [
        i for i in pending
        if records[i]["length"] > args.window_size
    ]

    print(f"Already completed: {int(np.sum(completed))}")
    print(f"Pending short:     {len(short_indices)}")
    print(f"Pending long:      {len(long_indices)}")

    short_batches = make_short_batches(
        short_indices,
        records,
        args.max_batch_size,
        args.max_batch_tokens,
    )

    newly_completed = 0
    failed_this_run = 0
    start_time = time.time()

    def mark_progress() -> None:
        nonlocal newly_completed
        if (
            newly_completed > 0
            and newly_completed % args.checkpoint_every == 0
        ):
            checkpoint()
            elapsed = max(time.time() - start_time, 1e-9)
            print(
                f"[checkpoint] newly completed={newly_completed}, "
                f"total completed={int(np.sum(completed))}/{n_records}, "
                f"rate={newly_completed / elapsed:.2f} proteins/s"
            )

    def process_short_indices(indices: list[int]) -> None:
        nonlocal newly_completed, failed_this_run
        if not indices:
            return

        try:
            sequences = [records[i]["sequence"] for i in indices]
            array = extract_short_batch(
                model, tokenizer, sequences, device
            )
            for row_index, vector in zip(indices, array, strict=True):
                embeddings[row_index] = vector
                completed[row_index] = True
                window_counts[row_index] = 1
                newly_completed += 1
                mark_progress()

        except torch.cuda.OutOfMemoryError as error:
            if device.type == "cuda":
                torch.cuda.empty_cache()
            if len(indices) == 1:
                append_error(
                    output_files["errors"],
                    records[indices[0]],
                    "short_sequence_oom",
                    error,
                )
                failed_this_run += 1
                print(
                    f"[FAILED OOM] {records[indices[0]]['clean_id']} "
                    f"length={records[indices[0]]['length']}"
                )
            else:
                midpoint = len(indices) // 2
                process_short_indices(indices[:midpoint])
                process_short_indices(indices[midpoint:])

        except Exception as error:
            if len(indices) == 1:
                append_error(
                    output_files["errors"],
                    records[indices[0]],
                    "short_sequence",
                    error,
                )
                failed_this_run += 1
                print(
                    f"[FAILED] {records[indices[0]]['clean_id']}: {error}"
                )
            else:
                # Isolate a problematic record without losing the whole batch.
                midpoint = len(indices) // 2
                process_short_indices(indices[:midpoint])
                process_short_indices(indices[midpoint:])

    for batch_number, batch_indices in enumerate(short_batches, start=1):
        process_short_indices(batch_indices)
        if batch_number == 1 or batch_number % 25 == 0:
            max_length = max(records[i]["length"] for i in batch_indices)
            print(
                f"[short batch {batch_number}/{len(short_batches)}] "
                f"size={len(batch_indices)}, max_length={max_length}, "
                f"completed={int(np.sum(completed))}/{n_records}"
            )

    for long_number, index in enumerate(long_indices, start=1):
        record = records[index]
        try:
            vector, n_windows = extract_long_sequence(
                model=model,
                tokenizer=tokenizer,
                sequence=record["sequence"],
                embedding_dim=embedding_dim,
                window_size=args.window_size,
                overlap=args.window_overlap,
                device=device,
            )
            embeddings[index] = vector
            completed[index] = True
            window_counts[index] = n_windows
            newly_completed += 1
            mark_progress()
            print(
                f"[long {long_number}/{len(long_indices)}] "
                f"{record['clean_id']} length={record['length']} "
                f"windows={n_windows}"
            )

        except Exception as error:
            append_error(
                output_files["errors"],
                record,
                "long_sequence",
                error,
            )
            failed_this_run += 1
            print(f"[FAILED long] {record['clean_id']}: {error}")
            if device.type == "cuda":
                torch.cuda.empty_cache()

    checkpoint()

    completed_count = int(np.sum(completed))
    finite_completed = bool(
        np.isfinite(np.asarray(embeddings)[np.asarray(completed)]).all()
    )
    elapsed_seconds = time.time() - start_time

    summary = {
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "selected_records": n_records,
        "completed_records": completed_count,
        "remaining_records": n_records - completed_count,
        "newly_completed_this_run": newly_completed,
        "failed_this_run": failed_this_run,
        "all_completed_embeddings_finite": finite_completed,
        "embedding_shape": list(embeddings.shape),
        "embedding_dtype": str(embeddings.dtype),
        "elapsed_seconds_this_run": round(elapsed_seconds, 3),
        "proteins_per_second_this_run": round(
            newly_completed / max(elapsed_seconds, 1e-9),
            4,
        ),
        "long_sequences_selected": sum(
            record["length"] > args.window_size for record in records
        ),
        "max_sequence_length": max(record["length"] for record in records),
        "peak_gpu_memory_gib": (
            round(
                torch.cuda.max_memory_allocated(device) / (1024 ** 3),
                3,
            )
            if device.type == "cuda"
            else None
        ),
    }

    output_files["summary"].write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("=" * 78)
    print("Extraction run finished")
    print("=" * 78)
    for key, value in summary.items():
        print(f"{key}: {value}")
    print(f"Embeddings: {output_files['embeddings']}")
    print(f"Row metadata: {output_files['rows']}")
    print(f"Summary: {output_files['summary']}")
    if output_files["errors"].exists():
        print(f"Errors: {output_files['errors']}")


if __name__ == "__main__":
    main()
