#!/usr/bin/env python3
"""Run the frozen ESMCHalo final-v3 model on protein FASTA input."""

from __future__ import annotations

import argparse
import csv
import os
import platform
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np

from esmchalo_contract import (
    ALLOWED_AMINO_ACIDS,
    EMBEDDING_LAYER,
    EMBEDDING_WIDTH,
    ESMC_REPOSITORY,
    ESMC_REVISION,
    FIXED_THRESHOLD,
    LONG_SEQUENCE_OVERLAP,
    LONG_SEQUENCE_STRIDE,
    LONG_SEQUENCE_WINDOW_SIZE,
    MODEL_FILENAME,
    MODEL_SHA256,
    SCOPE_WARNING,
    load_locked_artifact,
    model_contract,
    score_embeddings,
    sequence_sha256,
    write_json_atomic,
)


@dataclass(frozen=True)
class FastaRecord:
    record_id: str
    description: str
    sequence: str


def window_starts(
    length: int,
    window_size: int = LONG_SEQUENCE_WINDOW_SIZE,
    overlap: int = LONG_SEQUENCE_OVERLAP,
) -> list[int]:
    """Return the exact starts used by the locked historical extractor.

    The final window is always aligned to the sequence end. Consequently, its
    overlap with the preceding regular window can exceed the nominal overlap.
    """
    if length < 1:
        raise ValueError("Sequence length must be positive")
    if window_size < 1:
        raise ValueError("Window size must be positive")
    if overlap < 0 or overlap >= window_size:
        raise ValueError("Window overlap must be >= 0 and smaller than window size")
    if length <= window_size:
        return [0]
    step = window_size - overlap
    starts = list(range(0, length - window_size + 1, step))
    final_start = length - window_size
    if starts[-1] != final_start:
        starts.append(final_start)
    return starts


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_fasta(path: Path) -> list[FastaRecord]:
    records: list[FastaRecord] = []
    header: str | None = None
    sequence_parts: list[str] = []

    def commit() -> None:
        nonlocal header, sequence_parts
        if header is None:
            return
        sequence = "".join(sequence_parts).replace(" ", "").upper()
        if not sequence:
            raise ValueError(f"Empty sequence for FASTA header: {header}")
        record_id = header.split()[0]
        invalid = sorted(set(sequence) - ALLOWED_AMINO_ACIDS)
        if invalid:
            raise ValueError(
                f"Unsupported residue symbols for {record_id}: {''.join(invalid)}. "
                "Terminal stops and alignment gaps must be removed before inference."
            )
        records.append(FastaRecord(record_id, header, sequence))
        header = None
        sequence_parts = []

    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                commit()
                header = line[1:].strip()
                if not header:
                    raise ValueError(f"Blank FASTA header at line {line_number}")
            else:
                if header is None:
                    raise ValueError(
                        f"Sequence content before the first FASTA header at line {line_number}"
                    )
                sequence_parts.append("".join(line.split()))
    commit()
    if not records:
        raise ValueError(f"No FASTA records found in {path}")
    id_counts = Counter(record.record_id for record in records)
    duplicate_ids = sorted(value for value, count in id_counts.items() if count > 1)
    if duplicate_ids:
        raise ValueError(f"Duplicate FASTA identifiers are not allowed: {duplicate_ids[:10]}")
    return records


class ESMCEmbedder:
    def __init__(
        self,
        device_name: str,
        dtype_name: str,
        allow_model_download: bool,
        cache_dir: Path | None,
    ) -> None:
        import torch
        from transformers import AutoModelForMaskedLM, AutoTokenizer

        self.torch = torch
        if device_name == "auto":
            device_name = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device_name)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(f"CUDA device requested but CUDA is unavailable: {device_name}")
        if dtype_name == "auto":
            dtype_name = "bfloat16" if self.device.type == "cuda" else "float32"
        if dtype_name not in {"float16", "bfloat16", "float32"}:
            raise ValueError("dtype must be auto, float16, bfloat16 or float32")
        if self.device.type == "cpu" and dtype_name != "float32":
            raise ValueError("CPU inference is restricted to float32 in this release")
        self.dtype_name = dtype_name
        self.autocast_dtype = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }[dtype_name]
        common: dict[str, object] = {
            "revision": ESMC_REVISION,
            "trust_remote_code": True,
            "local_files_only": not allow_model_download,
        }
        if cache_dir is not None:
            common["cache_dir"] = str(cache_dir)
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(ESMC_REPOSITORY, **common)
            self.model = AutoModelForMaskedLM.from_pretrained(ESMC_REPOSITORY, **common)
        except Exception as error:
            mode = "download allowed" if allow_model_download else "local cache only"
            raise RuntimeError(
                f"Unable to load locked ESMC backend ({mode}) at revision {ESMC_REVISION}. "
                "If the exact checkpoint is not cached, rerun with --allow-model-download. "
                f"Original error: {error}"
            ) from error
        self.model.to(self.device)
        self.model.eval()
        hidden_size = int(
            getattr(
                self.model.config,
                "d_model",
                getattr(self.model.config, "hidden_size", -1),
            )
        )
        if hidden_size != EMBEDDING_WIDTH:
            raise ValueError(f"ESMC hidden width mismatch: {hidden_size} != {EMBEDDING_WIDTH}")
        self.detected_max_residues = self._detect_max_residues()
        if (
            self.detected_max_residues is not None
            and self.detected_max_residues < LONG_SEQUENCE_WINDOW_SIZE
        ):
            raise ValueError(
                "Pinned ESMC backend cannot satisfy the locked 2,046-residue "
                f"window contract: detected maximum={self.detected_max_residues}"
            )

    def _detect_max_residues(self) -> int | None:
        candidates: list[int] = []
        for name in [
            "max_position_embeddings",
            "max_seq_len",
            "max_sequence_length",
            "n_positions",
        ]:
            value = getattr(self.model.config, name, None)
            if isinstance(value, int) and 0 < value < 1_000_000:
                candidates.append(value)
        tokenizer_limit = getattr(self.tokenizer, "model_max_length", None)
        if isinstance(tokenizer_limit, int) and 0 < tokenizer_limit < 1_000_000:
            candidates.append(tokenizer_limit)
        if not candidates:
            return None
        special_tokens = int(self.tokenizer.num_special_tokens_to_add(pair=False))
        return max(1, min(candidates) - special_tokens)

    def _extract_residue_states(self, sequences: list[str]) -> list[np.ndarray]:
        """Extract one locked layer-36 state per non-special residue token."""
        if not sequences:
            return []
        too_long = [len(sequence) for sequence in sequences if len(sequence) > LONG_SEQUENCE_WINDOW_SIZE]
        if too_long:
            raise ValueError(
                "Internal error: a model call exceeded the locked 2,046-residue "
                f"window size: {too_long[:5]}"
            )
        encoded = self.tokenizer(
            sequences,
            return_tensors="pt",
            padding=True,
            add_special_tokens=True,
            return_special_tokens_mask=True,
        )
        special = encoded.pop("special_tokens_mask").to(bool)
        input_ids = encoded["input_ids"].to(self.device)
        attention = encoded["attention_mask"].to(self.device)
        valid_residues = (~special.to(self.device)) & attention.to(bool)
        observed_counts = valid_residues.sum(dim=1).detach().cpu().tolist()
        expected_counts = [len(sequence) for sequence in sequences]
        if observed_counts != expected_counts:
            raise ValueError(
                "Tokenizer/residue alignment failed: "
                f"observed {observed_counts}; expected {expected_counts}"
            )
        use_autocast = self.device.type == "cuda" and self.dtype_name != "float32"
        context = self.torch.autocast(
            device_type="cuda",
            dtype=self.autocast_dtype,
            enabled=use_autocast,
        )
        with self.torch.inference_mode(), context:
            output = self.model(
                input_ids=input_ids,
                attention_mask=attention,
                output_hidden_states=True,
                return_dict=True,
            )
        hidden_states = output.hidden_states
        if hidden_states is None or len(hidden_states) <= EMBEDDING_LAYER:
            count = len(hidden_states) if hidden_states is not None else 0
            raise ValueError(
                f"Locked layer {EMBEDDING_LAYER} is unavailable; hidden-state count={count}"
            )
        hidden = hidden_states[EMBEDDING_LAYER]
        if hidden.shape[-1] != EMBEDDING_WIDTH:
            raise ValueError(f"Layer embedding width mismatch: {tuple(hidden.shape)}")
        residue_states: list[np.ndarray] = []
        for row_index in range(len(sequences)):
            residue_hidden = hidden[row_index, valid_residues[row_index], :]
            residue_states.append(residue_hidden.float().detach().cpu().numpy())
        return residue_states

    def _embed_short_batch(self, records: list[FastaRecord]) -> np.ndarray:
        states = self._extract_residue_states([record.sequence for record in records])
        return np.stack(
            [state.mean(axis=0, dtype=np.float64) for state in states], axis=0
        )

    def _embed_long_record(self, record: FastaRecord) -> tuple[np.ndarray, int]:
        sequence = record.sequence
        starts = window_starts(len(sequence))
        residue_sums = np.zeros((len(sequence), EMBEDDING_WIDTH), dtype=np.float32)
        residue_counts = np.zeros(len(sequence), dtype=np.uint16)
        for start in starts:
            end = min(start + LONG_SEQUENCE_WINDOW_SIZE, len(sequence))
            segment = sequence[start:end]
            segment_hidden = self._extract_residue_states([segment])[0]
            expected_shape = (len(segment), EMBEDDING_WIDTH)
            if segment_hidden.shape != expected_shape:
                raise ValueError(
                    "Long-window embedding shape mismatch for "
                    f"{record.record_id}: {segment_hidden.shape} != {expected_shape}"
                )
            residue_sums[start:end] += segment_hidden
            residue_counts[start:end] += 1
            if self.device.type == "cuda":
                self.torch.cuda.empty_cache()
        if np.any(residue_counts == 0):
            missing = int(np.sum(residue_counts == 0))
            raise ValueError(
                f"Long-sequence coverage failure for {record.record_id}: "
                f"{missing} residues were not encoded"
            )
        residue_means = residue_sums / residue_counts[:, None]
        pooled = residue_means.mean(axis=0, dtype=np.float64).astype(np.float32)
        if not np.isfinite(pooled).all():
            raise FloatingPointError(
                f"Non-finite long-sequence embedding for {record.record_id}"
            )
        return pooled, len(starts)

    def embed_batch(self, records: Iterable[FastaRecord]) -> tuple[np.ndarray, np.ndarray]:
        """Embed a mixed batch while preserving input order and window counts."""
        batch = list(records)
        if not batch:
            raise ValueError("Cannot embed an empty batch")
        embeddings = np.empty((len(batch), EMBEDDING_WIDTH), dtype=np.float64)
        number_of_windows = np.ones(len(batch), dtype=np.uint16)
        short_indices = [
            index
            for index, record in enumerate(batch)
            if len(record.sequence) <= LONG_SEQUENCE_WINDOW_SIZE
        ]
        if short_indices:
            short_records = [batch[index] for index in short_indices]
            short_embeddings = self._embed_short_batch(short_records)
            embeddings[short_indices] = short_embeddings
        for index, record in enumerate(batch):
            if len(record.sequence) <= LONG_SEQUENCE_WINDOW_SIZE:
                continue
            pooled, count = self._embed_long_record(record)
            embeddings[index] = pooled
            number_of_windows[index] = count
        return embeddings, number_of_windows

    @property
    def device_description(self) -> str:
        if self.device.type != "cuda":
            return str(self.device)
        return f"{self.device}:{self.torch.cuda.get_device_name(self.device)}"


def write_predictions(
    output_path: Path,
    records: list[FastaRecord],
    scores: dict[str, np.ndarray],
    embedder: ESMCEmbedder,
    number_of_windows: np.ndarray,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    order = np.argsort(-scores["raw_logit"], kind="stable")
    ranks = np.empty(len(records), dtype=np.int64)
    ranks[order] = np.arange(1, len(records) + 1)
    fieldnames = [
        "input_order",
        "rank_within_input",
        "record_id",
        "description",
        "length",
        "number_of_windows",
        "sequence_sha256",
        "raw_logit_ranking_score",
        "raw_probability",
        "calibrated_probability",
        "fixed_threshold",
        "prediction",
        "prediction_label",
        "model_sha256",
        "esmc_revision",
        "embedding_layer",
        "embedding_width",
    ]
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for index, record in enumerate(records):
            prediction = int(scores["prediction"][index])
            writer.writerow(
                {
                    "input_order": index + 1,
                    "rank_within_input": int(ranks[index]),
                    "record_id": record.record_id,
                    "description": record.description.replace("\t", " "),
                    "length": len(record.sequence),
                    "number_of_windows": int(number_of_windows[index]),
                    "sequence_sha256": sequence_sha256(record.sequence),
                    "raw_logit_ranking_score": f"{scores['raw_logit'][index]:.12g}",
                    "raw_probability": f"{scores['raw_probability'][index]:.12g}",
                    "calibrated_probability": f"{scores['calibrated_probability'][index]:.12g}",
                    "fixed_threshold": f"{FIXED_THRESHOLD:.2f}",
                    "prediction": prediction,
                    "prediction_label": "halophilic_like" if prediction else "non_halophilic_like",
                    "model_sha256": MODEL_SHA256,
                    "esmc_revision": ESMC_REVISION,
                    "embedding_layer": EMBEDDING_LAYER,
                    "embedding_width": EMBEDDING_WIDTH,
                }
            )
    temporary.replace(output_path)


def run_self_test(model_path: Path) -> None:
    artifact = load_locked_artifact(model_path)
    synthetic = np.vstack(
        [
            np.zeros(EMBEDDING_WIDTH, dtype=np.float64),
            np.sin(np.arange(EMBEDDING_WIDTH, dtype=np.float64) / 17.0),
            np.cos(np.arange(EMBEDDING_WIDTH, dtype=np.float64) / 31.0) * 0.25,
        ]
    )
    expected = np.asarray(
        [0.3844607056538503, 3.041044098104965e-27, 0.9996322755877517]
    )
    observed = score_embeddings(synthetic, artifact)["calibrated_probability"]
    if not np.allclose(observed, expected, atol=1e-12, rtol=1e-11):
        raise AssertionError(f"Static scoring mismatch: observed={observed}; expected={expected}")
    print("STATIC_MODEL_CONTRACT_TEST_PASS")


def build_parser() -> argparse.ArgumentParser:
    package_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Frozen ESMCHalo final-v3 FASTA inference (no training or tuning)."
    )
    parser.add_argument("--input", type=Path, help="Input protein FASTA")
    parser.add_argument("--output", type=Path, help="Output TSV")
    parser.add_argument(
        "--model",
        type=Path,
        default=package_root / "artifacts" / MODEL_FILENAME,
        help="Locked final-v3 joblib artifact",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda:0, ...")
    parser.add_argument(
        "--dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"]
    )
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--allow-model-download", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.self_test:
        run_self_test(args.model)
        return
    if args.input is None or args.output is None:
        raise SystemExit("--input and --output are required unless --self-test is used")
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be at least 1")
    if not args.input.is_file():
        raise SystemExit(f"Input FASTA is absent: {args.input}")
    if args.output.exists() and not args.overwrite:
        raise SystemExit(f"Output already exists (use --overwrite to replace): {args.output}")
    print(f"SCOPE WARNING: {SCOPE_WARNING}", file=sys.stderr)
    records = parse_fasta(args.input)
    artifact = load_locked_artifact(args.model)
    started = utc_now()
    embedder = ESMCEmbedder(
        device_name=args.device,
        dtype_name=args.dtype,
        allow_model_download=args.allow_model_download,
        cache_dir=args.cache_dir,
    )
    pooled_batches: list[np.ndarray] = []
    window_count_batches: list[np.ndarray] = []
    for offset in range(0, len(records), args.batch_size):
        batch = records[offset : offset + args.batch_size]
        pooled, window_counts = embedder.embed_batch(batch)
        pooled_batches.append(pooled)
        window_count_batches.append(window_counts)
        completed = min(offset + len(batch), len(records))
        print(f"Embedded {completed}/{len(records)} proteins", file=sys.stderr)
    embeddings = np.concatenate(pooled_batches, axis=0)
    number_of_windows = np.concatenate(window_count_batches, axis=0)
    scores = score_embeddings(embeddings, artifact)
    write_predictions(args.output, records, scores, embedder, number_of_windows)
    metadata = {
        "status": "PASS",
        "started_utc": started,
        "completed_utc": utc_now(),
        "input_fasta": str(args.input.resolve()),
        "input_records": len(records),
        "output_tsv": str(args.output.resolve()),
        "model_contract": model_contract(artifact),
        "inference": {
            "device": embedder.device_description,
            "dtype": embedder.dtype_name,
            "batch_size": args.batch_size,
            "detected_max_residues": embedder.detected_max_residues,
            "window_size_residues": LONG_SEQUENCE_WINDOW_SIZE,
            "nominal_overlap_residues": LONG_SEQUENCE_OVERLAP,
            "stride_residues": LONG_SEQUENCE_STRIDE,
            "long_sequence_records": int(np.sum(number_of_windows > 1)),
            "maximum_windows_per_record": int(number_of_windows.max()),
            "model_download_allowed": bool(args.allow_model_download),
            "python": sys.version.split()[0],
            "platform": platform.platform(),
        },
        "interpretation": {
            "primary_cross_domain_output": "raw_logit_ranking_score",
            "thresholded_output": "secondary and target-domain validation required",
            "scope_warning": SCOPE_WARNING,
        },
        "training_or_tuning_performed": False,
        "threshold_changed": False,
    }
    metadata_path = args.output.with_suffix(args.output.suffix + ".run.json")
    write_json_atomic(metadata_path, metadata)
    print(f"PREDICTION_PASS output={args.output}", file=sys.stderr)
    print(f"RUN_METADATA={metadata_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
