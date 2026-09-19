"""Compatibility adapter to the original extractor using the public Biohub esm backend."""

from pathlib import Path
import hashlib
import json

import numpy as np
import torch
from esm.models.esmc import EsmcForMaskedLM, EsmcTokenizer

from historical_esmc_extractor import (
    extract_long_sequence,
    extract_short_batch,
)


class ESMCEmbedder:
    def __init__(
        self,
        device_name,
        dtype_name,
        allow_model_download,
        cache_dir,
        model_dir,
    ):
        if allow_model_download or cache_dir is not None:
            raise ValueError(
                "This verification uses only the exact historical local model directory."
            )

        if dtype_name not in ("auto", "bfloat16"):
            raise ValueError(
                "Historical inference requires bfloat16 weights."
            )

        self.device = torch.device(
            "cuda:0" if device_name == "auto" else device_name
        )

        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError(
                "Historical GPU execution required; no automatic fallback."
            )

        self.model_path = Path(model_dir).expanduser().resolve()

        if not self.model_path.is_dir():
            raise FileNotFoundError(self.model_path)

        expected = json.loads(
            Path(__file__)
            .with_name("HISTORICAL_MODEL_SHA256.json")
            .read_text()
        )

        for name, digest in expected.items():
            p = self.model_path / name

            if not p.is_file():
                raise FileNotFoundError(p)

            h = hashlib.sha256()

            with p.open("rb") as f:
                for chunk in iter(
                    lambda: f.read(8 * 1024 * 1024),
                    b"",
                ):
                    h.update(chunk)

            if h.hexdigest() != digest:
                raise RuntimeError(
                    "Historical ESMC hash mismatch: " + name
                )

        config = json.loads(
            (self.model_path / "config.json").read_text()
        )

        d_model = config.get(
            "d_model",
            config.get("hidden_size"),
        )

        if int(d_model) != 1152:
            raise ValueError(
                "Historical model dimension mismatch"
            )

        self.tokenizer = EsmcTokenizer()

        self.model = EsmcForMaskedLM.from_pretrained(
            str(self.model_path),
            device=self.device,
        ).eval()

        self.model = self.model.to(torch.bfloat16)

        print(
            "Historical backend:",
            type(self.model).__module__,
            type(self.model).__name__,
            "parameter dtype:",
            next(self.model.parameters()).dtype,
            flush=True,
        )

    def embed_batch(self, records):
        records = list(records)

        if len(records) != 1:
            raise ValueError(
                "This verification preserves historical batch size 1."
            )

        seq = records[0].sequence

        if len(seq) <= 2046:
            x = extract_short_batch(
                self.model,
                self.tokenizer,
                [seq],
                self.device,
            )
            n = 1
        else:
            row, n = extract_long_sequence(
                self.model,
                self.tokenizer,
                seq,
                1152,
                2046,
                256,
                self.device,
            )
            x = row[None, :]

        if x.shape != (1, 1152) or not np.isfinite(x).all():
            raise ValueError(
                "Invalid historical representation"
            )

        return x, np.array([n], dtype=np.uint16)
