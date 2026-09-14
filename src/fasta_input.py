from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import re
from collections import Counter
ALLOWED_AMINO_ACIDS=set("ACDEFGHIKLMNPQRSTVWYBXZJUO")
@dataclass(frozen=True)
class FastaRecord:
    record_id: str
    description: str
    sequence: str

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
