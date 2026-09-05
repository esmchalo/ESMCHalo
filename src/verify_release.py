#!/usr/bin/env python3
from pathlib import Path
import hashlib
import sys

root = Path(__file__).resolve().parents[1]
manifest = root / "MANIFEST.sha256"
failures = []
for line in manifest.read_text(encoding="utf-8").splitlines():
    expected, relative = line.split("  ", 1)
    path = root / relative
    if not path.is_file():
        failures.append(f"MISSING {relative}")
        continue
    observed = hashlib.sha256(path.read_bytes()).hexdigest()
    if observed != expected:
        failures.append(f"HASH_MISMATCH {relative}")
if failures:
    print("\n".join(failures), file=sys.stderr)
    raise SystemExit(1)
print("RELEASE_MANIFEST_PASS")
