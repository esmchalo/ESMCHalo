#!/usr/bin/env bash
set -euo pipefail
R9_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
R9_PY=/home/lvfang/miniconda3/envs/esmc_latest/bin/python
exec env -u PYTHONPATH PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 "$R9_PY" -s -u "$R9_DIR/run.py" --root /home/lvfang/ESMC_halophile --output /home/lvfang/ESMC_halophile/R9_fixed_score_ensemble_v1_output "$@"
