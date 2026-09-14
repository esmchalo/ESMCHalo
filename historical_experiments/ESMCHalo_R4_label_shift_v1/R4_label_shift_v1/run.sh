#!/usr/bin/env bash
set -euo pipefail
R4_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec env -u PYTHONPATH PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
 /home/lvfang/miniconda3/envs/esmc_latest/bin/python -s -u "$R4_DIR/run.py" \
 --root /home/lvfang/ESMC_halophile --output /home/lvfang/ESMC_halophile/R4_label_shift_v1_output
