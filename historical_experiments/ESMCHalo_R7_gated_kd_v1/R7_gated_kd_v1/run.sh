#!/usr/bin/env bash
set -euo pipefail
R7_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
R7_PY=/home/lvfang/miniconda3/envs/esmc_latest/bin/python
if ! env -u PYTHONPATH PYTHONNOUSERSITE=1 "$R7_PY" -s -u "$R7_DIR/test_runtime.py" > "$R7_DIR/runtime_tests.log" 2>&1; then
  cat "$R7_DIR/runtime_tests.log"
  exit 1
fi
printf '%s\n' 'RUNTIME TESTS PASSED; starting R7'
exec env -u PYTHONPATH PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
 "$R7_PY" -s -u "$R7_DIR/run.py" --root /home/lvfang/ESMC_halophile \
 --output /home/lvfang/ESMC_halophile/R7_gated_kd_v1_output "$@"
