#!/usr/bin/env bash
set -euo pipefail
R6_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
R6_PY=/home/lvfang/miniconda3/envs/esmc_latest/bin/python
if ! env -u PYTHONPATH PYTHONNOUSERSITE=1 "$R6_PY" -s -u "$R6_DIR/test_runtime.py" > "$R6_DIR/runtime_tests.log" 2>&1; then
  cat "$R6_DIR/runtime_tests.log"
  exit 1
fi
printf '%s\n' 'RUNTIME TESTS PASSED; starting R6'
exec env -u PYTHONPATH PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
 "$R6_PY" -s -u "$R6_DIR/run.py" --root /home/lvfang/ESMC_halophile \
 --output /home/lvfang/ESMC_halophile/R6_supervision_controls_v1_output "$@"
