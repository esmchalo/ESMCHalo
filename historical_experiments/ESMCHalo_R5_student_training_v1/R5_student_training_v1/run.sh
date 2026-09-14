#!/usr/bin/env bash
set -euo pipefail
R5_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
R5_PY=/home/lvfang/miniconda3/envs/esmc_latest/bin/python
if ! env -u PYTHONPATH PYTHONNOUSERSITE=1 "$R5_PY" -s -u "$R5_DIR/test_runtime.py" > "$R5_DIR/runtime_tests.log" 2>&1; then
  cat "$R5_DIR/runtime_tests.log"
  exit 1
fi
printf '%s\n' 'RUNTIME TESTS PASSED; starting R5'
exec env -u PYTHONPATH PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
 "$R5_PY" -s -u "$R5_DIR/run.py" --root /home/lvfang/ESMC_halophile \
 --output /home/lvfang/ESMC_halophile/R5_student_training_v1_output "$@"
