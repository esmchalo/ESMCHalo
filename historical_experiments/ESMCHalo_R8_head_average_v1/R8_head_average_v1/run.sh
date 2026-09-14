#!/usr/bin/env bash
set -euo pipefail
R8_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
R8_PY=/home/lvfang/miniconda3/envs/esmc_latest/bin/python
if ! env -u PYTHONPATH PYTHONNOUSERSITE=1 "$R8_PY" -s -u "$R8_DIR/test_runtime.py" > "$R8_DIR/runtime_tests.log" 2>&1; then
  cat "$R8_DIR/runtime_tests.log"
  exit 1
fi
printf '%s\n' 'RUNTIME TESTS PASSED; starting R8'
exec env -u PYTHONPATH PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
 "$R8_PY" -s -u "$R8_DIR/run.py" --root /home/lvfang/ESMC_halophile \
 --output /home/lvfang/ESMC_halophile/R8_head_average_v1_output "$@"
