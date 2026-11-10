#!/usr/bin/env bash
# Run one numbered analysis step with the project environment: bash run_step.sh 01_inventory [args...]
# Python runs isolated (-I) so nothing in the data folders or the current directory can shadow a module.
set -euo pipefail
cd "$(dirname "$0")"
if [ -x "env/py/bin/python" ]; then PY="env/py/bin/python"; else PY="${PYTHON:-python3}"; fi
export MPLCONFIGDIR="$PWD/env/cache/mpl"
export XDG_CACHE_HOME="$PWD/env/cache"
export TORCH_HOME="$PWD/env/weights/torch"
export HF_HOME="$PWD/env/weights/hf"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-1}"
export KMP_WARNINGS=0
mkdir -p logs "$MPLCONFIGDIR"
STEP="$1"; shift
echo "[$(date '+%F %T')] Starting ${STEP} $*"
set +e
"$PY" -I "code/${STEP}.py" "$@" > "logs/${STEP}.log" 2>&1
rc=$?
set -e
echo "[$(date '+%F %T')] Finished ${STEP} (exit ${rc}); log: logs/${STEP}.log"
exit $rc
