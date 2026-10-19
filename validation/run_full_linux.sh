#!/usr/bin/env bash
# Run from the repository root, after activating the new environment.
set -euo pipefail
test "$(uname -s)" = Linux
test -f data/release_manifest.json
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLBACKEND=Agg PYTHONDONTWRITEBYTECODE=1
VALIDATION_WORKERS="${VALIDATION_WORKERS:-4}"
resume=()
if [[ "${1:-}" == --resume ]]; then
  resume=(--resume)
elif [[ $# -ne 0 ]]; then
  echo 'Usage: bash validation/run_full_linux.sh [--resume]' >&2
  exit 2
fi
mkdir -p generated_validation/controller
trap 'rc=$?; printf "%s\n" "$rc" > generated_validation/controller/exit_code.txt' EXIT
date -Is > generated_validation/controller/started_at.txt
python -B validation/check_release.py
python -B validation/run_validation.py --mode saved --out generated_validation/saved "${resume[@]}"
python -B validation/run_validation.py --mode replay --out generated_validation/replay --workers "$VALIDATION_WORKERS" "${resume[@]}"
python -B validation/regenerate_inputs.py --out generated_validation/fields --workers "$VALIDATION_WORKERS" "${resume[@]}"
python -B validation/run_validation.py --mode replay --out generated_validation/from_fields --workers "$VALIDATION_WORKERS" \
  --generated-index generated_validation/fields/generated_index.json "${resume[@]}"
date -Is > generated_validation/controller/completed_at.txt
echo 'Numerical stages complete. Inspect all eight figure pairs and complete the acceptance report.'
