#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${REACT_PYTHON:-$ROOT_DIR/../.venv/bin/python}"
RESULTS="${1:-${MAM_REACTOR_RESULTS_PT:-}}"
OUTPUT_DIR="${2:-${MAM_REACTOR_FRD_OUTPUT_DIR:-}}"

if [[ -z "$RESULTS" || -z "$OUTPUT_DIR" ]]; then
  echo "Usage: $0 RESULTS_Frd.pt OUTPUT_DIR" >&2
  exit 2
fi
[[ -f "$RESULTS" ]] || { echo "Results file not found: $RESULTS" >&2; exit 2; }
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="$(command -v python)"
fi
mkdir -p "$OUTPUT_DIR"
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"
exec "$PYTHON_BIN" "$ROOT_DIR/tools/compute_frd_resumable.py" \
  --results "$RESULTS" \
  --output-dir "$OUTPUT_DIR" \
  --workers "${MAM_REACTOR_FRD_WORKERS:-16}"
