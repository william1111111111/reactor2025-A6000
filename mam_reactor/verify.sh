#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${REACT_PYTHON:-$ROOT_DIR/../.venv/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="$(command -v python)"
fi
export MAM_REACTOR_ROOT="$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export MPLCONFIGDIR="${MAM_REACTOR_MPLCONFIGDIR:-$ROOT_DIR/.mplconfig}"
cd "$ROOT_DIR"

"$PYTHON_BIN" -m py_compile regnn/*.py tools/compute_frd_resumable.py
"$PYTHON_BIN" - <<'PY'
from pathlib import Path
import torch

root = Path.cwd()
checkpoints = sorted((root / "checkpoints").glob("*.pth"))
for path in checkpoints:
    payload = torch.load(path, map_location="cpu")
    method = payload.get("method", payload.get("model_type", "unknown"))
    print(f"OK checkpoint={path.name} method={method}")
print(f"OK checkpoint_count={len(checkpoints)}")
PY
echo "Mam-Reactor archive verification passed."
