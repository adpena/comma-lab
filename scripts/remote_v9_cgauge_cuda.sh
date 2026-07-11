#!/usr/bin/env bash
# Provider-neutral remote driver for the V9 CGauge Torch/CUDA backend.
set -euo pipefail

export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export DALI_DISABLE_NVML="${DALI_DISABLE_NVML:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"

WORKSPACE="${WORKSPACE:-/tmp/pact}"
PYBIN="${PYBIN:-python3}"
MODE="${WITNESS_TRAINER_MODE:-}"
GT_CACHE="${WITNESS_GT_CACHE:-}"
GT_CACHE_SHA256="${WITNESS_GT_CACHE_SHA256:-}"
OUT_DIR="${WITNESS_OUT_DIR:-}"
EPOCHS="${WITNESS_EPOCHS:-3000}"
NUM_PAIRS="${WITNESS_NUM_PAIRS:-600}"
RESUME_FROM="${WITNESS_RESUME_FROM:-}"
MIN_FREE_GB="${WITNESS_MIN_FREE_GB:-20}"

if [ "$MODE" != "full" ]; then
  echo "REFUSED: WITNESS_TRAINER_MODE must explicitly equal full" >&2
  exit 64
fi
if [ -z "$GT_CACHE" ] || [ ! -f "$GT_CACHE" ]; then
  echo "REFUSED: staged WITNESS_GT_CACHE is absent: $GT_CACHE" >&2
  exit 65
fi
if [[ ! "$GT_CACHE_SHA256" =~ ^[0-9a-fA-F]{64}$ ]]; then
  echo "REFUSED: WITNESS_GT_CACHE_SHA256 must carry exact source-byte custody" >&2
  exit 69
fi
if [ -z "$OUT_DIR" ] || [[ "$OUT_DIR" != /modal_results/* ]]; then
  echo "REFUSED: WITNESS_OUT_DIR must be durable /modal_results custody" >&2
  exit 66
fi

cd "$WORKSPACE"
mkdir -p "$OUT_DIR"
"$PYBIN" - "$OUT_DIR" "$MIN_FREE_GB" <<'PY'
import shutil, sys
out, minimum = sys.argv[1], float(sys.argv[2])
free = shutil.disk_usage(out).free / 2**30
print(f"storage_preflight free_gib={free:.3f} required_gib={minimum:.3f}")
if free < minimum:
    raise SystemExit(67)
PY
"$PYBIN" - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit(68)
print("device", torch.cuda.get_device_name(0), "capability", torch.cuda.get_device_capability(0))
PY
"$PYBIN" - "$GT_CACHE" "$GT_CACHE_SHA256" "$OUT_DIR" <<'PY'
import hashlib, json, os, pathlib, sys
source, expected, out = pathlib.Path(sys.argv[1]), sys.argv[2].lower(), pathlib.Path(sys.argv[3])
h = hashlib.sha256()
with source.open("rb") as fh:
    for chunk in iter(lambda: fh.read(8 * 1024 * 1024), b""):
        h.update(chunk)
actual = h.hexdigest()
if actual != expected:
    raise SystemExit(f"REFUSED: staged GT cache SHA-256 mismatch {actual} != {expected}")
payload = {
    "schema": "witness_remote_asset_custody.v1",
    "path": str(source),
    "bytes": source.stat().st_size,
    "sha256": actual,
    "driver_mode": os.environ.get("WITNESS_TRAINER_MODE"),
    "nvml_disabled": os.environ.get("DALI_DISABLE_NVML"),
}
target = out / "remote_asset_custody.json"
tmp = target.with_suffix(".json.tmp")
tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
os.replace(tmp, target)
print(json.dumps(payload, sort_keys=True))
PY

ARGS=(
  --gt-cache "$GT_CACHE"
  --num-pairs "$NUM_PAIRS"
  --epochs "$EPOCHS"
  --out-dir "$OUT_DIR"
  --device cuda
  --compile-probe
)
if [ -n "$RESUME_FROM" ]; then
  ARGS+=(--resume-from "$RESUME_FROM")
fi
exec "$PYBIN" experiments/train_levelset_witness_realized_through_R_torch.py "${ARGS[@]}"
