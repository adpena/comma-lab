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
SEGNET_SHA256="${WITNESS_SEGNET_SHA256:-}"
POSENET_SHA256="${WITNESS_POSENET_SHA256:-}"
OUT_DIR="${WITNESS_OUT_DIR:-}"
EPOCHS="${WITNESS_EPOCHS:-3000}"
STOP_AFTER_EPOCHS="${WITNESS_STOP_AFTER_EPOCHS:-}"
DSL_CONFIG="${WITNESS_DSL_CONFIG:-v9_cgauge_432_launch}"
TORCH_COMPILE_MODE="${WITNESS_TORCH_COMPILE_MODE:-}"
CHILD_TIMEOUT_SECONDS="${WITNESS_CHILD_TIMEOUT_SECONDS:-}"
NUM_PAIRS="${WITNESS_NUM_PAIRS:-600}"
RESUME_FROM="${WITNESS_RESUME_FROM:-}"
RESUME_SHA256="${WITNESS_RESUME_SHA256:-}"
MIN_FREE_GIB=20
PREFLIGHT_ONLY="${WITNESS_PREFLIGHT_ONLY:-0}"
OUT_DIR_WAS_PRESENT=0

require_positive_integer() {
  local name="$1" value="$2"
  if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
    echo "REFUSED: $name must be a positive integer; got '$value'" >&2
    exit 64
  fi
}

normalize_modal_results_path() {
  "$PYBIN" - "$1" "$2" <<'PY'
import pathlib, sys
name, raw = sys.argv[1], sys.argv[2]
if not raw:
    raise SystemExit(f"REFUSED: {name} is required")
if raw != raw.strip() or any(char.isspace() or ord(char) < 32 for char in raw):
    raise SystemExit(f"REFUSED: {name} contains whitespace or control characters")
if any(char in raw for char in ",="):
    raise SystemExit(f"REFUSED: {name} contains env-override delimiters")
path = pathlib.PurePosixPath(raw)
if not raw.startswith("/modal_results/") or ".." in path.parts or str(path) != raw:
    raise SystemExit(f"REFUSED: {name} must be normalized /modal_results/** custody")
print(raw)
PY
}

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
require_positive_integer WITNESS_EPOCHS "$EPOCHS"
# Empty WITNESS_STOP_AFTER_EPOCHS selects timeout-stop mode: the trainer runs
# toward the typed horizon and the 1500-second child timeout is the stop
# (operator 2026-07-15: a 3-warmup-epoch cutoff is a toy-regime measurement).
if [ -n "$STOP_AFTER_EPOCHS" ]; then
  require_positive_integer WITNESS_STOP_AFTER_EPOCHS "$STOP_AFTER_EPOCHS"
  if [ "$STOP_AFTER_EPOCHS" -gt 3 ]; then
    echo "REFUSED: WITNESS_STOP_AFTER_EPOCHS must be empty (timeout-stop) or within 1..3" >&2
    exit 66
  fi
fi
require_positive_integer WITNESS_NUM_PAIRS "$NUM_PAIRS"
require_positive_integer WITNESS_CHILD_TIMEOUT_SECONDS "$CHILD_TIMEOUT_SECONDS"
case "$DSL_CONFIG" in
  v9_cgauge_432_launch|v9_cgauge_432_smoke_regime) ;;
  *)
    echo "REFUSED: WITNESS_DSL_CONFIG must be v9_cgauge_432_launch or v9_cgauge_432_smoke_regime; got '$DSL_CONFIG'" >&2
    exit 66
    ;;
esac
# Backend-only Inductor mode override (never typed-DSL science). MEASURED
# 2026-07-15 H100 r5 rc=124: max-autotune benchmarking ate the entire 1440s
# time-boxed window — time-boxed smokes must pass 'default'.
case "$TORCH_COMPILE_MODE" in
  ""|off|default|max-autotune) ;;
  *)
    echo "REFUSED: WITNESS_TORCH_COMPILE_MODE must be empty, off, default, or max-autotune; got '$TORCH_COMPILE_MODE'" >&2
    exit 66
    ;;
esac
if [ "$EPOCHS" -ne 3000 ]; then
  echo "REFUSED: WITNESS_EPOCHS must retain the 3000-epoch typed horizon" >&2
  exit 66
fi
if [ "$CHILD_TIMEOUT_SECONDS" -ne 1500 ]; then
  echo "REFUSED: WITNESS_CHILD_TIMEOUT_SECONDS must equal the conservative 1500-second backstop" >&2
  exit 66
fi
if [ "$CHILD_TIMEOUT_SECONDS" -le 90 ]; then
  echo "REFUSED: child timeout must preserve TERM/KILL reserve" >&2
  exit 66
fi
TRAINER_TERM_SECONDS=$((CHILD_TIMEOUT_SECONDS - 60))
TRAINER_KILL_AFTER_SECONDS=30
if [ "$NUM_PAIRS" -ne 600 ]; then
  echo "REFUSED: WITNESS_NUM_PAIRS must equal the fixed 600-pair Task438 smoke" >&2
  exit 66
fi
GT_CACHE="$(normalize_modal_results_path WITNESS_GT_CACHE "$GT_CACHE")"
OUT_DIR="$(normalize_modal_results_path WITNESS_OUT_DIR "$OUT_DIR")"
if [ -n "$RESUME_FROM" ]; then
  if [[ ! "$RESUME_SHA256" =~ ^[0-9a-f]{64}$ ]]; then
    echo "REFUSED: WITNESS_RESUME_SHA256 must be lowercase exact SHA-256 when resuming" >&2
    exit 69
  fi
  RESUME_FROM="$(normalize_modal_results_path WITNESS_RESUME_FROM "$RESUME_FROM")"
  if [ ! -f "$RESUME_FROM" ] || [ ! -s "$RESUME_FROM" ]; then
    echo "REFUSED: explicit WITNESS_RESUME_FROM must be a nonzero regular file: $RESUME_FROM" >&2
    exit 66
  fi
  if [ -e "$OUT_DIR" ]; then
    if [ ! -d "$OUT_DIR" ]; then
      echo "REFUSED: populated WITNESS_OUT_DIR must be a directory" >&2
      exit 66
    fi
    if [[ "$RESUME_FROM" != "$OUT_DIR/"* ]]; then
      echo "REFUSED: populated WITNESS_OUT_DIR requires resume checkpoint under the same output lineage" >&2
      exit 66
    fi
    OUT_DIR_WAS_PRESENT=1
  fi
elif [ -n "$RESUME_SHA256" ]; then
  echo "REFUSED: WITNESS_RESUME_SHA256 is forbidden without WITNESS_RESUME_FROM" >&2
  exit 69
elif [ -e "$OUT_DIR" ]; then
  echo "REFUSED: fresh timing smoke requires a fresh WITNESS_OUT_DIR; supply explicit resume" >&2
  exit 66
fi
if [[ "$GT_CACHE" != "/modal_results/assets/v9_cgauge/gt_${GT_CACHE_SHA256,,}.npz" ]]; then
  echo "REFUSED: WITNESS_GT_CACHE must be SHA-addressed and match WITNESS_GT_CACHE_SHA256" >&2
  exit 66
fi

validate_file_sha256() {
  "$PYBIN" - "$1" "$2" "$3" <<'PY'
import hashlib, json, pathlib, sys

source = pathlib.Path(sys.argv[1])
expected = sys.argv[2].lower()
label = sys.argv[3]
h = hashlib.sha256()
with source.open("rb") as fh:
    for chunk in iter(lambda: fh.read(8 * 1024 * 1024), b""):
        h.update(chunk)
actual = h.hexdigest()
if actual != expected:
    raise SystemExit(f"REFUSED: {label} SHA-256 mismatch {actual} != {expected}")
print(json.dumps({
    "schema": "witness_remote_file_custody.v1",
    "bytes": source.stat().st_size,
    "label": label,
    "path": str(source),
    "sha256": actual,
    "verdict": "SHA256_VALID",
}, sort_keys=True))
PY
}

validate_gt_cache() {
  validate_file_sha256 "$GT_CACHE" "$GT_CACHE_SHA256" "staged GT cache"
}

validate_scorer_custody() {
  if [[ ! "$SEGNET_SHA256" =~ ^[0-9a-fA-F]{64}$ ]] || [[ ! "$POSENET_SHA256" =~ ^[0-9a-fA-F]{64}$ ]]; then
    echo "REFUSED: scorer SHA-256 custody values are required" >&2
    exit 69
  fi
  validate_file_sha256 "$WORKSPACE/upstream/models/segnet.safetensors" "$SEGNET_SHA256" "SegNet"
  validate_file_sha256 "$WORKSPACE/upstream/models/posenet.safetensors" "$POSENET_SHA256" "PoseNet"
}

validate_resume_custody() {
  if [ -n "$RESUME_FROM" ]; then
    validate_file_sha256 "$RESUME_FROM" "$RESUME_SHA256" "resume checkpoint"
  fi
}

storage_preflight() {
  "$PYBIN" - "$OUT_DIR" "$MIN_FREE_GIB" <<'PY'
import pathlib, shutil, sys
out, minimum = pathlib.Path(sys.argv[1]), float(sys.argv[2])
probe = out
while not probe.exists() and probe != probe.parent:
    probe = probe.parent
free = shutil.disk_usage(probe).free / 2**30
print(f"storage_preflight probe={probe} free_gib={free:.3f} required_gib={minimum:.3f}")
if free < minimum:
    raise SystemExit(67)
PY
}

TRAINER_ARGS=(
  --gt-cache "$GT_CACHE"
  --num-pairs "$NUM_PAIRS"
  --epochs "$EPOCHS"
  --out-dir "$OUT_DIR"
  --device cuda
  --compile-probe
  --dsl-config "$DSL_CONFIG"
  --no-implicit-resume
  --expected-segnet-sha256 "$SEGNET_SHA256"
  --expected-posenet-sha256 "$POSENET_SHA256"
)
if [ -n "$STOP_AFTER_EPOCHS" ]; then
  TRAINER_ARGS+=(--stop-after-epochs "$STOP_AFTER_EPOCHS")
fi
if [ -n "$TORCH_COMPILE_MODE" ]; then
  TRAINER_ARGS+=(--torch-compile-mode "$TORCH_COMPILE_MODE")
fi
if [ -n "$RESUME_FROM" ]; then
  TRAINER_ARGS+=(--resume-from "$RESUME_FROM")
fi
printf '{"schema":"witness_remote_timeout_contract.v1","wrapper_child_timeout_seconds":%s,"trainer_term_seconds":%s,"trainer_kill_after_seconds":%s}\n' \
  "$CHILD_TIMEOUT_SECONDS" "$TRAINER_TERM_SECONDS" "$TRAINER_KILL_AFTER_SECONDS"

if [ "$PREFLIGHT_ONLY" = "1" ]; then
  cd "$WORKSPACE"
  storage_preflight
  validate_gt_cache
  validate_scorer_custody
  validate_resume_custody
  "$PYBIN" experiments/train_levelset_witness_realized_through_R_torch.py \
    "${TRAINER_ARGS[@]}" --preflight-only
  exit 0
fi
if [ "$PREFLIGHT_ONLY" != "0" ]; then
  echo "REFUSED: WITNESS_PREFLIGHT_ONLY must be 0 or 1" >&2
  exit 66
fi

# The CPU preflight normally caught this before H100 allocation; repeat the
# source/geometry gate in the GPU child so a direct invocation is still safe.
storage_preflight
GT_CUSTODY_RECEIPT="$(validate_gt_cache)"
printf '%s\n' "$GT_CUSTODY_RECEIPT"
validate_scorer_custody
validate_resume_custody
cd "$WORKSPACE"
if ! command -v timeout >/dev/null 2>&1; then
  echo "REFUSED: GNU timeout is required for the 1500-second child spending backstop" >&2
  exit 70
fi
"$PYBIN" - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit(68)
print("device", torch.cuda.get_device_name(0), "capability", torch.cuda.get_device_capability(0))
PY
if [ "$OUT_DIR_WAS_PRESENT" = "1" ]; then
  if [ ! -d "$OUT_DIR" ]; then
    echo "REFUSED: resume output directory disappeared after validation" >&2
    exit 66
  fi
else
  mkdir -p "$(dirname "$OUT_DIR")"
  if ! mkdir "$OUT_DIR"; then
    echo "REFUSED: fresh WITNESS_OUT_DIR collided after validation: $OUT_DIR" >&2
    exit 66
  fi
fi
GIT_HASH="$(git -C "$WORKSPACE" rev-parse HEAD 2>/dev/null || echo no-git)"
"$PYBIN" - "$GT_CUSTODY_RECEIPT" "$GT_CACHE" "$GT_CACHE_SHA256" "$OUT_DIR" "$GIT_HASH" <<'PY'
import json, os, pathlib, sys, time
receipt = json.loads(sys.argv[1])
source, expected, out = pathlib.Path(sys.argv[2]), sys.argv[3].lower(), pathlib.Path(sys.argv[4])
git_hash = sys.argv[5]
if (
    receipt.get("schema") != "witness_remote_file_custody.v1"
    or receipt.get("verdict") != "SHA256_VALID"
    or receipt.get("label") != "staged GT cache"
    or receipt.get("path") != str(source)
    or receipt.get("sha256") != expected
):
    raise SystemExit("REFUSED: captured staged GT custody receipt is inconsistent")
payload = {
    "schema": "witness_remote_asset_custody.v1",
    "path": str(source),
    "bytes": int(receipt["bytes"]),
    "sha256": receipt["sha256"],
    "source_validation_schema": receipt["schema"],
    "driver_mode": os.environ.get("WITNESS_TRAINER_MODE"),
    "nvml_disabled": os.environ.get("DALI_DISABLE_NVML"),
}
target = out / "remote_asset_custody.json"
tmp = target.with_suffix(".json.tmp")
tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
os.replace(tmp, target)
print(json.dumps(payload, sort_keys=True))
# Provenance (feedback_canonical_remote_bootstraps): every remote run emits
# provenance.json so a fresh agent can reconstruct the experiment.
provenance = {
    "predicted_band": json.loads(os.environ.get("PREDICTED_BAND", "null")),
    "schema": "remote_run_provenance.v1",
    "started_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "git_hash": git_hash,
    "lane_script": "scripts/remote_v9_cgauge_cuda.sh",
    "dsl_config": os.environ.get("WITNESS_DSL_CONFIG"),
    "trainer_mode": os.environ.get("WITNESS_TRAINER_MODE"),
    "epochs": os.environ.get("WITNESS_EPOCHS"),
    "num_pairs": os.environ.get("WITNESS_NUM_PAIRS"),
    "resume_from": os.environ.get("WITNESS_RESUME_FROM") or None,
    "gt_cache_sha256": receipt["sha256"],
    "out_dir": str(out),
}
prov_target = out / "provenance.json"
prov_tmp = prov_target.with_suffix(".json.tmp")
prov_tmp.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
os.replace(prov_tmp, prov_target)
print(json.dumps(provenance, sort_keys=True))
PY
exec timeout --foreground --signal=TERM --kill-after="${TRAINER_KILL_AFTER_SECONDS}s" \
  "${TRAINER_TERM_SECONDS}s" "$PYBIN" \
  experiments/train_levelset_witness_realized_through_R_torch.py "${TRAINER_ARGS[@]}"
