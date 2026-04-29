#!/bin/bash
# Lane T2-DROP: encoder-free renderer ablation.
#
# Informational-only ablation, not a stacking candidate.
# Predicted regression: ~1.5-3.0 [advisory only].
# Baseline: proven_baseline 1.33 [contest-CUDA].
#
# Stages:
#   0 = NVDEC probe
#   1 = environment setup
#   2 = train with --no-mask-encoder
#   3 = contest-CUDA auth eval
set -euo pipefail

WORKSPACE=${WORKSPACE:-/workspace/pact}
PYBIN=${PYBIN:-/opt/conda/bin/python}
cd "$WORKSPACE"
[ -f "$WORKSPACE/env.sh" ] && source "$WORKSPACE/env.sh"
export PYTHONHASHSEED=1234

LOG_DIR="$WORKSPACE/lane_t2drop_results"
PROVENANCE="$LOG_DIR/provenance.json"
HEARTBEAT="$LOG_DIR/heartbeat.log"
mkdir -p "$LOG_DIR"

log() { echo "[lane-t2drop] $(date -u +%FT%TZ) $*" | tee -a "$LOG_DIR/run.log"; }

GIT_HASH=$(git rev-parse HEAD 2>/dev/null || echo "no-git")
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>&1 | head -1)
DRIVER_VER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>&1 | head -1)

# provenance.json includes informational_only: true
"$PYBIN" - <<PY
import json, time, torch
prov = {
    "started_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "git_hash": "$GIT_HASH",
    "gpu_name": "$GPU_NAME",
    "driver_version": "$DRIVER_VER",
    "torch_version": torch.__version__,
    "cuda_version": getattr(torch.version, "cuda", None),
    "cuda_available": torch.cuda.is_available(),
    "lane_name": "T2-DROP",
    "lane_script": "scripts/remote_lane_t2drop_bootstrap.sh",
    "controlled_baseline_lane": "proven_baseline",
    "controlled_baseline": "same training command without --no-mask-encoder",
    "changed_mechanism": "zero-fill mask encoder output",
    "informational_only": True,
    "stacking_candidate": False,
    "predicted_band": [1.5, 3.0],
    "score_context": "[contest-CUDA] for final auth eval; [advisory only] for predicted regression",
    "eval_roundtrip": True,
    "output_dir": "$LOG_DIR",
}
with open("$PROVENANCE", "w") as f:
    json.dump(prov, f, indent=2)
print("provenance:", json.dumps(prov))
PY

( while true; do
    GPU=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>&1 | tr '\n' ' ')
    echo "[$(date -u +%FT%TZ)] lane=T2-DROP gpu=$GPU" >> "$HEARTBEAT"
    sleep 60
  done ) &
HB_PID=$!
trap 'kill $HB_PID 2>/dev/null || true' EXIT

log "=== Stage 0: NVDEC probe ==="
bash "$WORKSPACE/scripts/probe_nvdec.sh" || {
    log "FATAL: NVDEC probe failed; choose another Vast.ai host."
    exit 2
}

log "=== Stage 1: env setup ==="
"$PYBIN" - <<'PY'
import importlib.util
import sys
missing = [m for m in ("torch", "tac") if importlib.util.find_spec(m) is None]
if missing:
    raise SystemExit(f"missing modules: {missing}")
print("env ok:", sys.executable)
PY

ANCHOR_MASKS="experiments/results/lane_a_landed/extracted/masks.mkv"
ANCHOR_POSES="experiments/results/lane_a_landed/optimized_poses.pt"
for f in "$ANCHOR_MASKS" "$ANCHOR_POSES" upstream/videos/0.mkv upstream/models/segnet.safetensors upstream/models/posenet.safetensors; do
    [ -f "$f" ] || { echo "FATAL: missing $f" >&2; exit 1; }
done

log "=== Stage 2: train with --no-mask-encoder informational-only ablation ==="
"$PYBIN" -u -m tac.experiments.train_renderer \
    --profile proven_baseline \
    --tag t2drop_no_mask_encoder \
    --output-dir "$LOG_DIR" \
    --epochs 250 \
    --eval-every 25 \
    --precomputed experiments/precomputed_local \
    --device cuda \
    --eval-roundtrip \
    --no-mask-encoder \
    --auth-eval-masks "$ANCHOR_MASKS" \
    --auth-eval-poses "$ANCHOR_POSES" \
    --auth-eval-upstream-dir upstream \
    2>&1 | tee "$LOG_DIR/train.log"

ARCHIVE="$LOG_DIR/auth_eval_on_best_archive.zip"
[ -f "$ARCHIVE" ] || { echo "FATAL: training did not produce $ARCHIVE" >&2; exit 2; }

log "=== Stage 2b: verify archive with Python zipfile.ZipFile ==="
"$PYBIN" - <<PY
import zipfile
with zipfile.ZipFile("$ARCHIVE") as zf:
    names = set(zf.namelist())
required = {"renderer.bin", "masks.mkv", "optimized_poses.pt"}
missing = required - names
if missing:
    raise SystemExit(f"archive missing {sorted(missing)}")
print("archive ok:", "$ARCHIVE")
PY

log "=== Stage 3: contest-CUDA auth eval (eval_roundtrip preserved) ==="
rm -rf "$LOG_DIR/eval_work"
"$PYBIN" -u experiments/contest_auth_eval.py \
    --archive "$ARCHIVE" \
    --inflate-sh submissions/robust_current/inflate.sh \
    --upstream-dir upstream \
    --device cuda \
    --keep-work-dir \
    --work-dir "$LOG_DIR/eval_work" \
    2>&1 | tee "$LOG_DIR/auth_eval.log"

grep -q '^RESULT_JSON' "$LOG_DIR/auth_eval.log" || {
    log "FATAL: final contest-CUDA auth eval emitted no RESULT_JSON"
    exit 2
}
log "=== T2-DROP complete [contest-CUDA]; informational_only=true; see $LOG_DIR/auth_eval.log ==="
