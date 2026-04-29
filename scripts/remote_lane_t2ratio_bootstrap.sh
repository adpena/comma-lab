#!/bin/bash
# Lane T2-RATIO: SegNet-loss-weight sweep above the historical cap.
#
# Predicted band: [1.00, 1.15] [advisory only].
# Baseline: proven_baseline 1.33 [contest-CUDA].
# GPU budget note: ~12h total (5 trials x ~2h each).
#
# Stages:
#   0 = NVDEC probe
#   1 = environment setup
#   2 = sweep run
#   3 = collect results
#   4 = contest-CUDA auth eval on best-trial checkpoint/archive
#
# INTENTIONAL_OVERRIDE: this lane overrides the ordinary
# `segnet_loss_weight > 100` prohibition only inside the bounded sweep
# {120, 150, 200, 300, 500}, with PoseNet-loss-floor protection.
set -euo pipefail

WORKSPACE=${WORKSPACE:-/workspace/pact}
PYBIN=${PYBIN:-/opt/conda/bin/python}
cd "$WORKSPACE"
[ -f "$WORKSPACE/env.sh" ] && source "$WORKSPACE/env.sh"
export PYTHONHASHSEED=1234

LOG_DIR="$WORKSPACE/lane_t2ratio_results"
SWEEP_DIR="$LOG_DIR/sweep"
PROVENANCE="$LOG_DIR/provenance.json"
HEARTBEAT="$LOG_DIR/heartbeat.log"
mkdir -p "$LOG_DIR" "$SWEEP_DIR"

log() { echo "[lane-t2ratio] $(date -u +%FT%TZ) $*" | tee -a "$LOG_DIR/run.log"; }

GIT_HASH=$(git rev-parse HEAD 2>/dev/null || echo "no-git")
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>&1 | head -1)
DRIVER_VER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>&1 | head -1)

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
    "lane_name": "T2-RATIO",
    "lane_script": "scripts/remote_lane_t2ratio_bootstrap.sh",
    "controlled_baseline_lane": "proven_baseline",
    "controlled_baseline": "proven_baseline with segnet_loss_weight at the historical cap",
    "changed_mechanism": "segnet_loss_weight grid above 100 with PoseNet floor",
    "seg_weight_grid": [120, 150, 200, 300, 500],
    "epochs_per_trial": 250,
    "pose_floor_multiplier": 2.0,
    "predicted_band": [1.00, 1.15],
    "score_context": "[contest-CUDA] for final auth eval; [advisory only] for proxy sweep ranking",
    "eval_roundtrip": True,
    "output_dir": "$LOG_DIR",
}
with open("$PROVENANCE", "w") as f:
    json.dump(prov, f, indent=2)
print("provenance:", json.dumps(prov))
PY

( while true; do
    GPU=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>&1 | tr '\n' ' ')
    echo "[$(date -u +%FT%TZ)] lane=T2-RATIO gpu=$GPU" >> "$HEARTBEAT"
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

log "=== Stage 2: sweep run ==="
"$PYBIN" -u experiments/sweep_seg_weight.py \
    --output-dir "$SWEEP_DIR" \
    --epochs 250 \
    --check-every 25 \
    --profile proven_baseline \
    --tag-prefix t2ratio \
    --precomputed experiments/precomputed_local \
    --device cuda \
    --auth-eval-during-sweep \
    --auth-eval-masks "$ANCHOR_MASKS" \
    --auth-eval-poses "$ANCHOR_POSES" \
    --auth-eval-upstream-dir upstream \
    2>&1 | tee "$LOG_DIR/sweep.log"
    PIPE_RC=("${PIPESTATUS[@]}")
    if [ "${PIPE_RC[0]}" -ne 0 ]; then
        echo "FATAL: previous pipeline exited rc=${PIPE_RC[0]}" >&2; exit "${PIPE_RC[0]}"
    fi

RESULTS_JSON="$SWEEP_DIR/sweep_seg_weight_results.json"
[ -f "$RESULTS_JSON" ] || { echo "FATAL: missing $RESULTS_JSON" >&2; exit 2; }

log "=== Stage 3: collect best trial ==="
BEST_INFO="$LOG_DIR/best_trial.json"
"$PYBIN" - <<PY
import json
src = json.load(open("$RESULTS_JSON"))
best = src["best_trial"]
if not best:
    raise SystemExit("no best_trial in sweep results")
with open("$BEST_INFO", "w") as f:
    json.dump(best, f, indent=2)
print(json.dumps(best, indent=2))
PY

BEST_ARCHIVE=$("$PYBIN" - <<PY
import json
from pathlib import Path
best = json.load(open("$BEST_INFO"))
trial_dir = Path(best["trial_dir"])
archive = trial_dir / "auth_eval_on_best_archive.zip"
print(archive)
PY
)
[ -f "$BEST_ARCHIVE" ] || { echo "FATAL: missing best archive $BEST_ARCHIVE" >&2; exit 2; }

log "=== Stage 3b: verify archive with Python zipfile.ZipFile ==="
"$PYBIN" - <<PY
import zipfile
with zipfile.ZipFile("$BEST_ARCHIVE") as zf:
    names = set(zf.namelist())
required = {"renderer.bin", "masks.mkv", "optimized_poses.pt"}
missing = required - names
if missing:
    raise SystemExit(f"archive missing {sorted(missing)}")
print("archive ok:", "$BEST_ARCHIVE")
PY

log "=== Stage 4: contest-CUDA auth eval on best-trial checkpoint/archive ==="
rm -rf "$LOG_DIR/eval_work"
"$PYBIN" -u experiments/contest_auth_eval.py \
    --archive "$BEST_ARCHIVE" \
    --inflate-sh submissions/robust_current/inflate.sh \
    --upstream-dir upstream \
    --device "${AUTH_EVAL_DEVICE:-cuda}" \
    --keep-work-dir \
    --work-dir "$LOG_DIR/eval_work" \
    2>&1 | tee "$LOG_DIR/auth_eval_best.log"
    PIPE_RC=("${PIPESTATUS[@]}")
    if [ "${PIPE_RC[0]}" -ne 0 ]; then
        echo "FATAL: previous pipeline exited rc=${PIPE_RC[0]}" >&2; exit "${PIPE_RC[0]}"
    fi

grep -q '^RESULT_JSON' "$LOG_DIR/auth_eval_best.log" || {
    log "FATAL: final contest-CUDA auth eval emitted no RESULT_JSON"
    exit 2
}
log "=== T2-RATIO complete [contest-CUDA]; see $LOG_DIR/auth_eval_best.log ==="
