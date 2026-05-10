#!/bin/bash
# Lane A-Sweep: Bayesian sweep over Lane A pose TTO hyperparameters.
#
# Replaces the hand-tuned Lane A schedule (steps=500, batch-pairs=8,
# tto-lr=0.01, posetto-noise-std=0.5, baseline 1.15 [contest-CUDA]) with a
# 30-trial TPE Bayesian search. Predicted Lane A-Sweep band: [0.95, 1.15].
#
# This script is BOTH:
#   1. The per-trial template (placeholders __PARAM_X__ are substituted by
#      experiments/sweep_lane_a_pose_tto.py at trial-dispatch time), AND
#   2. The orchestrator wrapper (when run directly, it kicks off the sweep
#      driver which dispatches scripts inline + reads results).
#
# Trial template placeholders (every __PARAM_X__ is required, validated at
# sweep construction):
#   800           -> int, sampled
#   12         -> int, sampled
#   0.005              -> float, log-sampled
#   0.4   -> float, sampled
#   1      -> 1 (fixed True per CLAUDE.md)
#   cuda              -> "cuda" (fixed per CLAUDE.md)
#   lane_a_optimized                -> "lane_a_pose_tto"
#   BEST        -> e.g. "0017" or "BEST"
#   __SWEEP_SEARCH_SPACE_HASH__   -> SHA256[:16] for reproducibility
set -euo pipefail
WORKSPACE="${WORKSPACE:-/workspace/pact}"
PYBIN="${PYBIN:-/opt/conda/bin/python}"
[ -f "$WORKSPACE/env.sh" ] && source "$WORKSPACE/env.sh"
cd "$WORKSPACE"
export PYTHONHASHSEED=1234

SWEEP_NAME="lane_a_optimized"
TRIAL_NUMBER="BEST"
SEARCH_SPACE_HASH="__SWEEP_SEARCH_SPACE_HASH__"

# Two operational modes:
#   - If running as a TRIAL: TRIAL_NUMBER is a digit string or 'BEST'.
#     We execute one configured pose-TTO + auth eval and write a sidecar
#     result.json next to ourselves.
#   - If running as the ORCHESTRATOR: BEST is still the
#     literal string 'BEST' (unsubstituted). We dispatch
#     the Optuna sweep driver instead.
if [[ "$TRIAL_NUMBER" == "BEST" ]]; then
    LOG_DIR="$WORKSPACE/lane_a_sweep_results"
    mkdir -p "$LOG_DIR"

    # Provenance — sweep-aware (CLAUDE.md non-negotiable canonical bootstraps).
    PROVENANCE="$LOG_DIR/provenance.json"
    GIT_HASH=$(cd "$WORKSPACE" && git rev-parse HEAD 2>/dev/null || echo "no-git")
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>&1 | head -1)
    DRIVER_VER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>&1 | head -1)
    "$PYBIN" -c "
import json, time, torch
prov = {
    'started_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    'git_hash': '$GIT_HASH',
    'gpu_name': '$GPU_NAME',
    'driver_version': '$DRIVER_VER',
    'torch_version': torch.__version__,
    'cuda_version': getattr(torch.version, 'cuda', None),
    'cuda_available': torch.cuda.is_available(),
    'lane_script': 'scripts/remote_lane_a_sweep.sh',
    'lane_name': 'lane_a_sweep',
    'is_sweep': True,
    'n_trials': 30,
    'sweep_name': 'lane_a_pose_tto',
    'output_dir': '$LOG_DIR',
}
with open('$PROVENANCE', 'w') as f:
    json.dump(prov, f, indent=2)
print('provenance:', json.dumps(prov))
"

    # Stage 0: NVDEC probe (memory feedback_vastai_nvdec_host_variation).
    echo "=== Stage 0: NVDEC probe ==="
    bash "$WORKSPACE/scripts/probe_nvdec.sh" || {
        echo "FATAL: NVDEC probe failed — destroy this instance, pick a different host."
        exit 2
    }

    # Stage 1: bootstrap rsync (no-op if already in workspace).
    echo "=== Stage 1: bootstrap (rsync repo) ==="
    [ -d "$WORKSPACE/upstream" ] || {
        echo "FATAL: workspace missing upstream/ — run remote_setup_full.sh first"
        exit 1
    }

    # Stage 2: install optuna (CLAUDE.md uv mandatory).
    echo "=== Stage 2: install optuna ==="
    uv pip install --system optuna 2>&1 | tail -3

    # Stage 3: run the sweep driver. Each trial inside the driver substitutes
    # this same template with concrete params and runs it sequentially. The
    # driver tracks the best trial and emits scripts/remote_lane_a_optimized.sh.
    echo "=== Stage 3: Bayesian sweep over pose-TTO hyperparameters ==="
    set +e
    "$PYBIN" -u experiments/sweep_lane_a_pose_tto.py \
        --n-trials 30 \
        --objective auth_score \
        --output-dir "$LOG_DIR" 2>&1 | tee "$LOG_DIR/sweep.log" | tail -50
        PIPE_RC=("${PIPESTATUS[@]}")
    set -e
        if [ "${PIPE_RC[0]}" -ne 0 ]; then
            echo "FATAL: previous pipeline exited rc=${PIPE_RC[0]}" >&2; exit "${PIPE_RC[0]}"
        fi

    # Stage 4: run the OPTIMIZED config as the official Lane A-Sweep result.
    OPT_SCRIPT="$WORKSPACE/scripts/remote_lane_a_optimized.sh"
    [ -f "$OPT_SCRIPT" ] || {
        echo "FATAL: sweep driver did not emit $OPT_SCRIPT"
        exit 2
    }
    echo "=== Stage 4: official Lane A-Sweep result (best-trial archive) ==="
    set +e
    bash "$OPT_SCRIPT" 2>&1 | tee "$LOG_DIR/optimized.log" | tail -30
    PIPE_RC=("${PIPESTATUS[@]}")
    set -e
    if [ "${PIPE_RC[0]}" -ne 0 ]; then
        echo "FATAL: previous pipeline exited rc=${PIPE_RC[0]}" >&2; exit "${PIPE_RC[0]}"
    fi

    # Validate the official result has a RESULT_JSON (no-wasted-resources rule).
    grep -q '^RESULT_JSON' "$LOG_DIR/optimized.log" || {
        echo "FATAL: optimized run did not produce RESULT_JSON — invalid measurement"
        exit 2
    }
    echo "=== LANE_A_SWEEP_DONE — see $LOG_DIR ==="
    exit 0
fi

# ----- TRIAL MODE: substitute placeholders, run one config, emit result -----

LOG_DIR="$WORKSPACE/lane_a_sweep_results/trial_${TRIAL_NUMBER}"
mkdir -p "$LOG_DIR"

log() { echo "[lane-a-sweep trial=${TRIAL_NUMBER}] $(date -u +%FT%TZ) $*" | tee -a "$LOG_DIR/run.log"; }

log "=== trial provenance ==="
"$PYBIN" -c "
import json, time, torch
prov = {
    'started_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    'sweep_name': '$SWEEP_NAME',
    'trial_number': '$TRIAL_NUMBER',
    'search_space_hash': '$SEARCH_SPACE_HASH',
    'is_sweep': True,
    'tto_steps': 800,
    'batch_pairs': 12,
    'tto_lr': 0.005,
    'posetto_noise_std': 0.4,
    'eval_roundtrip': bool(1),
    'device': 'cuda',
    'cuda_available': torch.cuda.is_available(),
}
with open('$LOG_DIR/trial_provenance.json', 'w') as f:
    json.dump(prov, f, indent=2)
print('trial_provenance:', json.dumps(prov))
"

# Pre-flight: required artifacts (same as Lane A baseline).
for f in submissions/baseline_dilated_h64_0_90/renderer.bin \
         submissions/baseline_dilated_h64_0_90/optimized_poses.pt \
         upstream/videos/0.mkv \
         upstream/models/segnet.safetensors \
         upstream/models/posenet.safetensors; do
    [ -f "$f" ] || { echo "FATAL: missing $f" >&2; exit 1; }
done

log "=== rebuild masks (same as Lane A baseline) ==="
set +e
"$PYBIN" experiments/build_baseline_archive.py \
    --device cuda --crf 50 \
    --output "$LOG_DIR/archive_baseline_seed.zip" 2>&1 | tee "$LOG_DIR/build.log" | tail -3
    PIPE_RC=("${PIPESTATUS[@]}")
set -e
    if [ "${PIPE_RC[0]}" -ne 0 ]; then
        echo "FATAL: previous pipeline exited rc=${PIPE_RC[0]}" >&2; exit "${PIPE_RC[0]}"
    fi

mkdir -p "$LOG_DIR/extracted"
cd "$LOG_DIR/extracted" && unzip -o "$LOG_DIR/archive_baseline_seed.zip" 2>&1 | tail -3
cd "$WORKSPACE"

log "=== pose TTO with sampled hyperparameters ==="
log "  tto_steps=800 batch_pairs=12"
log "  tto_lr=0.005 posetto_noise_std=0.4"
set +e
"$PYBIN" -u experiments/optimize_poses.py \
    --checkpoint submissions/baseline_dilated_h64_0_90/renderer.bin \
    --masks "$LOG_DIR/extracted/masks.mkv" \
    --gt-poses-path submissions/baseline_dilated_h64_0_90/optimized_poses.pt \
    --device cuda \
    --steps 800 \
    --batch-pairs 12 \
    --lr 0.005 \
    --eval-roundtrip \
    --posetto-noise-std 0.4 \
    --output-dir "$LOG_DIR" 2>&1 | tee "$LOG_DIR/optimize_poses.log" | tail -20
    PIPE_RC=("${PIPESTATUS[@]}")
set -e
    if [ "${PIPE_RC[0]}" -ne 0 ]; then
        echo "FATAL: previous pipeline exited rc=${PIPE_RC[0]}" >&2; exit "${PIPE_RC[0]}"
    fi

[ -f "$LOG_DIR/optimized_poses.pt" ] || { echo "FATAL: missing optimized_poses.pt"; exit 2; }

log "=== build trial archive (Python zipfile, NOT shell zip) ==="
mkdir -p "$LOG_DIR/iter_0"
cp submissions/baseline_dilated_h64_0_90/renderer.bin "$LOG_DIR/iter_0/renderer.bin"
cp "$LOG_DIR/extracted/masks.mkv" "$LOG_DIR/iter_0/masks.mkv"
cp "$LOG_DIR/optimized_poses.pt" "$LOG_DIR/iter_0/optimized_poses.pt"
ARCHIVE="$LOG_DIR/archive_trial.zip"
"$PYBIN" -c "
import zipfile, os
src = '$LOG_DIR/iter_0'
dst = '$ARCHIVE'
with zipfile.ZipFile(dst, 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as z:
    for n in ('renderer.bin', 'masks.mkv', 'optimized_poses.pt'):
        p = os.path.join(src, n)
        assert os.path.isfile(p), f'missing {p}'
        z.write(p, arcname=n)
print(f'archive {dst}: {os.path.getsize(dst)} bytes')
"
ARCHIVE_BYTES=$(stat -c '%s' "$ARCHIVE" 2>/dev/null || stat -f '%z' "$ARCHIVE")
if [ -z "${ARCHIVE_BYTES:-}" ] || [ "$ARCHIVE_BYTES" -le 0 ]; then
    echo "FATAL: archive size empty or zero — refusing to call auth_eval"
    exit 2
fi

log "=== contest_auth_eval ==="
rm -rf "$LOG_DIR/eval_work"
set +e
"$PYBIN" -u experiments/contest_auth_eval.py \
    --archive "$ARCHIVE" \
    --inflate-sh submissions/robust_current/inflate.sh \
    --upstream-dir upstream \
    --device cuda \
    --keep-work-dir \
    --work-dir "$LOG_DIR/eval_work" 2>&1 | tee "$LOG_DIR/auth_eval.log" | tail -10
    PIPE_RC=("${PIPESTATUS[@]}")
set -e
    if [ "${PIPE_RC[0]}" -ne 0 ]; then
        echo "FATAL: previous pipeline exited rc=${PIPE_RC[0]}" >&2; exit "${PIPE_RC[0]}"
    fi

grep -q '^RESULT_JSON' "$LOG_DIR/auth_eval.log" || {
    echo "FATAL: auth_eval did not produce RESULT_JSON"
    exit 2
}

# Extract the score and write the sidecar result.json that
# BayesianSweep.parse_remote_result() expects.
SIDE_CAR="${BASH_SOURCE[0]%.sh}.result.json"
"$PYBIN" -c "
import json, re, sys
log = open('$LOG_DIR/auth_eval.log').read()
m = re.search(r'^RESULT_JSON:\s*(\{.*\})\s*$', log, re.M)
assert m, 'RESULT_JSON missing after grep passed?!'
payload = json.loads(m.group(1))
out = {
    'final_score': payload.get('final_score'),
    'auth_score':  payload.get('final_score'),
    'avg_posenet_dist': payload.get('avg_posenet_dist'),
    'avg_segnet_dist':  payload.get('avg_segnet_dist'),
    'rate_unscaled':    payload.get('rate_unscaled'),
    'archive_size_bytes': payload.get('archive_size_bytes'),
    'trial_number': '$TRIAL_NUMBER',
    'sweep_name':   '$SWEEP_NAME',
}
with open('$SIDE_CAR', 'w') as f:
    json.dump(out, f, indent=2)
print('wrote sidecar:', '$SIDE_CAR')
"

log "=== TRIAL_DONE — sidecar at $SIDE_CAR ==="
