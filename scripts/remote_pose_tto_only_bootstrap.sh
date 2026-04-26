#!/bin/bash
# Canonical bootstrap for POSE-TTO-ONLY (no QAT, no retrain) on an existing
# FP4 .bin renderer. Use this when the renderer is already trained + QAT'd
# and we want to ONLY iterate on poses. Pattern matches remote_train_bootstrap.sh
# and remote_pose_tto_bootstrap.sh — two-mode (default = launch tmux,
# --inner = do work).
#
# 2026-04-26: lifted from /tmp/lane_b_launcher.sh after the user enforced
# the "no ad-hoc .sh scripts" non-negotiable. Per CLAUDE.md "Canonical
# Pipeline Standard": this is the ONLY way to launch a remote pose-TTO-only
# job. Ad-hoc /tmp scripts are blocked by preflight.
#
# Difference vs scripts/remote_pose_tto_bootstrap.sh:
#   pose_tto_bootstrap.sh runs the FULL pipeline.py compress chain
#       (export → QAT → pose_tto → archive → eval) — needs a float .pt
#   pose_tto_ONLY_bootstrap.sh runs ONLY:
#       optimize_poses → archive → eval — accepts an FP4 .bin directly
#
# Usage (on remote):
#   bash scripts/remote_pose_tto_only_bootstrap.sh \
#       <renderer_bin> <masks_mkv> <output_subdir> [posetto_noise_std=0.5] [steps=1000]
#
# Example:
#   bash scripts/remote_pose_tto_only_bootstrap.sh \
#       /workspace/pact/baseline/renderer.bin \
#       /workspace/pact/baseline/masks.mkv \
#       experiments/results/lane_b
#
# Required deterministic-reproducibility env vars (CLAUDE.md non-negotiable):
#   CUBLAS_WORKSPACE_CONFIG=:4096:8  ← exported BEFORE any cuBLAS call
#   PYTHONHASHSEED=42                ← matches profile.seed
#   PYTHONUNBUFFERED=1               ← so logs land in real time

set -uo pipefail

WORKSPACE=/workspace/pact
PYBIN=/opt/conda/bin/python

# ── Mode dispatch ─────────────────────────────────────────────────────────
if [ "${1:-}" = "--inner" ]; then
    RENDERER_BIN="$2"
    MASKS_MKV="$3"
    OUTPUT_SUBDIR="$4"
    POSETTO_NOISE_STD="${5:-0.5}"  # Fridrich C1 fix: never default to 0
    STEPS="${6:-1000}"
    cd "$WORKSPACE"

    LOG_DIR="$WORKSPACE/$OUTPUT_SUBDIR"
    HEARTBEAT="$LOG_DIR/heartbeat.log"
    PROVENANCE="$LOG_DIR/provenance.json"
    mkdir -p "$LOG_DIR"

    # ── Stage 0: deterministic-reproducibility env (REQUIRED before cuBLAS) ─
    export PATH=/root/.local/bin:$PATH
    export PYTHONPATH=src:upstream:$PWD
    export TAC_UPSTREAM_DIR=/workspace/pact/upstream
    export PYTHONUNBUFFERED=1
    export PYTHONHASHSEED=42
    export CUBLAS_WORKSPACE_CONFIG=:4096:8

    # Provenance — written BEFORE any work so the record exists even if the
    # run dies mid-stage. CLAUDE.md canonical pipeline standard.
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
    'cuda_version': torch.version.cuda,
    'cuda_available': torch.cuda.is_available(),
    'pipeline': 'remote_pose_tto_only_bootstrap.sh -> optimize_poses -> archive -> auth_eval',
    'renderer_bin': '$RENDERER_BIN',
    'masks_mkv': '$MASKS_MKV',
    'output_dir': '$LOG_DIR',
    'posetto_noise_std': $POSETTO_NOISE_STD,
    'steps': $STEPS,
    'cublas_workspace_config': '$CUBLAS_WORKSPACE_CONFIG',
    'pythonhashseed': '$PYTHONHASHSEED',
}
with open('$PROVENANCE', 'w') as f:
    json.dump(prov, f, indent=2)
print('provenance:', json.dumps(prov, indent=2))
"

    # Heartbeat sub-shell — 60s cadence with profile cookie ⇒ never `pgrep -f` self-matches
    ( while true; do
        GPU=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>&1 | tr '\n' ' ')
        echo "[$(date -u +%FT%TZ)] pose_tto_only gpu=$GPU" >> "$HEARTBEAT"
        sleep 60
      done ) &
    HB_PID=$!
    trap "kill $HB_PID 2>/dev/null || true" EXIT

    # ── Stage 1: deps (apt + pip) — same as remote_train_bootstrap.sh ──────
    echo "=== STAGE 1: system + Python deps ==="
    if [ ! -x "$PYBIN" ]; then
        echo "FATAL: missing /opt/conda/bin/python — wrong container image"
        exit 1
    fi
    if ! command -v ffprobe >/dev/null 2>&1; then
        DEBIAN_FRONTEND=noninteractive apt-get install -y -qq ffmpeg 2>&1 | tail -2
    fi
    [ -f "$WORKSPACE/README.md" ] || { echo "FATAL: README.md missing — setuptools install will fail"; exit 1; }
    # 2026-04-26: canonical via pyproject.toml [project.optional-dependencies.runtime]
    "$PYBIN" -m ensurepip --upgrade 2>&1 | tail -1
    "$PYBIN" -m pip install -q --upgrade pip 2>&1 | tail -1
    "$PYBIN" -m pip install -q -e ".[runtime]" 2>&1 | tail -3

    # ── Stage 1b: ffmpeg-new + libsvtav1 (per memory: feedback_ffmpeg_svtav1_deploy) ─
    FFMPEG_NEW="$WORKSPACE/upstream/ffmpeg-new"
    SVT_LIB_DIR="$WORKSPACE/upstream/submissions/av1_roi_lanczos_unsharp/lib"
    if [ -f "$FFMPEG_NEW" ]; then
        chmod +x "$FFMPEG_NEW" 2>/dev/null || true
        if [ -f "$SVT_LIB_DIR/libSvtAv1Enc.so.2.3.0" ] && [ ! -e "$SVT_LIB_DIR/libSvtAv1Enc.so.2" ]; then
            ln -sf "$SVT_LIB_DIR/libSvtAv1Enc.so.2.3.0" "$SVT_LIB_DIR/libSvtAv1Enc.so.2"
        fi
        export LD_LIBRARY_PATH="$SVT_LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
        if "$FFMPEG_NEW" -version >/dev/null 2>&1; then
            echo "OK upstream/ffmpeg-new ready (n7.x with libsvtav1)"
        else
            echo "WARN: upstream/ffmpeg-new present but won't run — mask encoding may fall back to system ffmpeg" >&2
        fi
    fi

    # ── Stage 2: pose TTO with Fridrich-fixed noise_std default ────────────
    echo "=== STAGE 2: pose TTO (steps=$STEPS, noise_std=$POSETTO_NOISE_STD) ==="
    echo "[$(date -u +%FT%TZ)] launching optimize_poses.py" >> "$HEARTBEAT"
    "$PYBIN" -u experiments/optimize_poses.py \
        --checkpoint "$RENDERER_BIN" \
        --masks "$MASKS_MKV" \
        --device cuda \
        --steps "$STEPS" \
        --batch-pairs 8 \
        --eval-roundtrip \
        --posetto-noise-std "$POSETTO_NOISE_STD" \
        --output-dir "$LOG_DIR" 2>&1 | tee "$LOG_DIR/optimize_poses.log"

    if [ ! -f "$LOG_DIR/optimized_poses.bin" ]; then
        echo "FATAL: optimized_poses.bin not produced"
        exit 2
    fi

    # ── Stage 3: build canonical archive ───────────────────────────────────
    echo "=== STAGE 3: build archive ==="
    mkdir -p "$LOG_DIR/iter_0"
    cp "$RENDERER_BIN" "$LOG_DIR/iter_0/renderer.bin"
    cp "$MASKS_MKV" "$LOG_DIR/iter_0/masks.mkv"
    cp "$LOG_DIR/optimized_poses.bin" "$LOG_DIR/iter_0/optimized_poses.bin"
    cd "$LOG_DIR/iter_0"
    zip -j archive.zip renderer.bin masks.mkv optimized_poses.bin >/dev/null
    ARCHIVE_BYTES=$(stat -c '%s' archive.zip)
    echo "  archive: $ARCHIVE_BYTES bytes"
    cd "$WORKSPACE"

    # ── Stage 4: CUDA auth eval (only valid measurement) ───────────────────
    echo "=== STAGE 4: CUDA auth eval ==="
    "$PYBIN" -u experiments/auth_eval_renderer.py \
        --checkpoint "$LOG_DIR/iter_0/renderer.bin" \
        --upstream-dir upstream \
        --device cuda \
        --archive-size-bytes "$ARCHIVE_BYTES" \
        --poses "$LOG_DIR/iter_0/optimized_poses.bin" \
        2>&1 | tee "$LOG_DIR/auth_eval.log"

    echo "=== POSE_TTO_ONLY_DONE_$(date +%s) ==="
    echo "[$(date -u +%FT%TZ)] POSE_TTO_ONLY_PIPELINE_DONE" >> "$HEARTBEAT"
    exit 0
fi

# ── Default mode: launch tmux ─────────────────────────────────────────────
if [ $# -lt 3 ]; then
    echo "Usage: $0 <renderer_bin> <masks_mkv> <output_subdir> [posetto_noise_std=0.5] [steps=1000]" >&2
    exit 1
fi

RENDERER_BIN="$1"
MASKS_MKV="$2"
OUTPUT_SUBDIR="$3"
POSETTO_NOISE_STD="${4:-0.5}"
STEPS="${5:-1000}"

# Derive a stable session name from the output subdir
SESSION="$(basename "$OUTPUT_SUBDIR")_pose_tto"
LOG_DIR="$WORKSPACE/$OUTPUT_SUBDIR"
TRAIN_LOG="$LOG_DIR/train.log"

if [ ! -d "$WORKSPACE" ]; then
    echo "FATAL: $WORKSPACE missing — sync repo first." >&2
    exit 1
fi
mkdir -p "$LOG_DIR"

tmux kill-session -t "$SESSION" 2>/dev/null || true

SCRIPT_PATH="$WORKSPACE/scripts/remote_pose_tto_only_bootstrap.sh"
tmux new-session -d -s "$SESSION" \
    "bash '$SCRIPT_PATH' --inner '$RENDERER_BIN' '$MASKS_MKV' '$OUTPUT_SUBDIR' '$POSETTO_NOISE_STD' '$STEPS' 2>&1 | tee -a '$TRAIN_LOG'"

sleep 3
echo "tmux session '$SESSION' status:"
tmux ls 2>&1 | grep "$SESSION" || echo "  (NOT FOUND — check $TRAIN_LOG)"
echo
echo "DEPLOY_OK session=$SESSION log=$TRAIN_LOG"
