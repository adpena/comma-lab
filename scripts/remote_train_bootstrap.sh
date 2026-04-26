#!/bin/bash
# Canonical remote training bootstrap. Reusable for ANY profile via $1.
# Two-mode operation:
#   - default mode: launch tmux session that re-invokes this script with --inner
#   - --inner mode: do the actual bootstrap+train+compress+auth eval work
#
# This pattern avoids the "embedded heredoc inside bash -lc" quoting hell
# that broke the 2026-04-26 deploy attempts. The inner script body lives
# in a single self-contained file, not as a string substituted into tmux.
#
# Usage (on remote):
#   bash scripts/remote_train_bootstrap.sh <profile_name> [output_subdir]
#
# Example:
#   bash scripts/remote_train_bootstrap.sh den
#
# After launch you can disconnect SSH; the tmux session continues. Monitor:
#   ssh root@HOST "tmux capture-pane -t <profile>_train -p | tail -30"
#   ssh root@HOST "tail /workspace/pact/experiments/results/<profile>/heartbeat.log"
#
# Per CLAUDE.md "Canonical Pipeline Standard": this is the ONLY way to
# launch a remote training job. Ad-hoc /tmp scripts are blocked by
# preflight (_scan_text_for_dangerous_patterns).

set -euo pipefail

# ── Constants ─────────────────────────────────────────────────────────────
WORKSPACE=/workspace/pact

# ── Mode dispatch ─────────────────────────────────────────────────────────
if [ "${1:-}" = "--inner" ]; then
    PROFILE="$2"
    OUTPUT_SUBDIR="$3"
    cd "$WORKSPACE"

    export PATH=/root/.local/bin:$PATH
    export PYTHONPATH=src:upstream:$PWD
    export TAC_UPSTREAM_DIR=/workspace/pact/upstream
    export PYTHONUNBUFFERED=1
    export CUBLAS_WORKSPACE_CONFIG=:4096:8

    LOG_DIR="$WORKSPACE/$OUTPUT_SUBDIR"
    HEARTBEAT="$LOG_DIR/heartbeat.log"
    PROVENANCE="$LOG_DIR/provenance.json"
    mkdir -p "$LOG_DIR"

    # Provenance record (CLAUDE.md canonical pipeline standard: full
    # provenance per run — git hash, GPU, torch+CUDA versions, profile,
    # timestamps, deterministic config). Written BEFORE any work starts so
    # the record exists even if the run dies mid-stage.
    GIT_HASH=$(cd "$WORKSPACE" && git rev-parse HEAD 2>/dev/null || echo "no-git")
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>&1 | head -1)
    DRIVER_VER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>&1 | head -1)
    /opt/conda/bin/python -c "
import json, time, torch
prov = {
    'started_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    'git_hash': '$GIT_HASH',
    'gpu_name': '$GPU_NAME',
    'driver_version': '$DRIVER_VER',
    'torch_version': torch.__version__,
    'cuda_version': torch.version.cuda,
    'cuda_available': torch.cuda.is_available(),
    'profile': '$PROFILE',
    'output_dir': '$LOG_DIR',
    'pipeline': 'remote_train_bootstrap.sh -> train_renderer.py + pipeline.py compress',
    'cublas_workspace_config': '$CUBLAS_WORKSPACE_CONFIG',
}
with open('$PROVENANCE', 'w') as f:
    json.dump(prov, f, indent=2)
print('provenance:', json.dumps(prov, indent=2))
"

    # Heartbeat: 60s cadence, profile cookie ⇒ never `pgrep -f` self-matches.
    ( while true; do
        GPU=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>&1 | tr '\n' ' ')
        echo "[$(date -u +%FT%TZ)] profile=$PROFILE gpu=$GPU" >> "$HEARTBEAT"
        sleep 60
      done ) &
    HB_PID=$!
    trap "kill $HB_PID 2>/dev/null || true" EXIT

    # Stage 1: use container's base Python + install ALL system + Python
    # deps. Hardened against historical failure modes (2026-04-26):
    #   - fresh uv venv pulled wrong-CUDA torch wheels (Error 804) → use
    #     container's /opt/conda/bin/python which has matching torch+CUDA.
    #   - ffprobe/ffmpeg missing → preflight_check + optimize_poses crash.
    #   - timm/einops/segmentation_models_pytorch missing → upstream
    #     modules.py fails to import.
    #   - tac install needs README.md (setuptools readme= field).
    echo "=== STAGE 1 ($PROFILE): system + Python deps ==="
    PYBIN="/opt/conda/bin/python"
    if [ ! -x "$PYBIN" ]; then
        echo "FATAL: no /opt/conda/bin/python — container is not the expected"
        echo "       pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel image."
        exit 1
    fi
    if ! command -v ffprobe >/dev/null 2>&1; then
        DEBIAN_FRONTEND=noninteractive apt-get install -y -qq ffmpeg 2>&1 | tail -2
    fi
    [ -f "$WORKSPACE/README.md" ] || { echo "FATAL: README.md missing — setuptools install will fail. rsync the full repo."; exit 1; }
    "$PYBIN" -m pip install -q -e . 2>&1 | tail -3
    "$PYBIN" -m pip install -q av opencv-python pydantic timm einops segmentation_models_pytorch 2>&1 | tail -3

    # Stage 2: standalone determinism check (CUDA + CUBLAS_WORKSPACE_CONFIG +
    # profile contract). Standalone Python file ⇒ no heredoc quoting issues.
    echo "=== STAGE 2 ($PROFILE): determinism check ==="
    "$PYBIN" "$WORKSPACE/tools/check_determinism.py" "$PROFILE"

    # Stage 3: train_renderer (Phase 1+2+3, produces float .pt checkpoints).
    echo "=== STAGE 3 ($PROFILE): train_renderer ==="
    echo "[$(date -u +%FT%TZ)] launching train_renderer.py $PROFILE" >> "$HEARTBEAT"
    "$PYBIN" -u "$WORKSPACE/src/tac/experiments/train_renderer.py" \
        --profile "$PROFILE" \
        --tag "$PROFILE" \
        --device cuda \
        --output-dir "$OUTPUT_SUBDIR"

    # Stage 4: probe canonical checkpoint (matches deploy_vastai's
    # CANONICAL_CHECKPOINT_NAMES order).
    echo "=== STAGE 4 ($PROFILE): probe canonical checkpoint ==="
    CKPT=""
    for name in distill_phase3_best.pt distill_phase2_best.pt qat_best_float.pt distill_latest.pt; do
        if [ -f "$LOG_DIR/$name" ]; then
            CKPT="$LOG_DIR/$name"
            echo "  using $CKPT"
            break
        fi
    done
    if [ -z "$CKPT" ]; then
        echo "ABORT: no canonical checkpoint found in $LOG_DIR"
        ls -la "$LOG_DIR/"
        exit 2
    fi

    # Stage 5: pipeline.py compress = step_export (FP4 + self-compression
    # via RESIDUAL codebook + robust_scale + stochastic) → step_qat →
    # step_pose_tto → step_archive → step_eval (CUDA auth eval inline).
    echo "=== STAGE 5 ($PROFILE): pipeline.py compress (QAT + self-compress + archive + auth eval) ==="
    echo "[$(date -u +%FT%TZ)] launching pipeline.py compress $PROFILE checkpoint=$CKPT" >> "$HEARTBEAT"
    "$PYBIN" -u "$WORKSPACE/experiments/pipeline.py" compress \
        --profile "$PROFILE" \
        --video "$WORKSPACE/upstream/videos/0.mkv" \
        --checkpoint "$CKPT" \
        --device cuda \
        --output-dir "$OUTPUT_SUBDIR"

    echo "=== ${PROFILE}_DONE_$(date +%s) ==="
    echo "[$(date -u +%FT%TZ)] ${PROFILE}_PIPELINE_DONE" >> "$HEARTBEAT"
    exit 0
fi

# ── Default mode: launch tmux ─────────────────────────────────────────────
if [ $# -lt 1 ]; then
    echo "Usage: $0 <profile_name> [output_subdir]" >&2
    exit 1
fi

PROFILE="$1"
OUTPUT_SUBDIR="${2:-experiments/results/${PROFILE}}"
SESSION="${PROFILE}_train"
LOG_DIR="$WORKSPACE/$OUTPUT_SUBDIR"
TRAIN_LOG="$LOG_DIR/train.log"
HEARTBEAT_LOG="$LOG_DIR/heartbeat.log"

if [ ! -d "$WORKSPACE" ]; then
    echo "FATAL: $WORKSPACE missing — sync repo first." >&2
    exit 1
fi
mkdir -p "$LOG_DIR"

# Kill prior session if any (idempotent).
tmux kill-session -t "$SESSION" 2>/dev/null || true

# Launch the inner runner in tmux. Single-quoted so bash doesn't munge.
SCRIPT_PATH="$WORKSPACE/scripts/remote_train_bootstrap.sh"
tmux new-session -d -s "$SESSION" \
    "bash '$SCRIPT_PATH' --inner '$PROFILE' '$OUTPUT_SUBDIR' 2>&1 | tee -a '$TRAIN_LOG'"

sleep 3
echo "tmux session '$SESSION' status:"
tmux ls 2>&1 | grep "$SESSION" || echo "  (NOT FOUND — check $TRAIN_LOG)"
echo
echo "First log lines (after 5s):"
sleep 5
tail -20 "$TRAIN_LOG" 2>/dev/null || echo "  (log not yet started)"
echo
echo "DEPLOY_OK profile=$PROFILE session=$SESSION log=$TRAIN_LOG heartbeat=$HEARTBEAT_LOG"
