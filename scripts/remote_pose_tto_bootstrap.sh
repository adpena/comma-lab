#!/bin/bash
# Canonical remote bootstrap for FULL post-training stack on an existing
# float checkpoint (.pt). Runs pipeline.py compress which chains:
#   step_export    — FP4 + self-compression (RESIDUAL codebook + robust_scale)
#   step_qat       — QAT fine-tune to recover any FP4-induced quality loss
#   step_pose_tto  — gradient-descent pose TTO at compress time
#   step_archive   — canonical archive build (validates contents)
#   step_eval      — CUDA auth eval (the only valid measurement per
#                    CLAUDE.md MPS-NOISE rule)
#
# Usage (on remote, after rsync of code + checkpoint.pt + masks.mkv):
#   bash scripts/remote_pose_tto_bootstrap.sh <checkpoint_pt> <masks_mkv> <profile> [output_subdir]
#
# Example:
#   bash scripts/remote_pose_tto_bootstrap.sh \
#       /workspace/pact/uploads/distill_phase3_best.pt \
#       /workspace/pact/uploads/masks_av1mono_full_crf50.mkv \
#       shiraz
#
# Two-mode pattern matches remote_train_bootstrap.sh: default launches tmux,
# --inner does the actual work. Heartbeat to <output_dir>/heartbeat.log.
#
# Per CLAUDE.md "Canonical Pipeline Standard": this is the ONLY way to run
# post-training + auth eval on remote. Ad-hoc /tmp scripts are blocked by
# preflight.

set -euo pipefail

WORKSPACE=/workspace/pact

# ── --inner mode (the actual work) ────────────────────────────────────────
if [ "${1:-}" = "--inner" ]; then
    CHECKPOINT="$2"
    MASKS="$3"
    PROFILE="$4"
    OUTPUT_SUBDIR="$5"
    cd "$WORKSPACE"

    PYBIN="/opt/conda/bin/python"
    if [ ! -x "$PYBIN" ]; then
        echo "FATAL: missing /opt/conda/bin/python — wrong container image"; exit 1
    fi
    export PATH=/root/.local/bin:$PATH
    export PYTHONPATH=src:upstream:$PWD
    export TAC_UPSTREAM_DIR=/workspace/pact/upstream
    export PYTHONUNBUFFERED=1
    # Deterministic reproducibility — REQUIRED before any cuBLAS call.
    export CUBLAS_WORKSPACE_CONFIG=:4096:8

    # NVDEC probe (2026-04-27) — BEFORE any GPU spend / nvidia-smi telemetry.
    # --ensure-dali auto-installs nvidia-dali-cuda120 if missing (fresh
    # hosts haven't run remote_setup_full.sh yet, so DALI may not be there).
    # Codex R5-3-round-4 fix: closes the probe-before-DALI gap.
    # Reference: feedback_vastai_nvdec_host_variation memory entry.
    bash "$WORKSPACE/scripts/probe_nvdec.sh" --ensure-dali || {
        echo "[pose_tto_bootstrap] FATAL: NVDEC probe failed — destroy this Vast.ai instance" >&2
        exit 2
    }

    LOG_DIR="$WORKSPACE/$OUTPUT_SUBDIR"
    HEARTBEAT="$LOG_DIR/heartbeat.log"
    PROVENANCE="$LOG_DIR/provenance.json"
    mkdir -p "$LOG_DIR"

    # Provenance record (CLAUDE.md canonical pipeline standard: full
    # provenance per run — git hash, GPU, torch version, profile, timestamps).
    GIT_HASH=$(cd "$WORKSPACE" && git rev-parse HEAD 2>/dev/null || echo "no-git")
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>&1 | head -1)
    DRIVER_VER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>&1 | head -1)
    "$PYBIN" -c "
import json, time, torch, sys
prov = {
    'started_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    'git_hash': '$GIT_HASH',
    'gpu_name': '$GPU_NAME',
    'driver_version': '$DRIVER_VER',
    'torch_version': torch.__version__,
    'cuda_version': torch.version.cuda,
    'cuda_available': torch.cuda.is_available(),
    'profile': '$PROFILE',
    'checkpoint': '$CHECKPOINT',
    'masks': '$MASKS',
    'output_dir': '$LOG_DIR',
    'pipeline': 'remote_pose_tto_bootstrap.sh -> pipeline.py compress',
    'cublas_workspace_config': '$CUBLAS_WORKSPACE_CONFIG'.strip() or None,
}
with open('$PROVENANCE', 'w') as f:
    json.dump(prov, f, indent=2)
print('provenance written:', '$PROVENANCE')
"

    # Heartbeat: 60s, unique cookie ⇒ no `pgrep -f` self-match.
    ( while true; do
        GPU=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>&1 | tr '\n' ' ')
        echo "[$(date -u +%FT%TZ)] post_train_$PROFILE gpu=$GPU" >> "$HEARTBEAT"
        sleep 60
      done ) &
    HB_PID=$!
    trap "kill $HB_PID 2>/dev/null || true" EXIT

    # Stage 1: install system + Python deps. Hardened against historical
    # failure modes (2026-04-26):
    #   - ffprobe/ffmpeg missing on the base pytorch container → optimize_poses
    #     preflight crashes with FileNotFoundError. Install via apt.
    #   - missing timm/einops/segmentation_models_pytorch → upstream modules.py
    #     fails to import.
    #   - tac install needs README.md (setuptools readme= field) → assert it.
    echo "=== STAGE 1: install system + Python deps ==="
    if ! command -v ffprobe >/dev/null 2>&1; then
        DEBIAN_FRONTEND=noninteractive apt-get install -y -qq ffmpeg 2>&1 | tail -2
    fi
    [ -f "$WORKSPACE/README.md" ] || { echo "FATAL: README.md missing — setuptools install will fail. rsync the full repo."; exit 1; }
    # 2026-04-26: canonical via pyproject.toml [project.optional-dependencies.runtime]
    "$PYBIN" -m ensurepip --upgrade 2>&1 | tail -1
    "$PYBIN" -m pip install -q --upgrade pip 2>&1 | tail -1
    "$PYBIN" -m pip install -q -e ".[runtime]" 2>&1 | tail -3

    # Stage 2: determinism check (CUDA + CUBLAS_WORKSPACE_CONFIG + profile).
    echo "=== STAGE 2: determinism ==="
    "$PYBIN" "$WORKSPACE/tools/check_determinism.py" "$PROFILE"

    # Stage 3: full post-training stack via pipeline.py compress.
    # Chains: step_export (FP4 + RESIDUAL self-compress + robust_scale +
    # stochastic) → step_qat (recover FP4 quality loss) → step_pose_tto
    # (gradient pose TTO at compress time) → step_archive (canonical archive
    # build with manifest validation) → step_eval (CUDA auth eval).
    echo "=== STAGE 3: pipeline.py compress (FULL post-training + auth eval) ==="
    echo "[$(date -u +%FT%TZ)] launching pipeline.py compress profile=$PROFILE" >> "$HEARTBEAT"
    # batch-pairs=8 (default 16 OOMs on 4090 24GB with EfficientNet-B2 SegNet
    # in the gradient graph — 22.5GB allocated, 41MB free at crash on
    # 2026-04-26 SHIRAZ). 8 is the safe ceiling for this hardware.
    # PYTORCH_CUDA_ALLOC_CONF=expandable_segments helps with fragmentation.
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    "$PYBIN" -u "$WORKSPACE/experiments/pipeline.py" compress \
        --profile "$PROFILE" \
        --video "$WORKSPACE/upstream/videos/0.mkv" \
        --checkpoint "$CHECKPOINT" \
        --masks "$MASKS" \
        --device cuda \
        --output-dir "$OUTPUT_SUBDIR" \
        --pose-batch-pairs 8 \
        2>&1 | tee "$LOG_DIR/pipeline.log"

    # Stage 4: snapshot final results into a record file the human can keep.
    echo "=== STAGE 4: write run record ==="
    "$PYBIN" -c "
import json, time, glob
from pathlib import Path
record = {
    'finished_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
}
# Try to extract auth eval result from the JSON the pipeline.eval writes.
for cand in ['$LOG_DIR/eval/results.json', '$LOG_DIR/results.json',
             '$LOG_DIR/auth_eval_results.json']:
    if Path(cand).exists():
        record['auth_eval'] = json.loads(Path(cand).read_text())
        record['auth_eval_path'] = cand
        break
# List all artifacts produced.
record['artifacts'] = sorted(p.name for p in Path('$LOG_DIR').iterdir() if p.is_file())
with open('$LOG_DIR/run_record.json', 'w') as f:
    json.dump(record, f, indent=2)
print('run_record:', json.dumps(record, indent=2))
"

    echo "=== POST_TRAIN_DONE_$(date +%s) ==="
    echo "[$(date -u +%FT%TZ)] POST_TRAIN_PIPELINE_DONE" >> "$HEARTBEAT"
    exit 0
fi

# ── default mode: launch tmux ─────────────────────────────────────────────
if [ $# -lt 3 ]; then
    echo "Usage: $0 <checkpoint_pt> <masks_mkv> <profile> [output_subdir]" >&2
    exit 1
fi

CHECKPOINT="$1"
MASKS="$2"
PROFILE="$3"
OUTPUT_SUBDIR="${4:-experiments/results/${PROFILE}_pose_tto_$(date +%Y%m%dT%H%M%S)}"
SESSION="${PROFILE}_pose_tto"
LOG_DIR="$WORKSPACE/$OUTPUT_SUBDIR"
TRAIN_LOG="$LOG_DIR/run.log"
mkdir -p "$LOG_DIR"

# Validate inputs exist before forking tmux (cheap safety).
[ -f "$CHECKPOINT" ] || { echo "FATAL: checkpoint $CHECKPOINT not found"; exit 1; }
[ -f "$MASKS" ] || { echo "FATAL: masks $MASKS not found"; exit 1; }

tmux kill-session -t "$SESSION" 2>/dev/null || true
SCRIPT="$WORKSPACE/scripts/remote_pose_tto_bootstrap.sh"
tmux new-session -d -s "$SESSION" \
    "bash '$SCRIPT' --inner '$CHECKPOINT' '$MASKS' '$PROFILE' '$OUTPUT_SUBDIR' 2>&1 | tee -a '$TRAIN_LOG'"

sleep 3
echo "tmux session '$SESSION' status:"
tmux ls 2>&1 | grep "$SESSION" || echo "  (NOT FOUND — check $TRAIN_LOG)"
echo
echo "DEPLOY_OK session=$SESSION log=$TRAIN_LOG"
