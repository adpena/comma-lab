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
    # Profile names are stored lowercase in tac/profiles.py PROFILES dict
    # ('den', 'shiraz', 'wilde', ...). Operators frequently pass uppercase
    # ('DEN'). Normalize once here so check_determinism.py + train_renderer.py
    # + pipeline.py all see the canonical key. 2026-04-26: DEN-V2 burned
    # 18 min of GPU at $0.30/hr because the call was 'DEN' and PROFILES
    # has 'den' — argparse choices= rejected hard.
    PROFILE=$(echo "$2" | tr '[:upper:]' '[:lower:]')
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
    # DX #6 (2026-04-26): enriched provenance — ffmpeg/svtav1/brotli versions,
    # env vars, sys.argv, and full resolved profile dict so a CUDA reproduction
    # is fully self-contained from this JSON. Mirrors _capture_provenance()
    # in experiments/pipeline.py.
    /opt/conda/bin/python -c "
import json, os, subprocess, sys, time
import torch
def _shell(cmd):
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT, timeout=10).strip()
    except Exception as e:
        return f'<error:{e}>'
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
    'sys_argv': ['$0', '--inner', '$PROFILE', '$OUTPUT_SUBDIR'],
}
ffv = _shell(['ffmpeg', '-version'])
prov['ffmpeg_version'] = ffv.splitlines()[0] if ffv and not ffv.startswith('<error') else ffv
encs = _shell(['ffmpeg', '-encoders'])
svt = [ln.strip() for ln in encs.splitlines() if 'svtav1' in ln.lower() or 'svt-av1' in ln.lower()]
prov['libsvtav1_version'] = svt[0] if svt else None
try:
    import brotli
    prov['brotli_version'] = getattr(brotli, '__version__', 'unknown')
except Exception as e:
    prov['brotli_version_error'] = str(e)
prov['env_vars'] = {k: os.environ.get(k) for k in (
    'LD_LIBRARY_PATH', 'PYTHONPATH', 'CUBLAS_WORKSPACE_CONFIG',
    'PYTORCH_CUDA_ALLOC_CONF', 'TAC_UPSTREAM_DIR', 'PYTHONHASHSEED',
    'TAC_MODELS_DIR', 'INFLATE_TTO',
)}
try:
    sys.path.insert(0, '$WORKSPACE/src')
    from tac.profiles import PROFILES
    if '$PROFILE' in PROFILES:
        prof = PROFILES['$PROFILE']
        def _safe(v):
            try:
                json.dumps(v)
                return v
            except Exception:
                return repr(v)
        prov['profile_dict'] = {k: _safe(v) for k, v in dict(prof).items()}
except Exception as e:
    prov['profile_dict_error'] = str(e)
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
    # 2026-04-26: canonical via pyproject.toml [project.optional-dependencies.runtime]
    # — adding a new dep means editing pyproject.toml, NOT this bootstrap.
    # ensurepip first so pip exists even on a stripped-down container image.
    "$PYBIN" -m ensurepip --upgrade 2>&1 | tail -1
    "$PYBIN" -m pip install -q --upgrade pip 2>&1 | tail -1
    "$PYBIN" -m pip install -q -e ".[runtime]" 2>&1 | tail -3

    # Stage 1b: Make upstream/ffmpeg-new usable. Ubuntu 22.04's apt ffmpeg
    # is 4.4.2 — does NOT support `-svtav1-params` that mask_codec.py
    # requires. The upstream snapshot ships a static ffmpeg-new (n7.1)
    # that does, but it (a) needs +x and (b) needs LD_LIBRARY_PATH to
    # find libSvtAv1Enc.so.2 (also bundled in upstream). 2026-04-26 the
    # DEN-V2 deploy crashed at the mask-encode step because of this —
    # symlinking + LD path makes mask_codec._ffmpeg_binary() pick the
    # right binary automatically.
    FFMPEG_NEW="$WORKSPACE/upstream/ffmpeg-new"
    SVT_LIB_DIR="$WORKSPACE/upstream/submissions/av1_roi_lanczos_unsharp/lib"
    if [ -f "$FFMPEG_NEW" ]; then
        chmod +x "$FFMPEG_NEW" 2>/dev/null || true
        if [ -f "$SVT_LIB_DIR/libSvtAv1Enc.so.2.3.0" ] && [ ! -e "$SVT_LIB_DIR/libSvtAv1Enc.so.2" ]; then
            ln -sf "$SVT_LIB_DIR/libSvtAv1Enc.so.2.3.0" "$SVT_LIB_DIR/libSvtAv1Enc.so.2"
        fi
        export LD_LIBRARY_PATH="$SVT_LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
        # Quick smoke check: invoke ffmpeg-new -version so any future
        # missing-library failure surfaces here, not at Stage 5.
        if "$FFMPEG_NEW" -version >/dev/null 2>&1; then
            echo "OK upstream/ffmpeg-new ready (n7.x with libsvtav1)"
        else
            echo "WARN: upstream/ffmpeg-new present but won't run — mask encoding may fall back to system ffmpeg" >&2
        fi
    fi

    # Stage 2: standalone determinism check (CUDA + CUBLAS_WORKSPACE_CONFIG +
    # profile contract). Standalone Python file ⇒ no heredoc quoting issues.
    echo "=== STAGE 2 ($PROFILE): determinism check ==="
    "$PYBIN" "$WORKSPACE/tools/check_determinism.py" "$PROFILE"

    # Stage 3: train_renderer (Phase 1+2+3, produces float .pt checkpoints).
    # DX #4 (2026-04-26): auto-detect a recent checkpoint and resume. Without
    # this, every spot-instance preemption + relaunch starts from scratch and
    # loses 1-16h of training. `train_renderer.py` already supports
    # `--resume-from`; we just need to find the freshest renderer_*.pt within
    # the last 6h (older than that is almost certainly a stale prior session
    # and resuming risks shape/profile drift). `sort | tail -1` gives the
    # lexicographically last filename which — by tac's epoch-padded naming —
    # is also the latest-trained checkpoint.
    RESUME_ARG=""
    RESUME_CKPT=$(find "$LOG_DIR" -maxdepth 2 -name 'renderer_*.pt' -mmin -360 2>/dev/null | sort | tail -1)
    if [ -n "$RESUME_CKPT" ]; then
        echo "[$(date -u +%FT%TZ)] resume detected: $RESUME_CKPT" >> "$HEARTBEAT"
        echo "Resuming from $RESUME_CKPT"
        RESUME_ARG="--resume-from $RESUME_CKPT"
    fi
    echo "=== STAGE 3 ($PROFILE): train_renderer ==="
    echo "[$(date -u +%FT%TZ)] launching train_renderer.py $PROFILE resume='$RESUME_ARG'" >> "$HEARTBEAT"
    # shellcheck disable=SC2086 # intentional word-split on $RESUME_ARG
    # Note: --no-auth-eval-on-best is passed because Stage 5 (pipeline.py
    # compress) runs its own step_eval against the FULL triple-joint archive
    # (renderer + masks + poses) at the end of the chain. train_renderer's
    # built-in auth-eval-on-best fails loud without --auth-eval-masks AND
    # --auth-eval-poses (Council R3 fix), and at this point in the chain
    # masks.mkv hasn't been encoded yet and poses haven't been TTO-optimized.
    # The honest auth measurement comes from Stage 5, not Stage 3.
    "$PYBIN" -u "$WORKSPACE/src/tac/experiments/train_renderer.py" \
        --profile "$PROFILE" \
        --tag "$PROFILE" \
        --device cuda \
        --output-dir "$OUTPUT_SUBDIR" \
        --no-auth-eval-on-best \
        $RESUME_ARG

    # Stage 4: probe canonical checkpoint via tac.checkpoint_names registry.
    # ONE source of truth shared with deploy_vastai.py. 2026-04-26: DEN
    # deploy crashed because train_renderer emits `renderer_den_best_fp32.pt`
    # but the old list only had `distill_*.pt`. Centralised registry
    # eliminates this class of bug.
    echo "=== STAGE 4 ($PROFILE): probe canonical checkpoint ==="
    CKPT=$("$PYBIN" -c "
import sys; sys.path.insert(0, '$WORKSPACE/src')
from pathlib import Path
from tac.checkpoint_names import canonical_checkpoint_names
log_dir = Path('$LOG_DIR')
for name in canonical_checkpoint_names('$PROFILE'):
    p = log_dir / name
    if p.exists():
        print(p)
        break
")
    if [ -z "$CKPT" ]; then
        echo "ABORT: no canonical checkpoint found in $LOG_DIR"
        ls -la "$LOG_DIR/"
        exit 2
    fi

    # Stage 5: pipeline.py compress = step_export (FP4 + self-compression
    # via RESIDUAL codebook + robust_scale + stochastic) → step_qat →
    # step_pose_tto → step_archive → step_eval (CUDA auth eval inline).
    #
    # Quantizr council #4 CRITICAL (2026-04-26): added --half-frame
    # --binary-poses --brotli — without these, every deploy this week
    # shipped full-frame masks, .pt poses, no Brotli (PipelineConfig
    # defaults are False/False/False). That gave back ~110KB / +0.07
    # rate term to Quantizr for free. Now defaulted ON for production
    # deploys; profiles that want them off can pass --no-half-frame etc.
    echo "=== STAGE 5 ($PROFILE): pipeline.py compress (QAT + self-compress + archive + auth eval) ==="
    echo "[$(date -u +%FT%TZ)] launching pipeline.py compress $PROFILE checkpoint=$CKPT" >> "$HEARTBEAT"
    "$PYBIN" -u "$WORKSPACE/experiments/pipeline.py" compress \
        --profile "$PROFILE" \
        --video "$WORKSPACE/upstream/videos/0.mkv" \
        --checkpoint "$CKPT" \
        --device cuda \
        --half-frame \
        --binary-poses \
        --brotli \
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

# Mirror the inner-mode normalization so the tmux session name + output dir
# are consistent regardless of the operator's case preference.
PROFILE=$(echo "$1" | tr '[:upper:]' '[:lower:]')
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
