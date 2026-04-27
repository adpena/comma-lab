#!/bin/bash
# Canonical remote setup for ANY Vast.ai 4090 experiment lane.
# Captures every trap documented in memory:
#   - feedback_zip_dep_bootstrap_trap (no zip in PyTorch container)
#   - feedback_ffmpeg_svtav1_deploy (system ffmpeg too old, need BtbN n7+ for in_primaries + libsvtav1)
#   - feedback_vastai_nvdec_host_variation (DALI needs NVDEC; probe early)
#   - feedback_dead_flag_wiring_pattern (verify auth_eval_renderer args before invoking)
#   - feedback_default_to_convenience_trap (CUDA-required)
#   - feedback_proxy_auth_math_useless (only contest_auth_eval results count)
#
# Usage on the remote (after SCP'ing this file + the experiment script):
#   bash remote_setup_full.sh && bash <experiment_script>.sh

set -euo pipefail

WORKSPACE=/workspace/pact
PYBIN=/opt/conda/bin/python
log() { echo "[setup] $(date -u +%FT%TZ) $*"; }

log "=== Stage 0: GPU + driver ==="
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader

log "=== Stage 1: apt deps (ffmpeg + zip + unzip) ==="
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq ffmpeg zip unzip 2>&1 | tail -2

log "=== Stage 2: Python deps via /opt/conda + uv ==="
cd "$WORKSPACE"
"$PYBIN" -m ensurepip --upgrade 2>&1 | tail -1
"$PYBIN" -m pip install -q --upgrade pip 2>&1 | tail -1
"$PYBIN" -m pip install -q -e ".[runtime]" 2>&1 | tail -2

log "=== Stage 3: nvidia-dali-cuda120 (for upstream/evaluate.py DaliVideoDataset) ==="
"$PYBIN" -m pip install -q --extra-index-url https://developer.download.nvidia.com/compute/redist nvidia-dali-cuda120 2>&1 | tail -2

log "=== Stage 4: NVDEC probe (must pass — kill instance if not) ==="
# Refactored 2026-04-27: delegate to scripts/probe_nvdec.sh — the canonical
# probe shared with all lane scripts. Single source of truth means the probe
# logic only needs to be tuned in one place.
bash "$WORKSPACE/scripts/probe_nvdec.sh" || {
    log "FATAL: NVDEC probe failed — destroy this instance and pick a different host."
    exit 2
}

log "=== Stage 5: uv + venv + av (for inflate_renderer.py) ==="
curl -LsSf https://astral.sh/uv/install.sh | sh 2>&1 | tail -1
export PATH="$HOME/.local/bin:$PATH"
cd "$WORKSPACE"
# Idempotent: only create venv if it doesn't exist yet (re-running setup
# after a partial failure would otherwise hit `--clear required` error).
if [ ! -d "$WORKSPACE/.venv" ]; then
    uv venv 2>&1 | tail -1
fi
uv pip install av 2>&1 | tail -1

log "=== Stage 6: ffmpeg-master (BtbN nightly with in_primaries + libsvtav1) ==="
# System ffmpeg 4.4.2 lacks both. Required for build_baseline_archive (svtav1) AND inflate.sh (in_primaries).
curl -sL https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-linux64-gpl.tar.xz \
    -o /tmp/ffmpeg-master.tar.xz
cd /tmp && tar xf ffmpeg-master.tar.xz
FFMPEG_NEW=/tmp/ffmpeg-master-latest-linux64-gpl/bin/ffmpeg
# IMPORTANT: `set -euo pipefail` + `grep -q` interact badly — grep -q exits
# 0 on first match, closing the pipe, which sends SIGPIPE to ffmpeg, which
# exits non-zero, which pipefail propagates as a failed pipeline. Workaround:
# capture full ffmpeg output first, then grep without pipefail risk.
SCALE_HELP=$("$FFMPEG_NEW" -hide_banner -h filter=scale 2>&1)
if ! echo "$SCALE_HELP" | grep -q "in_primaries"; then
    echo "FATAL: ffmpeg-master lacks in_primaries — wrong build" >&2
    exit 1
fi
ENCODERS=$("$FFMPEG_NEW" -encoders 2>&1)
if ! echo "$ENCODERS" | grep -qi svtav1; then
    echo "FATAL: ffmpeg-master lacks libsvtav1 — wrong build" >&2
    exit 1
fi
log "  ffmpeg-master OK at $FFMPEG_NEW"

log "=== Stage 7: env exports for downstream scripts ==="
cat > "$WORKSPACE/env.sh" <<EOF
export PATH=\$HOME/.local/bin:\$PATH
export PYTHONPATH=src:upstream:\$PWD
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TAC_FFMPEG=$FFMPEG_NEW
export FFMPEG_BIN=$FFMPEG_NEW
EOF
log "  env.sh written — source before running experiments"

log "=== SETUP_COMPLETE ==="
