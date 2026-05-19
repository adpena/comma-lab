#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Remote driver for the Phase B MPS-train CUDA-score gap diagnostic dispatch.
# Operator-routable via tools/operator_authorize.py --recipe
# mps_gap_experiment_tiny_renderer_modal_a10g_dispatch --target modal
#
# This driver runs on the Modal A10G worker. The local-side harness
# (tools/run_mps_gap_experiment.sh) trains the tiny renderer on MPS and
# packages the checkpoint + frame cache; this driver runs the trained
# checkpoint forward on CUDA + emits the target-device components.

set -euo pipefail

# Per Catalog #244: canonical NVML / CUDA env block immediately after
# set -euo pipefail (must appear BEFORE any DALI / nvidia-smi probe). The
# constants mirror tac.deploy.modal.runtime canonical values.
export DALI_DISABLE_NVML=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

WORKSPACE=${WORKSPACE:-/workspace/pact}
cd "$WORKSPACE"

MPS_GAP_VIDEO_PATH=${MPS_GAP_VIDEO_PATH:-${WORKSPACE}/upstream/videos/0.mkv}
MPS_GAP_OUTPUT_DIR=${MPS_GAP_OUTPUT_DIR:-${WORKSPACE}/mps_gap_results}
MPS_GAP_CHECKPOINT_INPUT=${MPS_GAP_CHECKPOINT_INPUT:-${WORKSPACE}/local_mps_checkpoint/checkpoint_ema.pt}
MPS_GAP_FRAME_CACHE_INPUT=${MPS_GAP_FRAME_CACHE_INPUT:-${WORKSPACE}/local_mps_checkpoint/frame_cache.pt}
MPS_GAP_INCLUDE_SCORER=${MPS_GAP_INCLUDE_SCORER:-1}

# Stage 1: required-input validation per Catalog #152 — refuse early before
# the GPU meter starts.
for f in "$MPS_GAP_VIDEO_PATH" "$MPS_GAP_CHECKPOINT_INPUT" "$MPS_GAP_FRAME_CACHE_INPUT"; do
    if [ ! -f "$f" ]; then
        echo "[FATAL] required input file missing: $f"
        echo "Make sure the local MPS training step ran first via tools/run_mps_gap_experiment.sh"
        exit 25
    fi
done

mkdir -p "$MPS_GAP_OUTPUT_DIR"

SCORER_FLAG=""
if [ "$MPS_GAP_INCLUDE_SCORER" = "1" ]; then
    SCORER_FLAG="--include-scorer"
fi

# Stage 2: target-device forward + components emit
.venv/bin/python experiments/mps_gap_experiment_a10g_dispatch.py \
    --video-path "$MPS_GAP_VIDEO_PATH" \
    --checkpoint-input "$MPS_GAP_CHECKPOINT_INPUT" \
    --frame-cache-input "$MPS_GAP_FRAME_CACHE_INPUT" \
    --output-dir "$MPS_GAP_OUTPUT_DIR" \
    --target-device cuda \
    $SCORER_FLAG

echo "[mps_gap_experiment] DONE [diagnostic-CUDA Modal A10G]"
