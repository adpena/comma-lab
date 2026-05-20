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

# === Catalog #204 cross-driver expansion (2026-05-19) + WAVE-3-HARDEN-1 META extension (2026-05-20) ===
# OUTPUT_DIR under $WORKSPACE = /workspace/pact/mps_gap_results resolves to
# /tmp/pact/mps_gap_results on Modal workers (Modal mounts working tree under
# /tmp/pact/, NOT /workspace/pact/). The dispatch tool experiments/mps_gap_
# experiment_a10g_dispatch.py does NOT enforce the /tmp guard, but the symptom
# IS the same bug class: output written to /tmp/pact/... is NOT synced back to
# the local repo by Modal's harvest pattern. Silent data loss.
# Fix: redirect to /modal_results/<INSTANCE_JOB_ID>/output/ (durable Modal
# volume; modal_train_lane.py harvests it back to local repo at completion).
# Sister of master_gradient_fec6_modal_t4_cuda_anchor + master_gradient_fec6_
# modal_cpu + stack_of_stacks + stc_v2 + a1_plus_lapose driver fixes per the
# canonical Catalog #204 3-branch pattern.
# Override placed AFTER MPS_GAP_OUTPUT_DIR resolution; INSTANCE_JOB_ID is the
# canonical dispatch-id env var threaded by modal_train_lane.py.
if [ "${MODAL_RUNTIME:-0}" = "1" ] && [ -d "/modal_results" ] && [ -n "${INSTANCE_JOB_ID:-}" ]; then
    MPS_GAP_OUTPUT_DIR="/modal_results/${INSTANCE_JOB_ID}/output/mps_gap_results"
fi

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

# Stage 1.5: canonical bootstrap per CLAUDE.md "Forbidden re-implementing remote bootstrap
# inline" + Catalog #163. The Modal worker image has python at /usr/bin/python but tac runtime
# requires the uv-managed .venv with torch+pyav+brotli pinned. The canonical
# bootstrap_runtime_deps() function in remote_archive_only_eval.sh installs uv, pins torch
# to the correct CUDA wheel (driver-version-aware per CLAUDE.md "Forbidden uv torch install
# without driver-version pin"), and exports PYBIN.
if [ ! -f "$WORKSPACE/scripts/remote_archive_only_eval.sh" ]; then
    echo "[FATAL] canonical bootstrap script missing at scripts/remote_archive_only_eval.sh"
    exit 22
fi
# shellcheck source=/dev/null
REMOTE_ARCHIVE_ONLY_EVAL_SOURCE_ONLY=1 source "$WORKSPACE/scripts/remote_archive_only_eval.sh"
if declare -f bootstrap_runtime_deps > /dev/null 2>&1; then
    bootstrap_runtime_deps
else
    echo "[FATAL] bootstrap_runtime_deps function not found after sourcing"
    exit 23
fi
if [ -z "${PYBIN:-}" ] || [ ! -x "$PYBIN" ]; then
    echo "[FATAL] PYBIN not set or not executable after bootstrap (PYBIN=${PYBIN:-unset})"
    exit 24
fi

# Stage 2: target-device forward + components emit (use $PYBIN from bootstrap, NOT .venv/bin/python)
"$PYBIN" experiments/mps_gap_experiment_a10g_dispatch.py \
    --video-path "$MPS_GAP_VIDEO_PATH" \
    --checkpoint-input "$MPS_GAP_CHECKPOINT_INPUT" \
    --frame-cache-input "$MPS_GAP_FRAME_CACHE_INPUT" \
    --output-dir "$MPS_GAP_OUTPUT_DIR" \
    --target-device cuda \
    $SCORER_FLAG

echo "[mps_gap_experiment] DONE [diagnostic-CUDA Modal A10G]"
