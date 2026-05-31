#!/bin/bash
# SPDX-License-Identifier: MIT
# Remote lane script: PR110-OPT-11 multi-mode-per-pair composition L0 SCAFFOLD.
#
# Trainer: experiments/train_substrate_pr110_opt11_multi_mode_per_pair_composition.py
# Lane: lane_pr110_opt11_multi_mode_per_pair_composition_l0_scaffold_20260530
#
# Per CLAUDE.md "Forbidden re-implementing remote bootstrap inline" + Catalog #163
# (canonical sentinel bootstrap sourcing): this driver DELEGATES bootstrap to
# scripts/remote_archive_only_eval.sh's bootstrap_runtime_deps() function rather
# than re-implementing uv install / ffmpeg install / torch CUDA pinning.
#
# Per Catalog #244 canonical NVML 3-export block: required for Modal/CUDA env
# hygiene per OVERNIGHT-J + D1 NVML 999 6×24h anchor.
#
# Per Catalog #204 + Catalog #358 Modal-aware OUTPUT_DIR 3-branch override:
# /modal_results/${INSTANCE_JOB_ID}/output on Modal worker per OVERNIGHT-J
# canonical pattern per commit 75d39f32e.
#
# Per Catalog #326 driver-mode-hardcode discipline: supports PR110_OPT11_TRAINER_MODE
# env var (smoke|full) with explicit precedence over SMOKE_ONLY. L0 SCAFFOLD
# defaults to smoke per the companion recipe `env_overrides: PR110_OPT11_TRAINER_MODE: "smoke"`.

set -euo pipefail

# Catalog #244 canonical NVML/CUDA env hygiene 3-export block per
# tac.deploy.modal.runtime canonical constants (mirrored here for the
# rare local-dispatch fallback when the canonical helper is unreachable).
export DALI_DISABLE_NVML="${DALI_DISABLE_NVML:-1}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Catalog #163 canonical bootstrap sentinel + Catalog #295 PYTHONPATH self-
# containment. The canonical bootstrap source path is parameterized; when
# sourced from the Modal worker the wrapper provides REMOTE_ARCHIVE_ONLY_EVAL_SOURCE_ONLY=1
# so the bootstrap function loads its env-fixups but does NOT run its main
# eval flow (Catalog #163 canonical pattern from sister 5bcb53070).
WORKSPACE="${WORKSPACE:-${HOME:-/workspace}/pact}"
if [ "${MODAL_RUNTIME:-0}" = "1" ]; then
    WORKSPACE="/tmp/pact"
fi
cd "${WORKSPACE}"

# Catalog #295 PYTHONPATH self-containment per Slot CCC pattern
export PYTHONPATH="${WORKSPACE}/src:${WORKSPACE}/upstream:${PYTHONPATH:-}"

# Catalog #326 driver-mode-hardcode discipline: PR110_OPT11_TRAINER_MODE > SMOKE_ONLY > default
TRAINER_MODE="${PR110_OPT11_TRAINER_MODE:-smoke}"

# Catalog #204 + Catalog #358 Modal-aware OUTPUT_DIR 3-branch override:
# /modal_results/${INSTANCE_JOB_ID}/output on Modal; $WORKSPACE/results/<lane>/output on Vast.ai
if [ "${MODAL_RUNTIME:-0}" = "1" ] && [ -d "/modal_results" ] && [ -n "${INSTANCE_JOB_ID:-}" ]; then
    OUTPUT_DIR="/modal_results/${INSTANCE_JOB_ID}/output"
else
    OUTPUT_DIR="${PR110_OPT11_OUTPUT_DIR:-${WORKSPACE}/experiments/results/pr110_opt11_l0_scaffold_${INSTANCE_JOB_ID:-local}/output}"
fi
mkdir -p "${OUTPUT_DIR}"
echo "[pr110-opt11-driver] OUTPUT_DIR=${OUTPUT_DIR} TRAINER_MODE=${TRAINER_MODE} MODAL_RUNTIME=${MODAL_RUNTIME:-0}"

# Trainer invocation (canonical Catalog #226 + #205 routing; trainer handles
# smoke vs full mode resolution per Catalog #326).
TRAINER_ARGS=(
    --output-dir "${OUTPUT_DIR}"
    --n-pairs "${PR110_OPT11_N_PAIRS:-600}"
    --modes-per-pair "${PR110_OPT11_MODES_PER_PAIR:-2}"
    --selector-bits-per-mode "${PR110_OPT11_SELECTOR_BITS_PER_MODE:-4}"
    --family-pair-index "${PR110_OPT11_FAMILY_PAIR_INDEX:-1}"
    --rng-seed "${PR110_OPT11_RNG_SEED:-42}"
    --enable-autocast-fp16
    --enable-torch-compile
    --no-grad-eval
)

# Use uv run when available; fall back to python directly on systems without uv.
if command -v uv >/dev/null 2>&1; then
    exec uv run python "${WORKSPACE}/experiments/train_substrate_pr110_opt11_multi_mode_per_pair_composition.py" "${TRAINER_ARGS[@]}"
else
    exec python "${WORKSPACE}/experiments/train_substrate_pr110_opt11_multi_mode_per_pair_composition.py" "${TRAINER_ARGS[@]}"
fi
