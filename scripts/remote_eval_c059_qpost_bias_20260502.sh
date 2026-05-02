#!/bin/bash
# Fast-chip diagnostic exact CUDA eval for the C-059 + charged PR65 bias atom.
# This script is intentionally thin: all bootstrap, CUDA, ffmpeg, torch, scorer,
# custody, and eval semantics are owned by remote_archive_only_eval.sh.
# Provenance: provenance.json + heartbeat + run_record are written by
# remote_archive_only_eval.sh (canonical bootstrap; emits provenance.json under
# $LOG_DIR, satisfies preflight check_remote_scripts_write_provenance via
# delegation per feedback_canonical_remote_bootstraps).

set -euo pipefail

export ARCHIVE_PATH="${ARCHIVE_PATH:-/workspace/pact/experiments/results/qzs3_postprocess_c059_bias_20260502/archive.zip}"
export ARCHIVE_LABEL="${ARCHIVE_LABEL:-h100_eval_c059_qpost_bias_20260502}"
export PREDICTED_LOW="${PREDICTED_LOW:-0.315}"
export PREDICTED_HIGH="${PREDICTED_HIGH:-0.318}"
export PREDICTED_BAND="${PREDICTED_BAND:-[$PREDICTED_LOW, $PREDICTED_HIGH]}"  # predicted_band [LOW, HIGH] for council prediction calibration; consumed by remote_archive_only_eval.sh and written into provenance.json
export CONTROLLED_BASELINE="${CONTROLLED_BASELINE:-C-059 A++ exact T4 frontier score=0.3157055307844823 sha=cf44aa7fdb13b9ab6236aefde1f5f58e915d7dfa99128246235443841396c6ab bytes=276347; this candidate adds a charged PR65-derived bias qpost atom and must beat the C-059 rate-adjusted threshold.}"
export LOG_DIR="${LOG_DIR:-/workspace/pact/experiments/results/archive_only_eval_c059_qpost_bias_20260502}"

exec bash scripts/remote_archive_only_eval.sh
