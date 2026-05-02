#!/bin/bash
# Fast-chip diagnostic exact CUDA eval for the C-059 mixed-local MQZ1 renderer.
# All bootstrap/eval custody lives in remote_archive_only_eval.sh.
# Provenance: provenance.json + heartbeat + run_record are written by
# remote_archive_only_eval.sh (canonical bootstrap; emits provenance.json under
# $LOG_DIR, satisfies preflight check_remote_scripts_write_provenance via
# delegation per feedback_canonical_remote_bootstraps).

set -euo pipefail

export ARCHIVE_PATH="${ARCHIVE_PATH:-/workspace/pact/experiments/results/mixed_local_qzs_block_allocation_20260502/mixed_64_frame1_head_32_frame2_head_32_pose_mlp_32/archive.zip}"
export ARCHIVE_LABEL="${ARCHIVE_LABEL:-h100_eval_mixed_qzs_c059_20260502}"
export PREDICTED_LOW="${PREDICTED_LOW:-0.315}"
export PREDICTED_HIGH="${PREDICTED_HIGH:-0.317}"
export PREDICTED_BAND="${PREDICTED_BAND:-[$PREDICTED_LOW, $PREDICTED_HIGH]}"  # predicted_band [LOW, HIGH] for council prediction calibration; consumed by remote_archive_only_eval.sh and written into provenance.json
export CONTROLLED_BASELINE="${CONTROLLED_BASELINE:-C-059 A++ exact T4 frontier score=0.3157055307844823 sha=cf44aa7fdb13b9ab6236aefde1f5f58e915d7dfa99128246235443841396c6ab bytes=276347; this candidate replaces the packed renderer with charged MQZ1 mixed block sizes and must preserve component distances to convert its 252-byte rate win.}"
export LOG_DIR="${LOG_DIR:-/workspace/pact/experiments/results/archive_only_eval_mixed_qzs_c059_20260502}"

exec bash scripts/remote_archive_only_eval.sh
