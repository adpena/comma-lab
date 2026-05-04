#!/usr/bin/env bash
set -euo pipefail

WORKSPACE="${WORKSPACE:-/workspace/pact}"
PYBIN="${PYBIN:-/opt/conda/bin/python}"
cd "$WORKSPACE"

OUT="experiments/results/qfaithful_geometry_closed_candidates_20260502/h100_diag_2146_pr64_qp1_direct_zoom_v2_20260502T0700Z"
ARCHIVE_REL="experiments/results/qfaithful_geometry_closed_candidates_20260502/2146_qzs3_pr64_qp1_direct_zoom_v2/archive.zip"
mkdir -p "$OUT"
exec > >(tee -a "$OUT/driver.log") 2>&1

echo "[qfaithful-h100] start_utc=$(date -u +%FT%TZ)"
echo "[qfaithful-h100] archive=$ARCHIVE_REL"

export WORKSPACE
export ARCHIVE_PATH="$WORKSPACE/$ARCHIVE_REL"
export ARCHIVE_LABEL="qfaithful_2146_pr64_qp1_direct_zoom_v2_h100_diag"
export LOG_DIR="$WORKSPACE/$OUT"
export PREDICTED_LOW="0.30"
export PREDICTED_HIGH="25.0"
export CONTROLLED_BASELINE="C-063 A++ frontier 0.3156230307844823; prior Q-FAITHFUL no-zoom snapshots collapsed"
export KEEP_EVAL_WORK="0"

bash scripts/remote_archive_only_eval.sh

export PYTHONPATH="$WORKSPACE/src:$WORKSPACE/upstream:$WORKSPACE:${PYTHONPATH:-}"
export UV_PROJECT_ENVIRONMENT="$WORKSPACE/$OUT/component_trace_uv_env"
echo "[qfaithful-h100] component_trace_start_utc=$(date -u +%FT%TZ)"
"$PYBIN" -u experiments/contest_component_trace.py \
    --archive "$ARCHIVE_PATH" \
    --inflate-sh submissions/robust_current/inflate.sh \
    --upstream-dir upstream \
    --uncompressed-dir upstream/videos \
    --video-names-file upstream/public_test_video_names.txt \
    --device cuda \
    --batch-size 16 \
    --contest-auth-eval-json "$OUT/contest_auth_eval.json" \
    --output-json "$OUT/component_trace.json" \
    --work-dir "$OUT/component_trace_work" \
    --inflate-timeout 1800 \
    2>&1 | tee "$OUT/component_trace.log"

rm -rf "$OUT/component_trace_work/inflated" \
       "$OUT/component_trace_work/extracted" \
       "$OUT/component_trace_work/archive.zip" \
       "$UV_PROJECT_ENVIRONMENT"

date -u +%FT%TZ > "$OUT/done_utc.txt"
echo "[qfaithful-h100] done_utc=$(cat "$OUT/done_utc.txt")"
