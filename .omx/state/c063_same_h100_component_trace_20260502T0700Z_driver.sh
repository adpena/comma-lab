#!/usr/bin/env bash
set -euo pipefail

WORKSPACE="${WORKSPACE:-/workspace/pact}"
PYBIN="${PYBIN:-/opt/conda/bin/python}"
cd "$WORKSPACE"

OUT="experiments/results/c063_same_h100_component_trace_20260502T0700Z"
ARCHIVE_REL="experiments/results/lightning_batch/exact_eval_lossless_repack_c059_brotli_t4_20260502T0537Z/archive.zip"
mkdir -p "$OUT"
exec > >(tee -a "$OUT/driver.log") 2>&1

echo "[c063-h100-trace] start_utc=$(date -u +%FT%TZ)"
echo "[c063-h100-trace] archive=$ARCHIVE_REL"

export WORKSPACE
export ARCHIVE_PATH="$WORKSPACE/$ARCHIVE_REL"
export ARCHIVE_LABEL="c063_lossless_repack_same_h100_trace"
export LOG_DIR="$WORKSPACE/$OUT"
export PREDICTED_LOW="0.314"
export PREDICTED_HIGH="0.317"
export CONTROLLED_BASELINE="C-063 A++ T4 frontier 0.3156230307844823; used for same-H100 allocator calibration"
export KEEP_EVAL_WORK="0"

bash scripts/remote_archive_only_eval.sh

export PYTHONPATH="$WORKSPACE/src:$WORKSPACE/upstream:$WORKSPACE:${PYTHONPATH:-}"
export UV_PROJECT_ENVIRONMENT="$WORKSPACE/$OUT/component_trace_uv_env"
echo "[c063-h100-trace] component_trace_start_utc=$(date -u +%FT%TZ)"
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
echo "[c063-h100-trace] done_utc=$(cat "$OUT/done_utc.txt")"
