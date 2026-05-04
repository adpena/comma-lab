#!/usr/bin/env bash
set -euo pipefail

WORKSPACE="${WORKSPACE:-/workspace/pact}"
PYBIN="${PYBIN:-/opt/conda/bin/python}"
cd "$WORKSPACE"

OUT="experiments/results/pr67_same_h100_component_trace_20260502T0712Z"
SOURCE_DIR="experiments/results/archive_eval_pr67_public_qpose14_qzs3_filmq9g_slsb1_r55_20260502T0213Z"
ARCHIVE_REL="$SOURCE_DIR/archive.zip"
AUTH_REL="$SOURCE_DIR/contest_auth_eval.json"
mkdir -p "$OUT"
exec > >(tee -a "$OUT/driver.log") 2>&1

echo "[pr67-h100-trace] start_utc=$(date -u +%FT%TZ)"
echo "[pr67-h100-trace] archive=$ARCHIVE_REL"
sha256sum "$ARCHIVE_REL" "$AUTH_REL" > "$OUT/source_sha256.txt"
cp "$AUTH_REL" "$OUT/source_contest_auth_eval.json"

export PYTHONPATH="$WORKSPACE/src:$WORKSPACE/upstream:$WORKSPACE:${PYTHONPATH:-}"
export UV_PROJECT_ENVIRONMENT="$WORKSPACE/$OUT/component_trace_uv_env"
"$PYBIN" -u experiments/contest_component_trace.py \
    --archive "$ARCHIVE_REL" \
    --inflate-sh submissions/robust_current/inflate.sh \
    --upstream-dir upstream \
    --uncompressed-dir upstream/videos \
    --video-names-file upstream/public_test_video_names.txt \
    --device cuda \
    --batch-size 16 \
    --contest-auth-eval-json "$AUTH_REL" \
    --output-json "$OUT/component_trace.json" \
    --work-dir "$OUT/component_trace_work" \
    --inflate-timeout 1800 \
    2>&1 | tee "$OUT/component_trace.log"

rm -rf "$OUT/component_trace_work/inflated" \
       "$OUT/component_trace_work/extracted" \
       "$OUT/component_trace_work/archive.zip" \
       "$UV_PROJECT_ENVIRONMENT"

date -u +%FT%TZ > "$OUT/done_utc.txt"
echo "[pr67-h100-trace] done_utc=$(cat "$OUT/done_utc.txt")"
