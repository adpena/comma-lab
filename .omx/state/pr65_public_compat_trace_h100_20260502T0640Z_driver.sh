#!/usr/bin/env bash
set -euo pipefail

cd /workspace/pact

OUT_DIR="experiments/results/public_pr65_torch25_compat_trace_20260502T0640Z"
ADAPTER="experiments/results/public_leaderboard_inflate_adapters_20260502T0630Z/pr65_torch25_compat/inflate.sh"
ARCHIVE="reports/raw/leaderboard_intel_20260501/pr65_archive.zip"
mkdir -p "$OUT_DIR"

{
  echo "[pr65-compat-trace] $(date -u +%FT%TZ) start"
  ARCHIVE_PATH="$ARCHIVE" \
  ARCHIVE_LABEL="public_pr65_torch25_compat_reverse_engineering_trace" \
  INFLATE_SH="$ADAPTER" \
  LOG_DIR="$OUT_DIR" \
  PREDICTED_LOW="0.30" \
  PREDICTED_HIGH="0.40" \
  CONTROLLED_BASELINE="external PR65 compatibility replay; not exact public-submission faithfulness evidence" \
  KEEP_EVAL_WORK=0 \
    bash scripts/remote_archive_only_eval.sh
  echo "[pr65-compat-trace] $(date -u +%FT%TZ) done"
} 2>&1 | tee "$OUT_DIR/driver.log"
