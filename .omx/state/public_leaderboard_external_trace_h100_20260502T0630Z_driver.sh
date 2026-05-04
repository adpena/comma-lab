#!/usr/bin/env bash
set -euo pipefail

cd /workspace/pact

OUT_ROOT="experiments/results/public_leaderboard_external_trace_20260502T0630Z"
ADAPTER_ROOT="experiments/results/public_leaderboard_inflate_adapters_20260502T0630Z"
mkdir -p "$OUT_ROOT"

run_one() {
  local label="$1"
  local archive="$2"
  local inflate_sh="$3"
  local low="$4"
  local high="$5"
  local log_dir="$OUT_ROOT/$label"
  mkdir -p "$log_dir"
  {
    echo "[public-trace] $(date -u +%FT%TZ) start label=$label archive=$archive inflate_sh=$inflate_sh"
    ARCHIVE_PATH="$archive" \
    ARCHIVE_LABEL="public_${label}_external_faithful_trace" \
    INFLATE_SH="$inflate_sh" \
    LOG_DIR="$log_dir" \
    PREDICTED_LOW="$low" \
    PREDICTED_HIGH="$high" \
    CONTROLLED_BASELINE="external public leaderboard archive; not an own-score claim" \
    KEEP_EVAL_WORK=0 \
      bash scripts/remote_archive_only_eval.sh
    echo "[public-trace] $(date -u +%FT%TZ) done label=$label"
    if [ -f "$log_dir/contest_auth_eval.json" ]; then
      /opt/conda/bin/python - "$log_dir/contest_auth_eval.json" <<'PY'
import json
import sys
path = sys.argv[1]
data = json.load(open(path))
summary = {
    "score": data.get("score_recomputed_from_components"),
    "bytes": data.get("archive_size_bytes"),
    "sha": data.get("archive_sha256"),
    "pose": data.get("avg_posenet_dist"),
    "seg": data.get("avg_segnet_dist"),
    "samples": data.get("num_samples"),
    "device": data.get("device"),
}
print("[public-trace-summary]", json.dumps(summary, sort_keys=True))
PY
    fi
  } 2>&1 | tee "$log_dir/driver.log"
}

run_one \
  "pr65" \
  "reports/raw/leaderboard_intel_20260501/pr65_archive.zip" \
  "$ADAPTER_ROOT/pr65/inflate.sh" \
  "0.30" \
  "0.35" || true

run_one \
  "pr67" \
  "reports/raw/leaderboard_intel_20260501/pr67_archive.zip" \
  "$ADAPTER_ROOT/pr67/inflate.sh" \
  "0.30" \
  "0.35" || true

echo "[public-trace] $(date -u +%FT%TZ) all done"
