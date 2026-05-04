#!/usr/bin/env bash
set -euo pipefail

cd /workspace/pact

OUT_ROOT="experiments/results/public_external_component_trace_20260502T0642Z"
mkdir -p "$OUT_ROOT"

export PATH="/root/.local/bin:$HOME/.local/bin:$PATH"
export PYTHONPATH="/workspace/pact/src:/workspace/pact/upstream:/workspace/pact:${PYTHONPATH:-}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

ensure_parity_ffmpeg() {
  local candidate="${FFMPEG_BIN:-}"
  if [ -z "$candidate" ] && [ -x /workspace/ffmpeg-btbn/bin/ffmpeg ]; then
    candidate="/workspace/ffmpeg-btbn/bin/ffmpeg"
  fi
  if [ -z "$candidate" ]; then
    candidate="$(command -v ffmpeg || true)"
  fi
  local needs_btbn=1
  if [ -n "$candidate" ] && [ -x "$candidate" ]; then
    local scale_help
    scale_help="$("$candidate" -hide_banner -h filter=scale 2>&1 || true)"
    needs_btbn=0
    for opt in in_range out_range in_color_matrix in_primaries in_transfer; do
      if ! printf '%s\n' "$scale_help" | grep -q "$opt"; then
        needs_btbn=1
        break
      fi
    done
  fi
  if [ "$needs_btbn" -eq 1 ]; then
    echo "[component-trace] installing BtbN ffmpeg for color-contract parity"
    curl -fL --retry 5 --retry-delay 3 \
      -o /tmp/ffmpeg-btbn.tar.xz \
      "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-linux64-gpl.tar.xz"
    mkdir -p /workspace/ffmpeg-btbn
    tar -xf /tmp/ffmpeg-btbn.tar.xz -C /workspace/ffmpeg-btbn --strip-components=1
    candidate="/workspace/ffmpeg-btbn/bin/ffmpeg"
  fi
  export FFMPEG_BIN="$candidate"
}

ensure_parity_ffmpeg

run_trace() {
  local label="$1"
  local archive="$2"
  local inflate_sh="$3"
  local auth_json="$4"
  local out_dir="$OUT_ROOT/$label"

  mkdir -p "$out_dir"
  {
    echo "[component-trace] label=$label"
    echo "[component-trace] archive=$archive"
    echo "[component-trace] inflate_sh=$inflate_sh"
    echo "[component-trace] auth_json=$auth_json"
    date -u +%Y-%m-%dT%H:%M:%SZ
  } | tee "$out_dir/run.log"

  /opt/conda/bin/python -u experiments/contest_component_trace.py \
    --archive "$archive" \
    --inflate-sh "$inflate_sh" \
    --upstream-dir upstream \
    --device cuda \
    --top-k 160 \
    --contest-auth-eval-json "$auth_json" \
    --output-json "$out_dir/component_trace.json" \
    --work-dir "$out_dir/component_trace_work" \
    --keep-work-dir \
    2>&1 | tee "$out_dir/component_trace.log"
}

run_trace \
  "pr67_public_adapter" \
  "reports/raw/leaderboard_intel_20260501/pr67_archive.zip" \
  "experiments/results/public_leaderboard_inflate_adapters_20260502T0630Z/pr67/inflate.sh" \
  "experiments/results/public_leaderboard_external_trace_20260502T0630Z/pr67/eval_work/contest_auth_eval.json"

run_trace \
  "pr65_torch25_compat_adapter" \
  "reports/raw/leaderboard_intel_20260501/pr65_archive.zip" \
  "experiments/results/public_leaderboard_inflate_adapters_20260502T0630Z/pr65_torch25_compat/inflate.sh" \
  "experiments/results/public_pr65_torch25_compat_trace_20260502T0640Z/eval_work/contest_auth_eval.json"

date -u +%Y-%m-%dT%H:%M:%SZ > "$OUT_ROOT/done_utc.txt"
