#!/usr/bin/env bash
# Score experiments in sequence, waiting for each to finish.
# Usage: nohup bash experiments/run_scorer_queue.sh > experiments/scorer_queue.log 2>&1 &
set -euo pipefail

cd "$(dirname "$0")/.."
REPORT_DIR="reports/raw/2026-04-06-av1-roi-experiments"

score_experiment() {
  local name="$1"
  local archive="$2"
  local report="$3"
  echo "=== Scoring $name ($(wc -c < "$archive") bytes) ==="
  cp "$archive" submissions/robust_current/archive.zip
  uv run python -m comma_lab.cli eval-submission robust_current --device cpu --report-copy "$report" 2>&1
  echo "=== Done: $name ==="
  echo ""
}

# Queue: score each experiment's saved archive in sequence
score_experiment "Exp-J" "$REPORT_DIR/exp_j_archive.zip" "$REPORT_DIR/exp_j_report.txt"
score_experiment "Exp-I" "$REPORT_DIR/exp_i_archive.zip" "$REPORT_DIR/exp_i_report.txt"
score_experiment "Exp-D" "$REPORT_DIR/exp_d_archive.zip" "$REPORT_DIR/exp_d_report.txt"

echo "All scoring complete!"
