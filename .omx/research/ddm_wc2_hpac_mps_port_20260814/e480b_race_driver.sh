#!/bin/bash
# rx2 identity race, e480b MPS endpoint (declared inputs; wc2 §5e chain).
# Stages sequential, fail-closed on first non-zero rc.
set -euo pipefail
cd /Users/adpena/Projects/pact
E480B=/Volumes/APDataStore/pact/ddm_rx2_current_mc36_label_hpac/gpu_race/full_e480b
ARGS=(
  --checkpoint "$E480B/checkpoints/full_mps_e480.checkpoints/qat_stage_end_epoch_0480.pt"
  --terminal-epoch 480
  --training-report "$E480B/reports/trainer.json"
  --training-manifest "$E480B/checkpoints/full_mps_e480.artifacts.json"
  --torch-threads 4
)
for stage in preflight prepare export-base fit materialize encode build cpu-decode finalize; do
  echo "=== RACE STAGE: $stage ($(date -u +%H:%M:%SZ)) ==="
  .venv/bin/python experiments/ddm_rx2_mc36_identity_race.py "$stage" "${ARGS[@]}"
done
echo "=== RACE COMPLETE ==="
