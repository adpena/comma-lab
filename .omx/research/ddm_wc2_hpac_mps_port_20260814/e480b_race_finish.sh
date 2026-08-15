#!/bin/bash
# rx2 e480b race, finishing stages: build remaining fit-selected variants,
# then cpu-decode + finalize. (Attempt 2 completed preflight..build[neutral].)
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
for variant in s1p25_c1p0 s1p25_c1p25 s1p25_c0p75 s1p5_c1p25; do
  echo "=== RACE STAGE: materialize --variant $variant ($(date -u +%H:%M:%SZ)) ==="
  .venv/bin/python experiments/ddm_rx2_mc36_identity_race.py materialize --variant "$variant" "${ARGS[@]}"
  echo "=== RACE STAGE: encode --variant $variant ($(date -u +%H:%M:%SZ)) ==="
  .venv/bin/python experiments/ddm_rx2_mc36_identity_race.py encode --variant "$variant" "${ARGS[@]}"
  echo "=== RACE STAGE: build --variant $variant ($(date -u +%H:%M:%SZ)) ==="
  .venv/bin/python experiments/ddm_rx2_mc36_identity_race.py build --variant "$variant" "${ARGS[@]}"
done
for stage in cpu-decode finalize; do
  echo "=== RACE STAGE: $stage ($(date -u +%H:%M:%SZ)) ==="
  .venv/bin/python experiments/ddm_rx2_mc36_identity_race.py "$stage" "${ARGS[@]}"
done
echo "=== RACE COMPLETE ==="
