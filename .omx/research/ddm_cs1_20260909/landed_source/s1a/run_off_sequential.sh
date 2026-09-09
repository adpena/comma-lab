#!/bin/bash
# s1a stage-A sequential OFF driver (sealed MAIN_LAUNCH_ORDER sha 708eae6a...)
# driver-computes-verdict: rc propagates; each trainer run is itself resumable-from-disk.
set -uo pipefail
cd /Users/adpena/Projects/pact
# env demanded by the trainer's _device guard (PYTORCH_ENABLE_MPS_FALLBACK must be "0")
# + the #1122 AppleDouble/pycache cure for SSD-adjacent runs
export PYTORCH_ENABLE_MPS_FALLBACK=0
export PYTHONDONTWRITEBYTECODE=1
for seed in 20260815 20260816; do
  echo "=== OFF seed ${seed} start $(date -u +%FT%TZ) ==="
  .venv/bin/python experiments/ddm_wd3_scorer_aware_width_distillation.py train \
    --compiled-config "/Volumes/APDataStore/pact/ddm_s1a_stage_a_adapter/training/off_seed_${seed}/COMPILED_CONFIG.json"
  rc=$?
  echo "=== OFF seed ${seed} rc=${rc} end $(date -u +%FT%TZ) ==="
  if [ "${rc}" -ne 0 ]; then exit "${rc}"; fi
done
echo "BOTH_OFF_SEEDS_COMPLETE"
