#!/bin/bash
# s1a stage-A seed-2 RESUME driver (r8) — continues the r7 run killed by an
# external SIGKILL at 2026-08-25T18:40:23Z (~ep30 of seed 20260816).
# Resumes from wd3_epoch_0030.pt via the masked-identity cure (a5fd9ace0b).
# Sealed MAIN_LAUNCH_ORDER sha 708eae6a...; seed-1 (20260815) completed rc=0 PASS.
set -uo pipefail
cd /Users/adpena/Projects/pact
export PYTORCH_ENABLE_MPS_FALLBACK=0
export PYTHONDONTWRITEBYTECODE=1
echo "=== OFF seed 20260816 RESUME(ep30) start $(date -u +%FT%TZ) ==="
.venv/bin/python experiments/ddm_wd3_scorer_aware_width_distillation.py train \
  --compiled-config "/Volumes/APDataStore/pact/ddm_s1a_stage_a_adapter/training/off_seed_20260816/COMPILED_CONFIG_resume_ep30.json"
rc=$?
echo "=== OFF seed 20260816 RESUME rc=${rc} end $(date -u +%FT%TZ) ==="
if [ "${rc}" -ne 0 ]; then exit "${rc}"; fi
echo "BOTH_OFF_SEEDS_COMPLETE"
