#!/bin/bash
set -uo pipefail
R=".omx/research/ddm_m1_20260808"
T="/Volumes/VertigoDataTier/pact"
COMMON=(--lr 2e-07 --ce-fraction 0.0 --softplus-fraction -999.0 --bits 4
  --microbatch-pairs 4 --microbatch-policy auto --microbatch-hygiene per-step
  --microbatch-chunk-cache --verdict-batch-size 32
  --input-cache "$T/ddm_hb1_20260806/inputs/gt_seg_cache.pt"
  --target-cache "$T/ddm_hb1_20260806/inputs/gt_seg_cache.pt"
  --init "$T/pr130_eureka_intake_20260806/repro_repo/artifacts/checkpoints/semantic_renderer_w96_b4_qat4_12k.pt"
  --launch-ticket-path "$R/launch_ticket_v5_event_driven.json")
for i in 1 2 3 4 5; do
  .venv/bin/python tools/safe_run.py --rss-mb 90000 --timeout 3600 \
    --projected-gib 16.0 --label "ddm_m1_sigma_fp16_run${i}" -- \
    .venv/bin/python experiments/ddm_mx1_pr130_semantic_renderer.py \
      --mode mlx-train --device gpu --pairs 120 --steps 5 --seed 20260808 \
      --train-compute-dtype fp16 --checkpoint-every 5 --eval-every 5 \
      "${COMMON[@]}" \
      --run-dir "$R/sigma/run_${i}" --out "$R/sigma/run_${i}/result.json" || { echo "SIGMA_RUN_${i}_FAILED"; exit 6; }
done
.venv/bin/python tools/safe_run.py --rss-mb 90000 --timeout 3600 \
  --projected-gib 16.0 --label "ddm_m1_sigma_fp32_ref" -- \
  .venv/bin/python experiments/ddm_mx1_pr130_semantic_renderer.py \
    --mode mlx-train --device gpu --pairs 120 --steps 5 --seed 20260808 \
    --train-compute-dtype fp32 --checkpoint-every 5 --eval-every 5 \
    "${COMMON[@]}" \
    --run-dir "$R/sigma/run_fp32" --out "$R/sigma/run_fp32/result.json" || { echo "SIGMA_FP32_FAILED"; exit 7; }
echo "SIGMA_LEG_COMPLETE"
