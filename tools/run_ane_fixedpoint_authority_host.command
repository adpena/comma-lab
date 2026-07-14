#!/bin/zsh
set -euo pipefail

cd "${0:A:h}/.."
exec .venv/bin/python tools/compile_ane_fixedpoint_authority_ticket.py \
  --qdq-receipt experiments/results/throughput_authority_ladder_20260714/dynamic_fixedpoint_scorer_forward_n600.json \
  --settled-r4-receipt experiments/results/ane_unlock_correction_20260713/r4_variants.json \
  --formulation-id dynamic_exact_absmax_uniform_precision_v1 \
  --output experiments/results/throughput_authority_ladder_20260714/ane_fixedpoint_authority_ticket.json
