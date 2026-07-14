#!/bin/zsh
set -euo pipefail

ROOT=/Users/adpena/Projects/pact
OUT_ROOT="${PACT_THROUGHPUT_MATH_OUT:-$ROOT/experiments/results/throughput_frontier_math_20260714}"
# Deliberately point at fresh, content-current receipt names.  Older resumable
# receipts were produced by different source bytes and must fail closed rather
# than being silently treated as Task #494 authority.
FIXEDPOINT="${PACT_FIXEDPOINT_RECEIPT:-$ROOT/experiments/results/throughput_authority_ladder_20260714/fixedpoint_scorer_forward_n600_current.json}"
FULL_R="${PACT_FULL_R_RECEIPT:-$ROOT/experiments/results/throughput_authority_ladder_20260714/full_r_adjoint_n600_current.json}"
VERDICT="$ROOT/.omx/research/frozen_scorer_verdict_wallclock_n96_20260714.json"
PYTHAGOREAN="$ROOT/.omx/research/pythagorean_exact_arithmetic_bitident_probe_20260713.json"
TILE="$ROOT/experiments/results/cheapen_real95_tilehalo_fp16_20260713/tile_halo_receipt.json"
SPARSE="$ROOT/experiments/results/p0_sparse_adjoint_costate_vjp_20260713/measurement_receipt.json"

cd "$ROOT"

exec env \
  PYTHONHASHSEED=0 \
  OMP_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python tools/probe_throughput_frontier_math.py \
    --fixedpoint-receipt "$FIXEDPOINT" \
    --full-r-receipt "$FULL_R" \
    --tile-halo-receipt "$TILE" \
    --sparse-adjoint-receipt "$SPARSE" \
    --n-pairs 600 \
    --output-root "$OUT_ROOT" \
    --resume
