#!/bin/bash
# ddm_ane3 ITEM 1 -- fixed-output SegNet remeasurement.
#
# Each row is its own restart boundary.  A completed JSON receipt is preserved
# and skipped on resume; converted trees remain on Vertigo and every per-pair
# flip ledger remains on APDataStore.
set -euo pipefail

cd /Users/adpena/Projects/pact

PY=.venv_executorch_spike/bin/python
EXP=experiments/ddm_ane2_engineer_precision_drift.py
ENUM=/Volumes/VertigoDataTier/pact/ddm_ane2_precision/enumerate.json
REF=/Volumes/VertigoDataTier/pact/ddm_ane2_precision/reference_generated_n600/reference_segnet.npz
RAW=/Volumes/VertigoDataTier/pact/ddm_afr1_tile48_receiver_identity/identity_v1/out/0.raw
MODELS=/Volumes/VertigoDataTier/pact/ddm_ane3_residue/models/item1
PAYLOAD=/Volumes/APDataStore/pact/ddm_ane3_residue/item1
RUNTIME=/Users/adpena/Projects/pact/.omx/tmp/codex_runs/ddm_ane3_runtime

export PYTHONPATH=src:upstream
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

mkdir -p "$MODELS" "$PAYLOAD" "$RUNTIME/tmp" "$RUNTIME/home/Library/Caches"

# Redirect process-owned Core ML scratch/cache into a writable, retained root.
# This does not change authority: MLComputePlan still records actual device use.
export TMPDIR="$RUNTIME/tmp/"
export CFFIXED_USER_HOME="$RUNTIME/home"

check_storage() {
  df -h /Volumes/VertigoDataTier/pact /Volumes/APDataStore/pact
  avail_kib=$(df -Pk /Volumes/VertigoDataTier/pact | awk 'NR==2 {print $4}')
  if [ "$avail_kib" -lt 10485760 ]; then
    echo "REFUSED: Vertigo has less than 10 GiB free" >&2
    exit 2
  fi
}

run_ladder() {
  label=$1
  split=$2
  report="$PAYLOAD/${label}_n600.json"
  if [ -s "$report" ]; then
    echo "[resume] retained $report"
    return
  fi
  check_storage
  "$PY" "$EXP" ladder \
    --model segnet \
    --enumerate-json "$ENUM" \
    --reference "$REF" \
    --raw "$RAW" \
    --splits "$split" \
    --modes CPU_AND_NE \
    --reps 3 \
    --threads 1 \
    --compute-units CPU_AND_NE \
    --out-dir "$MODELS/$label" \
    --payload-dir "$PAYLOAD/$label" \
    --eval-pairs 600 \
    --out "$report"
}

run_selective() {
  label=$1
  ordinals=$2
  report="$PAYLOAD/${label}_n600.json"
  if [ -s "$report" ]; then
    echo "[resume] retained $report"
    return
  fi
  check_storage
  "$PY" "$EXP" selective \
    --model segnet \
    --enumerate-json "$ENUM" \
    --reference "$REF" \
    --raw "$RAW" \
    --fp32-set "$label=$ordinals" \
    --modes CPU_AND_NE \
    --reps 3 \
    --threads 1 \
    --compute-units CPU_AND_NE \
    --out-dir "$MODELS/$label" \
    --payload-dir "$PAYLOAD/$label" \
    --eval-pairs 600 \
    --out "$report"
}

run_ladder all_fp16 0
run_selective g13 243:261
run_selective g13stem 0:38,243:261
run_ladder tail_k64 64

echo "DDM_ANE3_ITEM1_COMPLETE"
