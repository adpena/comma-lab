#!/usr/bin/env bash
set -euo pipefail
cd /Users/adpena/Projects/pact
DEVICE="${1:-cpu}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_THREADS="${NUM_THREADS:-4}"
OUT_ROOT="/Volumes/VertigoDataTier/pact/ddm_pe4_20260805_r2/scorer_batch"
mkdir -p "$OUT_ROOT"
echo "[pe4] fourth scorer candidate; run only after MAIN harvests the active PE2 batch" >&2
echo "[pe4] scoring pe3_hybrid_75kb_receiver on $DEVICE" >&2
.venv/bin/python experiments/ddm_fz2_byteclose_and_eval.py --sub-dir /Volumes/VertigoDataTier/pact/ddm_pe4_20260805_r2/sub_auto_pairbit_pe4_pe3_hybrid_75kb_receiver --out "$OUT_ROOT/pe3_hybrid_75kb_n600_${DEVICE}.json" --inflate-out "/Volumes/VertigoDataTier/pact/ddm_pe4_20260805_r2/sub_auto_pairbit_pe4_pe3_hybrid_75kb_receiver/inflated" --device "$DEVICE" --batch-size "$BATCH_SIZE" --num-threads "$NUM_THREADS"
# MAIN fix 2026-08-05: --inflate-out must be <sub-dir>/inflated per evaluate.py:68 (2nd instance of the pe2-v1 class; harness-default fix owed)
