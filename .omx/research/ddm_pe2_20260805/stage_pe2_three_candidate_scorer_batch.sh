#!/usr/bin/env bash
set -euo pipefail
cd /Users/adpena/Projects/pact
DEVICE="${1:-cpu}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_THREADS="${NUM_THREADS:-4}"
OUT_ROOT="/Volumes/VertigoDataTier/pact/ddm_pe2_20260805/scorer_batch"
mkdir -p "$OUT_ROOT"
echo "[pe2] one scorer-slot batch; run only after MAIN confirms sq2 released the slot" >&2
echo "[pe2] scoring pe1_full_explicit_curve_k8 on $DEVICE" >&2
.venv/bin/python experiments/ddm_fz2_byteclose_and_eval.py --sub-dir /Volumes/VertigoDataTier/pact/ddm_pe2_20260805/sub_auto_pairbit_pe2_pe1_full_explicit_curve_k8_receiver --out "$OUT_ROOT/pe1_full_explicit_curve_k8_n600_${DEVICE}.json" --inflate-out "$OUT_ROOT/pe1_full_explicit_curve_k8_inflate_${DEVICE}" --device "$DEVICE" --batch-size "$BATCH_SIZE" --num-threads "$NUM_THREADS"
echo "[pe2] scoring pe1_surgical_generator_pair_waterfill_75kb on $DEVICE" >&2
.venv/bin/python experiments/ddm_fz2_byteclose_and_eval.py --sub-dir /Volumes/VertigoDataTier/pact/ddm_pe2_20260805/sub_auto_pairbit_pe2_pe1_surgical_generator_pair_waterfill_75kb_receiver --out "$OUT_ROOT/pe1_surgical_generator_pair_waterfill_75kb_n600_${DEVICE}.json" --inflate-out "$OUT_ROOT/pe1_surgical_generator_pair_waterfill_75kb_inflate_${DEVICE}" --device "$DEVICE" --batch-size "$BATCH_SIZE" --num-threads "$NUM_THREADS"
echo "[pe2] scoring bf1_lane_crop_r3 on $DEVICE" >&2
.venv/bin/python experiments/ddm_fz2_byteclose_and_eval.py --sub-dir /Volumes/VertigoDataTier/pact/ddm_pe2_20260805/sub_auto_pairbit_pe2_bf1_lane_crop_r3_receiver --out "$OUT_ROOT/bf1_lane_crop_r3_n600_${DEVICE}.json" --inflate-out "$OUT_ROOT/bf1_lane_crop_r3_inflate_${DEVICE}" --device "$DEVICE" --batch-size "$BATCH_SIZE" --num-threads "$NUM_THREADS"
