#!/usr/bin/env bash
set -euo pipefail

# OD4 queued scorer gate. Fill SUB_DIR only after a receiver-closed staged
# submission exists. od3 owns the scorer slot at OD4 build time, so this script
# is a fire-order artifact, not an active launch.
SUB_DIR="${SUB_DIR:?set SUB_DIR to the receiver-closed staged submission directory}"
OUT="${OUT:-.omx/research/ddm_od4_20260805/od4_receiver_gate_fz2_receipt.json}"

.venv/bin/python experiments/ddm_fz2_byteclose_and_eval.py \
  --sub-dir "${SUB_DIR}" \
  --out "${OUT}" \
  --inflate-out "${SUB_DIR}/inflated" \
  --device cpu \
  --batch-size 16 \
  --num-threads 6
