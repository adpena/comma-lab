#!/usr/bin/env bash
set -euo pipefail

# OD5 queued scorer gate. od3 owns the scorer slot at OD5 build time, so this
# is a fire-order artifact only. Bind SUB_DIR after a receiver-closed staged
# submission exists.
SUB_DIR="${SUB_DIR:?set SUB_DIR to the receiver-closed staged submission directory}"
OUT="${OUT:-.omx/research/ddm_od5_20260805/od5_receiver_gate_receipt.json}"

.venv/bin/python experiments/ddm_fz2_byteclose_and_eval.py \
  --sub-dir "${SUB_DIR}" \
  --out "${OUT}" \
  --inflate-out "${SUB_DIR}/inflated" \
  --device cpu \
  --batch-size 16 \
  --num-threads 6
