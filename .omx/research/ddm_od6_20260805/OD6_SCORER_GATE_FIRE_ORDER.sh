#!/usr/bin/env bash
set -euo pipefail

# OD6 queued scorer gate only. od3 owns the active scorer slot at OD6 build
# time; run this only after a receiver-closed staged OD6 submission exists and
# the scorer lane is explicitly claimed.
SUB_DIR="${SUB_DIR:?set SUB_DIR to the receiver-closed staged submission directory}"
OUT="${OUT:-.omx/research/ddm_od6_20260805/od6_receiver_gate_receipt.json}"

.venv/bin/python experiments/ddm_fz2_byteclose_and_eval.py \
  --sub-dir "${SUB_DIR}" \
  --out "${OUT}" \
  --inflate-out "${SUB_DIR}/inflated" \
  --device cpu \
  --batch-size 16 \
  --num-threads 6
