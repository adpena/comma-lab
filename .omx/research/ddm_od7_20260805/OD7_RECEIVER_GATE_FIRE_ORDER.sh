#!/usr/bin/env bash
set -euo pipefail

# OD7 receiver gate only.  Do not run while another full scorer lane is active.
SUB_DIR="${SUB_DIR:?set SUB_DIR to /Volumes/VertigoDataTier/pact/ddm_od7_20260805/<run>/sub_od7}"
OUT="${OUT:-.omx/research/ddm_od7_20260805/od7_receiver_gate_receipt.json}"

.venv/bin/python experiments/ddm_fz2_byteclose_and_eval.py   --sub-dir "${SUB_DIR}"   --out "${OUT}"   --inflate-out "${SUB_DIR}/inflated"   --device cpu   --batch-size 16   --num-threads 6
