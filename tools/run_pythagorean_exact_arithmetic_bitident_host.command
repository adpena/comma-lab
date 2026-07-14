#!/bin/zsh
# SPDX-License-Identifier: MIT
set -euo pipefail

ROOT="/Users/adpena/Projects/pact"
OUT="$ROOT/.omx/research/pythagorean_exact_arithmetic_bitident_probe_20260713.json"
LOG="$ROOT/.omx/research/pythagorean_exact_arithmetic_bitident_probe_20260713.host.log"
cd "$ROOT"

exec .venv/bin/python tools/probe_pythagorean_exact_arithmetic_bitident.py \
  --n 10 \
  --resume \
  --output "$OUT" \
  >>"$LOG" 2>&1
