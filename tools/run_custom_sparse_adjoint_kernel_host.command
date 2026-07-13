#!/bin/zsh
# SPDX-License-Identifier: MIT
set -euo pipefail

ROOT="/Users/adpena/Projects/pact"
OUT="$ROOT/experiments/results/custom_sparse_adjoint_kernel_20260713"
mkdir -p "$OUT"
cd "$ROOT"

exec .venv/bin/python tools/bench_custom_sparse_adjoint_kernel.py \
  --output-dir "$OUT" \
  --resume \
  --warmups 2 \
  --repeats 7 \
  >>"$OUT/host_terminal.log" 2>&1
