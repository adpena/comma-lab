#!/bin/zsh
# SPDX-License-Identifier: MIT
# Static/receipt-only: this wrapper cannot train, score, dispatch, or call a provider.
set -euo pipefail

ROOT="/Users/adpena/Projects/pact"
OUTPUT="${1:?usage: $0 OUTPUT.json [all-ones-receipt.json sigma-receipt.json]}"
SIGMA_SPEC="${SIGMA_SPEC:-fitted-20260707}"
cd "$ROOT"
if (( $# == 1 )); then
  exec .venv/bin/python tools/probe_sigma_ccprime_gamma_limit.py \
    --sigma-spec "$SIGMA_SPEC" --output "$OUTPUT"
elif (( $# == 3 )); then
  exec .venv/bin/python tools/probe_sigma_ccprime_gamma_limit.py \
    --sigma-spec "$SIGMA_SPEC" --output "$OUTPUT" \
    --all-ones-receipt "$2" --sigma-receipt "$3"
fi
print -u2 "usage: $0 OUTPUT.json [all-ones-receipt.json sigma-receipt.json]"
exit 64
