#!/bin/zsh
# SPDX-License-Identifier: MIT
# Zero-launch host wrapper.  It consumes two existing n600 trajectory receipts.
set -euo pipefail

ROOT="/Users/adpena/Projects/pact"
CE_RECEIPT="${1:?usage: $0 CE_RECEIPT.json MARGIN_RECEIPT.json OUTPUT.json}"
MARGIN_RECEIPT="${2:?usage: $0 CE_RECEIPT.json MARGIN_RECEIPT.json OUTPUT.json}"
OUTPUT="${3:?usage: $0 CE_RECEIPT.json MARGIN_RECEIPT.json OUTPUT.json}"

cd "$ROOT"
exec .venv/bin/python tools/probe_ordinal_perclass_convergence.py \
  --ce-receipt "$CE_RECEIPT" \
  --margin-receipt "$MARGIN_RECEIPT" \
  --output "$OUTPUT"
