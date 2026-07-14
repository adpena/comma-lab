#!/bin/zsh
# SPDX-License-Identifier: MIT
# Zero-launch host wrapper. It only analyzes existing receipt bytes.
set -euo pipefail

ROOT="/Users/adpena/Projects/pact"
PREREGISTRATION="${1:?usage: $0 PREREGISTRATION.json CONTROL.json TREATMENT.json OUTPUT.json}"
CONTROL_RECEIPT="${2:?usage: $0 PREREGISTRATION.json CONTROL.json TREATMENT.json OUTPUT.json}"
TREATMENT_RECEIPT="${3:?usage: $0 PREREGISTRATION.json CONTROL.json TREATMENT.json OUTPUT.json}"
OUTPUT="${4:?usage: $0 PREREGISTRATION.json CONTROL.json TREATMENT.json OUTPUT.json}"

cd "$ROOT"
exec .venv/bin/python tools/probe_weight_decay_plasticity_ab.py \
  --preregistration "$PREREGISTRATION" \
  --control-receipt "$CONTROL_RECEIPT" \
  --treatment-receipt "$TREATMENT_RECEIPT" \
  --output "$OUTPUT"
