#!/bin/bash
set -euo pipefail
cd /Users/adpena/Projects/pact
orientation="$1"
case "$orientation" in 0|1) ;; *) exit 2 ;; esac
for gap in 0 1 2 4; do
  .venv/bin/python experiments/ddm_bnd3_address_term.py geometry \
    --resume-from /Volumes/VertigoDataTier/pact/ddm_bnd3_address_term \
    --orientation "$orientation" --gap "$gap"
  for minimum in 1 2 4; do
    .venv/bin/python experiments/ddm_bnd3_address_term.py price \
      --resume-from /Volumes/VertigoDataTier/pact/ddm_bnd3_address_term \
      --orientation "$orientation" --gap "$gap" --minimum "$minimum"
  done
done
