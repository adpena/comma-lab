#!/bin/bash
set -euo pipefail
cd /Users/adpena/Projects/pact
for mask in 1 2 4 8 16; do
  .venv/bin/python -u experiments/ddm_tc4_price.py fit --resume-from /Volumes/VertigoDataTier/pact/ddm_tc4_context_slate/move41 --mask "$mask"
  .venv/bin/python -u experiments/ddm_tc4_price.py encode --resume-from /Volumes/VertigoDataTier/pact/ddm_tc4_context_slate/move41 --mask "$mask"
done
