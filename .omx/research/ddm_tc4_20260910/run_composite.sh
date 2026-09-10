#!/bin/bash
set -euo pipefail
cd /Users/adpena/Projects/pact
.venv/bin/python -u experiments/ddm_tc4_price.py fit --resume-from /Volumes/VertigoDataTier/pact/ddm_tc4_context_slate/move41 --mask 19
.venv/bin/python -u experiments/ddm_tc4_price.py encode --resume-from /Volumes/VertigoDataTier/pact/ddm_tc4_context_slate/move41 --mask 19
