#!/bin/bash
set -euo pipefail
cd /Users/adpena/Projects/pact
.venv/bin/python -u experiments/ddm_tc4_receiver.py smoke --resume-from /Volumes/VertigoDataTier/pact/ddm_tc4_context_slate/move41
.venv/bin/python -u experiments/ddm_tc4_receiver.py public --resume-from /Volumes/VertigoDataTier/pact/ddm_tc4_context_slate/move41 --stop-after 1
.venv/bin/python -u experiments/ddm_tc4_receiver.py public --resume-from /Volumes/VertigoDataTier/pact/ddm_tc4_context_slate/move41 --stop-after 2
.venv/bin/python -u experiments/ddm_tc4_receiver.py public --resume-from /Volumes/VertigoDataTier/pact/ddm_tc4_context_slate/move41 --stop-after 2 --name public_fresh2
