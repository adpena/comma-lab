#!/bin/bash
set -euo pipefail
cd /Users/adpena/Projects/pact
orientation="$1"
case "$orientation" in 0|1) ;; *) exit 2 ;; esac
.venv/bin/python experiments/ddm_bnd3_frozen_field.py fixed geometry \
  --resume-from /Volumes/VertigoDataTier/pact/ddm_bnd3_address_term \
  --orientation "$orientation" --gap 4
for minimum in 1 2 4; do
  .venv/bin/python experiments/ddm_bnd3_frozen_field.py fixed price \
    --resume-from /Volumes/VertigoDataTier/pact/ddm_bnd3_address_term \
    --orientation "$orientation" --gap 4 --minimum "$minimum"
done
for gap in 0 1 2 4; do
  for minimum in 1 2 4; do
    .venv/bin/python experiments/ddm_bnd3_frozen_field.py planar \
      --resume-from /Volumes/VertigoDataTier/pact/ddm_bnd3_address_term \
      --orientation "$orientation" --gap "$gap" --minimum "$minimum"
  done
done
