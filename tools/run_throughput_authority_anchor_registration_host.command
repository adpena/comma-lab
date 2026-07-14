#!/bin/zsh
set -euo pipefail

cd "${0:A:h}/.."
exec .venv/bin/python tools/register_throughput_authority_ladder_anchors.py \
  --write --require-all
