#!/usr/bin/env bash
# Run the full training + archive pipeline for jas0xf_adversarial_neural_representation.
# All args are forwarded to compress.py. See `python compress.py --help`.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec python3 -u "${HERE}/compress.py" "$@"
