#!/usr/bin/env bash
# Must produce a raw video file at `<output_dir>/<base_name>.raw`.
# A `.raw` file is a flat binary dump of uint8 RGB frames with shape `(N, H, W, 3)`
# where N is the number of frames, H and W match the original video dimensions, no header.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p "${HERE}/inflated"
python3 "${HERE}/ditcher.py" --mode decompress --device cuda

echo "done"
