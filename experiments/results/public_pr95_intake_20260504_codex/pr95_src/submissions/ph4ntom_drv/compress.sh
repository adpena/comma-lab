#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
cd "$ROOT"
python "$HERE/compress.py" --video-dir "$ROOT/videos" --video-names "$ROOT/public_test_video_names.txt" --batch-size 2 --device "${1:-cuda:0}"
