#!/usr/bin/env bash
set -euo pipefail
if [ "$#" -ne 3 ]; then
  echo "Usage: inflate.sh <archive_dir> <output_dir> <video_names_file>" >&2
  exit 2
fi
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${PYTHON:-python3}" "$HERE/inflate.py" "$1" "$2" "$3"
