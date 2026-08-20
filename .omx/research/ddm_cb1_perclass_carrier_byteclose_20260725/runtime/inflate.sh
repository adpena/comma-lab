#!/usr/bin/env bash
set -euo pipefail
if [ "$#" -ne 3 ]; then
  echo "Usage: inflate.sh <archive_dir> <output_dir> <video_names_file>" >&2
  exit 2
fi
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# The archive manifest is the runtime-tree dependency declaration authority.
# PYTHON selects the locked environment provisioned from that declaration.
PYBIN="${PYTHON:-python3}"
exec "$PYBIN" "$HERE/inflate.py" "$1" "$2" "$3"
