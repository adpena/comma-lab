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
# IC2 declares and bootstraps its two non-Torch runtime wheels.
if ! "$PYBIN" - <<'PY'
import brotli
import cv2
if cv2.__version__ != "4.11.0":
    raise ImportError(f"unexpected OpenCV runtime: {cv2.__version__}")
PY
then
  "$PYBIN" -m pip install --disable-pip-version-check --no-input --no-cache-dir     "Brotli==1.2.0" "opencv-python-headless==4.11.0.86"
fi
"$PYBIN" - <<'PY'
import brotli
import cv2
if cv2.__version__ != "4.11.0":
    raise ImportError(f"IC2 OpenCV bootstrap failed: {cv2.__version__}")
PY
exec "$PYBIN" "$HERE/inflate.py" "$1" "$2" "$3"
