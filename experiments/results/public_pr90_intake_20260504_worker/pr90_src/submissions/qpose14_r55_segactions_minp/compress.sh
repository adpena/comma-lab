#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ARCHIVE_URL="${ARCHIVE_URL:-https://github.com/EthanYangTW/comma_video_compression_challenge/releases/download/qpose14-r55-segactions-minp-v2/archive.zip}"

if command -v curl >/dev/null 2>&1; then
  curl -L "$ARCHIVE_URL" -o "$HERE/archive.zip"
else
  python3 - "$ARCHIVE_URL" "$HERE/archive.zip" <<'PY'
import sys
import urllib.request

urllib.request.urlretrieve(sys.argv[1], sys.argv[2])
PY
fi

python3 - "$HERE/archive.zip" <<'PY'
import hashlib
import sys
from pathlib import Path

expected = "01dc02badf851d99108fd92c570271f36f74cc5424c6d2a8f1b499cb4d1c3446"
path = Path(sys.argv[1])
actual = hashlib.sha256(path.read_bytes()).hexdigest()
if actual != expected:
    raise SystemExit(f"archive checksum mismatch: {actual} != {expected}")
print(f"wrote {path} sha256={actual}")
PY
