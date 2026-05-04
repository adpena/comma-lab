#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PD="$(cd "${HERE}/../.." && pwd)"

IN_DIR="${PD}/videos"
VIDEO_NAMES_FILE="${PD}/public_test_video_names.txt"
ARCHIVE_DIR="${HERE}/archive"

DEVICE="cuda"

# Argument processing
while [[ $# -gt 0 ]]; do
  case "$1" in
    --in-dir|--in_dir)
      IN_DIR="${2%/}"; shift 2 ;;
    --video-names-file|--video_names_file)
      VIDEO_NAMES_FILE="$2"; shift 2 ;;
    --device)
      DEVICE="$2"; shift 2 ;; 
    *)
      echo "Unknown arg: $1" >&2
      exit 2 ;;
  esac
done

# Remove and prepare folder for result of compression
rm -rf "$ARCHIVE_DIR"
mkdir -p "$ARCHIVE_DIR"

echo "==> Launching compression with ditcher.py"

if [ -d "${PD}/.venv" ]; then
    source "${PD}/.venv/bin/activate"
fi

python3 "${HERE}/ditcher.py" \
    --mode compress --device $DEVICE
echo "==> Compressing results to archive.zip..."
cd "$ARCHIVE_DIR"
if [ "$(ls -A .)" ]; then
    zip -q -r "${HERE}/archive.zip" .
    echo "    Finished: ${HERE}/archive.zip"
else
    echo "    [WARNING] Folder archive is empty! Check where ditcher.py stores results."
fi