#!/usr/bin/env bash
# Per HNeRV parity discipline lesson 9 (runtime closure).
# Per Catalog #146 contest contract: 3-arg signature archive_dir output_dir file_list.
set -euo pipefail

if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <archive_dir> <output_dir> <file_list>" >&2
    exit 1
fi

DATA_DIR="$1"
OUTPUT_DIR="$2"
FILE_LIST="$3"

HERE="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$OUTPUT_DIR"

# NSCS02 archive ships as 0.bin (HNeRV parity L3 monolithic single-file).
SRC_BIN=""
for candidate in "$DATA_DIR/0.bin" "$DATA_DIR/nscs02.bin" "$DATA_DIR/x"; do
    if [ -f "$candidate" ]; then
        SRC_BIN="$candidate"
        break
    fi
done
if [ -z "$SRC_BIN" ]; then
    echo "FATAL: NSCS02 source blob (0.bin/nscs02.bin/x) missing in $DATA_DIR" >&2
    exit 1
fi

# Render each video listed in file_list. The contest test set has one
# video, so this loop typically iterates once.
while IFS= read -r video_name; do
    [ -z "$video_name" ] && continue
    case "$video_name" in \#*) continue ;; esac
    stem="${video_name%.*}"
    "${PYTHON:-python3}" "$HERE/inflate.py" "$SRC_BIN" "$OUTPUT_DIR/${stem}.raw"
done < "$FILE_LIST"
