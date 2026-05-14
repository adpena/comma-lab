#!/usr/bin/env bash
# C6 MDL-IBPS contest inflate runtime.
# Per Catalog #146: 3 positional args (archive_dir, output_dir, file_list).
# Per Catalog #163: set -euo pipefail.
set -euo pipefail

if [ $# -lt 3 ]; then
    echo "usage: $0 <archive_dir> <output_dir> <file_list>" >&2
    exit 2
fi

ARCHIVE_DIR="$1"
OUTPUT_DIR="$2"
FILE_LIST="$3"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${HERE}/src:${PYTHONPATH:-}"
exec uv run --python 3.13 --with torch==2.5.1 --with brotli --with numpy "${HERE}/inflate.py" "${ARCHIVE_DIR}" "${OUTPUT_DIR}" "${FILE_LIST}"
