#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 3 ]]; then
  echo "usage: inflate.sh ARCHIVE_DIR OUTPUT_DIR VIDEO_NAMES_FILE" >&2
  exit 2
fi

runtime_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
python_bin="${PYTHON_BIN:-${PYTHON:-python3}}"
exec "${python_bin}" -B "${runtime_dir}/inflate.py" "$1" "$2" "$3"
