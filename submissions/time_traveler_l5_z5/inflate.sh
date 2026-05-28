#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Z5 Rao-Ballard hierarchical predictive coding + Hinton-distilled scorer
# surrogate contest-compliant inflate wrapper.
#
# Per CLAUDE.md HNeRV parity discipline L9 (runtime closure: torch + brotli
# only; numpy stdlib-adjacent). Per Catalog #146 (contest 3-positional-arg
# signature: archive_dir output_dir file_list). Per Catalog #163 (set -euo
# pipefail). Per CLAUDE.md "Strict scorer rule" + Catalog #6 (no scorer load
# at inflate; predictor + decoder only).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${1?missing archive_dir}"
OUTPUT_DIR="${2?missing output_dir}"
FILE_LIST="${3?missing file_list}"
mkdir -p "$OUTPUT_DIR"
exec "${PYTHON:-python3}" "$HERE/inflate.py" "$DATA_DIR" "$OUTPUT_DIR" "$FILE_LIST"
