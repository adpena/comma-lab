#!/usr/bin/env bash
# Capstone VQ-NeRV contest inflate.sh — reference runtime stub.
#
# Contract (matches the anr/PR95 submission convention + upstream/evaluate.py):
#   inflate.sh DATA_DIR OUTPUT_DIR FILE_LIST
#     DATA_DIR    holds the archive member ("x" or "<base>.bin") + the
#                 "capstone_config_v1" sidecar (when DATA_DIR/archive.zip is used,
#                 both live inside the zip; the runtime reads the zip directly).
#     OUTPUT_DIR  receives "<base>.raw" — the flat uint8 (N, 874, 1164, 3) tensor
#                 that upstream/frame_utils.TensorVideoDataset reads from
#                 <submission_dir>/inflated/.
#     FILE_LIST   one video name per line (e.g. "0.mkv").
#
# The decode is pure-numpy (tac.capstone_vq_nerv.inflate) — CPU/CUDA-agnostic,
# no MLX, no torch, NO scorer loaded (Strict scorer rule). A self-contained
# contest submission tree must vendor tac.capstone_vq_nerv.{inflate,export,
# numpy_reference} + numpy + brotli; this stub assumes PYTHONPATH reaches them.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="$1"
OUTPUT_DIR="$2"
FILE_LIST="$3"
mkdir -p "$OUTPUT_DIR"

while IFS= read -r line; do
  [ -z "$line" ] && continue
  BASE="${line%.*}"
  # Prefer a zip archive (carries the config sidecar); fall back to a bare member.
  SRC="${DATA_DIR}/archive.zip"
  [ -f "$SRC" ] || SRC="${DATA_DIR}/x"
  [ -f "$SRC" ] || SRC="${DATA_DIR}/${BASE}.bin"
  [ ! -f "$SRC" ] && echo "ERROR: capstone archive not found under ${DATA_DIR}" >&2 && exit 1
  DST="${OUTPUT_DIR}/${BASE}.raw"
  printf "Inflating %s (capstone VQ-NeRV)... " "$line"
  "${PYTHON:-python3}" -m tac.capstone_vq_nerv.inflate "$SRC" "$DST"
done < "$FILE_LIST"
