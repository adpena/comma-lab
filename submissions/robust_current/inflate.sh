#!/usr/bin/env bash
set -euo pipefail

ARCHIVE_DIR="${1:?archive dir required}"
INFLATED_DIR="${2:?inflated dir required}"
VIDEO_NAMES_FILE="${3:?video names file required}"

SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_ENV_PATH="${CONFIG_ENV_PATH:-$SELF_DIR/config.env}"
if [[ "$CONFIG_ENV_PATH" != /* ]]; then
  CONFIG_ENV_PATH="$SELF_DIR/$CONFIG_ENV_PATH"
fi
if [ -f "$CONFIG_ENV_PATH" ]; then
  # shellcheck source=/dev/null
  source "$CONFIG_ENV_PATH"
fi

FFMPEG_BIN="${FFMPEG_BIN:-ffmpeg}"
UV_BIN="${UV_BIN:-uv}"
ROI_SCRIPT_PY="${ROI_SCRIPT_PY:-$SELF_DIR/analyze_roi.py}"
INFLATE_POSTFILTER="${INFLATE_POSTFILTER:-}"
ROI_ENABLE="${ROI_ENABLE:-0}"
SOURCE_W="${SOURCE_W:-1164}"
SOURCE_H="${SOURCE_H:-874}"
SOURCE_COLOR_RANGE="${SOURCE_COLOR_RANGE:-tv}"
SOURCE_COLOR_MATRIX="${SOURCE_COLOR_MATRIX:-bt709}"
SOURCE_COLOR_PRIMARIES="${SOURCE_COLOR_PRIMARIES:-bt709}"
SOURCE_COLOR_TRC="${SOURCE_COLOR_TRC:-bt709}"
RGB_OUTPUT_RANGE="${RGB_OUTPUT_RANGE:-pc}"
UPSCALE_FLAGS="${UPSCALE_FLAGS:-lanczos}"
ROI_X_FRAC="${ROI_X_FRAC:-0.15}"
ROI_Y_FRAC="${ROI_Y_FRAC:-0.22}"
ROI_W_FRAC="${ROI_W_FRAC:-0.70}"
ROI_H_FRAC="${ROI_H_FRAC:-0.55}"
ROI2_ENABLE="${ROI2_ENABLE:-0}"
ROI2_X_FRAC="${ROI2_X_FRAC:-0.72}"
ROI2_Y_FRAC="${ROI2_Y_FRAC:-0.10}"
ROI2_W_FRAC="${ROI2_W_FRAC:-0.22}"
ROI2_H_FRAC="${ROI2_H_FRAC:-0.55}"
ROI_METADATA_ENABLE="${ROI_METADATA_ENABLE:-0}"
PYTHON_INFLATE="${PYTHON_INFLATE:-0}"
mkdir -p "$INFLATED_DIR"

require_cmd() {
  local bin="$1"
  if ! command -v "$bin" >/dev/null 2>&1; then
    echo "ERROR: required tool not found in PATH: $bin" >&2
    exit 1
  fi
}

ffmpeg_filter_option_available() {
  local option="$1"
  local scale_help
  scale_help="$("$FFMPEG_BIN" -hide_banner -h filter=scale 2>/dev/null)"
  grep -q "$option" <<<"$scale_help"
}

require_ffmpeg_parity() {
  require_cmd "$FFMPEG_BIN"
  require_cmd "$UV_BIN"

  for opt in in_range out_range in_color_matrix in_primaries in_transfer; do
    if ! ffmpeg_filter_option_available "$opt"; then
      echo "ERROR: $FFMPEG_BIN scale filter is missing required option '$opt' for the explicit decode color-contract path." >&2
      echo "This environment would drift from the canonical path. Set FFMPEG_BIN to a parity-compatible ffmpeg build." >&2
      exit 1
    fi
  done
}

require_ffmpeg_parity


upscale_rgb_base_filter() {
  local width="$1"
  local height="$2"
  local flags="$3"
  printf 'scale=%s:%s:flags=%s:in_range=%s:out_range=%s:in_color_matrix=%s:in_primaries=%s:in_transfer=%s,format=rgb24' \
    "$width" "$height" "$flags" \
    "$SOURCE_COLOR_RANGE" "$RGB_OUTPUT_RANGE" \
    "$SOURCE_COLOR_MATRIX" "$SOURCE_COLOR_PRIMARIES" "$SOURCE_COLOR_TRC"
}

upscale_filter() {
  local width="$1"
  local height="$2"
  local flags="$3"
  local base
  base="$(upscale_rgb_base_filter "$width" "$height" "$flags")"
  if [ -n "$INFLATE_POSTFILTER" ]; then
    printf '%s,%s' "$base" "$INFLATE_POSTFILTER"
  else
    printf '%s' "$base"
  fi
}

calc_even_dim() {
  python3 - "$@" <<'PY'
import sys
value = int(float(sys.argv[1]) * float(sys.argv[2]))
if value < 2:
    value = 2
if value % 2:
    value -= 1
print(value)
PY
}

calc_even_origin() {
  python3 - "$@" <<'PY'
import sys
scale = int(sys.argv[1])
frac = float(sys.argv[2])
size = int(sys.argv[3])
value = int(round(scale * frac))
value = max(0, min(value, scale - size))
if value % 2:
    value -= 1
print(max(0, value))
PY
}

while IFS= read -r rel; do
  [ -n "$rel" ] || continue
  stem="${rel%.*}"
  out_rel="${stem}.raw"
  out_path="$INFLATED_DIR/$out_rel"
  mkdir -p "$(dirname "$out_path")"

  if [ "$ROI_ENABLE" = "1" ]; then
    metadata_path="$ARCHIVE_DIR/$stem/roi_metadata.json"
    if [ "$ROI_METADATA_ENABLE" = "1" ] && [ -f "$metadata_path" ]; then
      inflate_cmd=("$UV_BIN" run python "$ROI_SCRIPT_PY" inflate-metadata
        --archive-dir "$ARCHIVE_DIR/$stem"
        --metadata "$metadata_path"
        --out "$out_path"
        --ffmpeg-bin "$FFMPEG_BIN"
        --upscale-flags "$UPSCALE_FLAGS")
      inflate_cmd+=(--source-color-range "$SOURCE_COLOR_RANGE" --source-color-matrix "$SOURCE_COLOR_MATRIX" --source-color-primaries "$SOURCE_COLOR_PRIMARIES" --source-color-trc "$SOURCE_COLOR_TRC" --rgb-output-range "$RGB_OUTPUT_RANGE")
      if [ -n "$INFLATE_POSTFILTER" ]; then
        inflate_cmd+=(--postfilter "$INFLATE_POSTFILTER")
      fi
      if [ "$ROI2_ENABLE" = "1" ]; then
        inflate_cmd+=(--roi2-enable)
      fi
      "${inflate_cmd[@]}"
    else
      base_path="$ARCHIVE_DIR/$stem/base.mkv"
      roi_path="$ARCHIVE_DIR/$stem/roi.mkv"
      roi2_path="$ARCHIVE_DIR/$stem/roi2.mkv"
      roi_w="$(calc_even_dim "$SOURCE_W" "$ROI_W_FRAC")"
      roi_h="$(calc_even_dim "$SOURCE_H" "$ROI_H_FRAC")"
      roi_x="$(calc_even_origin "$SOURCE_W" "$ROI_X_FRAC" "$roi_w")"
      roi_y="$(calc_even_origin "$SOURCE_H" "$ROI_Y_FRAC" "$roi_h")"

      if [ "$ROI2_ENABLE" = "1" ] && [ -f "$roi2_path" ]; then
        roi2_w="$(calc_even_dim "$SOURCE_W" "$ROI2_W_FRAC")"
        roi2_h="$(calc_even_dim "$SOURCE_H" "$ROI2_H_FRAC")"
        roi2_x="$(calc_even_origin "$SOURCE_W" "$ROI2_X_FRAC" "$roi2_w")"
        roi2_y="$(calc_even_origin "$SOURCE_H" "$ROI2_Y_FRAC" "$roi2_h")"

        echo "Inflating ROI two-pass+aux $base_path + $roi_path + $roi2_path -> $out_path"
        "$FFMPEG_BIN" -y -i "$base_path" -i "$roi_path" -i "$roi2_path" \
          -filter_complex "[0:v]$(upscale_rgb_base_filter "$SOURCE_W" "$SOURCE_H" "$UPSCALE_FLAGS")[base];[1:v]$(upscale_rgb_base_filter "$roi_w" "$roi_h" "$UPSCALE_FLAGS")[roi1];[2:v]$(upscale_rgb_base_filter "$roi2_w" "$roi2_h" "$UPSCALE_FLAGS")[roi2];[base][roi1]overlay=${roi_x}:${roi_y}[tmp];[tmp][roi2]overlay=${roi2_x}:${roi2_y}$(if [ -n "$INFLATE_POSTFILTER" ]; then printf ',format=rgb24,%s' "$INFLATE_POSTFILTER"; else printf ',format=rgb24'; fi)[out]" \
          -map "[out]" -an -sn -pix_fmt rgb24 -f rawvideo "$out_path"
      else
        echo "Inflating ROI two-pass $base_path + $roi_path -> $out_path"
        "$FFMPEG_BIN" -y -i "$base_path" -i "$roi_path" \
          -filter_complex "[0:v]$(upscale_rgb_base_filter "$SOURCE_W" "$SOURCE_H" "$UPSCALE_FLAGS")[base];[1:v]$(upscale_rgb_base_filter "$roi_w" "$roi_h" "$UPSCALE_FLAGS")[roi];[base][roi]overlay=${roi_x}:${roi_y}$(if [ -n "$INFLATE_POSTFILTER" ]; then printf ',format=rgb24,%s' "$INFLATE_POSTFILTER"; else printf ',format=rgb24'; fi)[out]" \
          -map "[out]" -an -sn -pix_fmt rgb24 -f rawvideo "$out_path"
      fi
    fi
  else
    in_rel="${stem}.mkv"
    in_path="$ARCHIVE_DIR/$in_rel"
    if [ "$PYTHON_INFLATE" = "postfilter" ]; then
      echo "Inflating (canonical + learned post-filter) $ARCHIVE_DIR -> $INFLATED_DIR"
      "$UV_BIN" run python "$SELF_DIR/inflate_postfilter.py" \
        "$ARCHIVE_DIR" "$INFLATED_DIR" "$VIDEO_NAMES_FILE" \
        "${POSTFILTER_PATH:-$SELF_DIR/postfilter_int8.pt}"
      break
    elif [ "$PYTHON_INFLATE" = "grain_mask" ]; then
      echo "Inflating (saliency-masked grain) $ARCHIVE_DIR -> $INFLATED_DIR"
      "$UV_BIN" run --with av --with torch --with numpy python "$SELF_DIR/inflate_grain_mask.py" \
        "$ARCHIVE_DIR" "$INFLATED_DIR" "$VIDEO_NAMES_FILE" \
        "${GRAIN_MASK_SALIENCY:-experiments/masks/posenet_saliency.npy}" \
        "${GRAIN_MASK_STRENGTH:-8.0}"
      break
    elif [ "$PYTHON_INFLATE" = "1" ]; then
      echo "Inflating (canonical PyAV + torch bicubic) $ARCHIVE_DIR -> $INFLATED_DIR"
      "$UV_BIN" run --with av --with torch --with numpy python "$SELF_DIR/inflate.py" \
        "$ARCHIVE_DIR" "$INFLATED_DIR" "$VIDEO_NAMES_FILE"
      break  # Python script handles all videos in one call
    else
      echo "Inflating $in_path -> $out_path"
      "$FFMPEG_BIN" -y -i "$in_path" -vf "$(upscale_filter "$SOURCE_W" "$SOURCE_H" "$UPSCALE_FLAGS")" -an -sn -pix_fmt rgb24 -f rawvideo "$out_path"
    fi
  fi
done < "$VIDEO_NAMES_FILE"
