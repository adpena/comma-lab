#!/usr/bin/env bash
set -euo pipefail

SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_ROOT="$(cd "$SELF_DIR/../../workspace/upstream/comma_video_compression_challenge" 2>/dev/null && pwd || true)"
UPSTREAM_ROOT="${COMMA_CHALLENGE_ROOT:-$DEFAULT_ROOT}"

if [ -z "${UPSTREAM_ROOT}" ] || [ ! -f "${UPSTREAM_ROOT}/evaluate.sh" ]; then
  echo "ERROR: Could not find upstream challenge root. Set COMMA_CHALLENGE_ROOT." >&2
  exit 1
fi

CONFIG_ENV_PATH="${CONFIG_ENV_PATH:-$SELF_DIR/config.env}"
if [[ "$CONFIG_ENV_PATH" != /* ]]; then
  CONFIG_ENV_PATH="$SELF_DIR/$CONFIG_ENV_PATH"
fi
if [ -f "$CONFIG_ENV_PATH" ]; then
  # shellcheck source=/dev/null
  source "$CONFIG_ENV_PATH"
fi

FFMPEG_BIN="${FFMPEG_BIN:-ffmpeg}"
FFPROBE_BIN="${FFPROBE_BIN:-ffprobe}"
UV_BIN="${UV_BIN:-uv}"
ROI_SCRIPT_PY="${ROI_SCRIPT_PY:-$SELF_DIR/analyze_roi.py}"
VIDEO_NAMES_FILE="${VIDEO_NAMES_FILE:-$UPSTREAM_ROOT/public_test_video_names.txt}"
ARCHIVE_ZIP="$SELF_DIR/archive.zip"
VIDEO_CODEC="${VIDEO_CODEC:-libx265}"
SVT_AV1_PRESET="${SVT_AV1_PRESET:-0}"
SVT_AV1_CRF="${SVT_AV1_CRF:-33}"
SVT_AV1_PARAMS="${SVT_AV1_PARAMS:-film-grain=22:keyint=180}"
SOURCE_COLOR_RANGE="${SOURCE_COLOR_RANGE:-tv}"
SOURCE_COLOR_MATRIX="${SOURCE_COLOR_MATRIX:-bt709}"
SOURCE_COLOR_PRIMARIES="${SOURCE_COLOR_PRIMARIES:-bt709}"
SOURCE_COLOR_TRC="${SOURCE_COLOR_TRC:-bt709}"
ROI_ENABLE="${ROI_ENABLE:-0}"
ROI_X_FRAC="${ROI_X_FRAC:-0.15}"
ROI_Y_FRAC="${ROI_Y_FRAC:-0.22}"
ROI_W_FRAC="${ROI_W_FRAC:-0.70}"
ROI_H_FRAC="${ROI_H_FRAC:-0.55}"
ROI_BASE_CRF_DELTA="${ROI_BASE_CRF_DELTA:-4}"
ROI_CRF_DELTA="${ROI_CRF_DELTA:-0}"
ROI2_ENABLE="${ROI2_ENABLE:-0}"
ROI2_X_FRAC="${ROI2_X_FRAC:-0.72}"
ROI2_Y_FRAC="${ROI2_Y_FRAC:-0.10}"
ROI2_W_FRAC="${ROI2_W_FRAC:-0.22}"
ROI2_H_FRAC="${ROI2_H_FRAC:-0.55}"
ROI2_CRF_DELTA="${ROI2_CRF_DELTA:-0}"
ROI_METADATA_ENABLE="${ROI_METADATA_ENABLE:-0}"
ROI_METADATA_WINDOW_FRAMES="${ROI_METADATA_WINDOW_FRAMES:-200}"
ROI_METADATA_SAMPLE_STEP="${ROI_METADATA_SAMPLE_STEP:-10}"
ROI_METADATA_TILE_COLS="${ROI_METADATA_TILE_COLS:-12}"
ROI_METADATA_TILE_ROWS="${ROI_METADATA_TILE_ROWS:-9}"
ROI_PREPROCESS_ENABLE="${ROI_PREPROCESS_ENABLE:-0}"
ROI_PREPROCESS_SCRIPT="${ROI_PREPROCESS_SCRIPT:-$SELF_DIR/roi_preprocess.py}"
ROI_PREPROCESS_BLUR_SIGMA="${ROI_PREPROCESS_BLUR_SIGMA:-2.5}"
ROI_PREPROCESS_FEATHER_RADIUS="${ROI_PREPROCESS_FEATHER_RADIUS:-48}"
ROI_PREPROCESS_BLEND="${ROI_PREPROCESS_BLEND:-0.60}"
ROI_PREPROCESS_CORRIDOR_TL="${ROI_PREPROCESS_CORRIDOR_TL:-0.20,0.45}"
ROI_PREPROCESS_CORRIDOR_TR="${ROI_PREPROCESS_CORRIDOR_TR:-0.80,0.45}"
ROI_PREPROCESS_CORRIDOR_BR="${ROI_PREPROCESS_CORRIDOR_BR:-1.0,1.0}"
ROI_PREPROCESS_CORRIDOR_BL="${ROI_PREPROCESS_CORRIDOR_BL:-0.0,1.0}"
ROI_PREPROCESS_ADAPTIVE="${ROI_PREPROCESS_ADAPTIVE:-0}"
ROI_PREPROCESS_CHROMA_ONLY="${ROI_PREPROCESS_CHROMA_ONLY:-0}"
ROI_PREPROCESS_MASK_FILE="${ROI_PREPROCESS_MASK_FILE:-}"
PRE_DENOISE="${PRE_DENOISE:-}"
TMP_ROOT="${TMPDIR:-/tmp}"
if [ ! -d "$TMP_ROOT" ]; then
  TMP_ROOT="/tmp"
fi
WORK_DIR="$(mktemp -d "${TMP_ROOT%/}/robust_current_archive.XXXXXX")"
ARCHIVE_DIR="$WORK_DIR/archive"
ARCHIVE_ZIP_TMP="$WORK_DIR/archive.zip"

cleanup() {
  rm -rf "$WORK_DIR"
}

trap cleanup EXIT

if [ "$ROI_ENABLE" = "1" ] && [ "$VIDEO_CODEC" != "libx265" ] && [ "$VIDEO_CODEC" != "libsvtav1" ]; then
  echo "ERROR: ROI packaging supports VIDEO_CODEC=libx265 or libsvtav1 only." >&2
  echo "Disable ROI or switch VIDEO_CODEC before packaging." >&2
  exit 1
fi

require_cmd() {
  local bin="$1"
  if ! command -v "$bin" >/dev/null 2>&1; then
    echo "ERROR: required tool not found in PATH: $bin" >&2
    exit 1
  fi
}

ffmpeg_encoder_available() {
  local encoder="$1"
  local encoders
  encoders="$("$FFMPEG_BIN" -hide_banner -encoders 2>/dev/null | tr -d '\r')"
  grep -q "$encoder" <<<"$encoders"
}

ffmpeg_filter_option_available() {
  local option="$1"
  local scale_help
  scale_help="$("$FFMPEG_BIN" -hide_banner -h filter=scale 2>/dev/null)"
  grep -q "$option" <<<"$scale_help"
}

require_ffmpeg_parity() {
  require_cmd "$FFMPEG_BIN"
  require_cmd "$FFPROBE_BIN"
  require_cmd "$UV_BIN"

  if [ "$VIDEO_CODEC" = "libsvtav1" ] && ! ffmpeg_encoder_available "libsvtav1"; then
    echo "ERROR: $FFMPEG_BIN does not provide the libsvtav1 encoder required by the AV1 lanes." >&2
    echo "Set FFMPEG_BIN/FFPROBE_BIN to a newer parity-compatible ffmpeg toolchain." >&2
    exit 1
  fi

  for opt in in_range out_range in_color_matrix out_color_matrix in_primaries out_primaries in_transfer out_transfer; do
    if ! ffmpeg_filter_option_available "$opt"; then
      echo "ERROR: $FFMPEG_BIN scale filter is missing required option '$opt' for the explicit color-contract path." >&2
      echo "This environment would drift from the canonical path. Set FFMPEG_BIN/FFPROBE_BIN to a parity-compatible ffmpeg build." >&2
      exit 1
    fi
  done
}

require_ffmpeg_parity


encode_video() {
  local in_path="$1"
  local out_path="$2"
  local vf_chain="$3"
  if [ "$VIDEO_CODEC" = "libsvtav1" ]; then
    "$FFMPEG_BIN" -y -i "$in_path" \
      -vf "$vf_chain" \
      -an -c:v libsvtav1 \
      -preset "$SVT_AV1_PRESET" \
      -crf "$SVT_AV1_CRF" \
      -svtav1-params "$SVT_AV1_PARAMS" \
      -color_range "$SOURCE_COLOR_RANGE" \
      -colorspace "$SOURCE_COLOR_MATRIX" \
      -color_primaries "$SOURCE_COLOR_PRIMARIES" \
      -color_trc "$SOURCE_COLOR_TRC" \
      -map_metadata -1 \
      "$out_path"
  else
    "$FFMPEG_BIN" -y -i "$in_path" \
      -vf "$vf_chain" \
      -an -c:v libx265 \
      -preset "$X265_PRESET" \
      -crf "$X265_CRF" \
      -x265-params "keyint=${X265_GOP}:min-keyint=${X265_GOP}:scenecut=0:bframes=${X265_BFRAMES}:ref=${X265_REF}" \
      -color_range "$SOURCE_COLOR_RANGE" \
      -colorspace "$SOURCE_COLOR_MATRIX" \
      -color_primaries "$SOURCE_COLOR_PRIMARIES" \
      -color_trc "$SOURCE_COLOR_TRC" \
      "$out_path"
  fi
}

codec_encode_args() {
  local crf_val="$1"
  if [ "$VIDEO_CODEC" = "libsvtav1" ]; then
    printf '%s' "-c:v libsvtav1 -preset ${SVT_AV1_PRESET} -crf ${crf_val} -svtav1-params ${SVT_AV1_PARAMS}"
  else
    printf '%s' "-c:v libx265 -preset ${X265_PRESET} -crf ${crf_val} -x265-params keyint=${X265_GOP}:min-keyint=${X265_GOP}:scenecut=0:bframes=${X265_BFRAMES}:ref=${X265_REF}"
  fi
}

downscale_filter() {
  local width="$1"
  local height="$2"
  local flags="$3"
  local base_filter
  base_filter="$(printf 'scale=%s:%s:flags=%s:in_range=%s:out_range=%s:in_color_matrix=%s:out_color_matrix=%s:in_primaries=%s:out_primaries=%s:in_transfer=%s:out_transfer=%s,format=yuv420p' \
    "$width" "$height" "$flags" \
    "$SOURCE_COLOR_RANGE" "$SOURCE_COLOR_RANGE" \
    "$SOURCE_COLOR_MATRIX" "$SOURCE_COLOR_MATRIX" \
    "$SOURCE_COLOR_PRIMARIES" "$SOURCE_COLOR_PRIMARIES" \
    "$SOURCE_COLOR_TRC" "$SOURCE_COLOR_TRC")"
  if [ -n "$PRE_DENOISE" ]; then
    printf '%s,%s' "$PRE_DENOISE" "$base_filter"
  else
    printf '%s' "$base_filter"
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

rm -rf "$SELF_DIR/archive" "$ARCHIVE_ZIP"
mkdir -p "$ARCHIVE_DIR"

while IFS= read -r rel; do
  [ -n "$rel" ] || continue
  in_path="$UPSTREAM_ROOT/videos/$rel"
  stem="${rel%.*}"

  # ROI preprocessing: degrade non-corridor regions before encoding
  if [ "$ROI_PREPROCESS_ENABLE" = "1" ]; then
    preprocessed_path="$WORK_DIR/${stem}_preprocessed.mkv"
    "$UV_BIN" run --with numpy --with scipy python "$ROI_PREPROCESS_SCRIPT" \
      --input "$in_path" \
      --output "$preprocessed_path" \
      --ffmpeg-bin "$FFMPEG_BIN" \
      --ffprobe-bin "$FFPROBE_BIN" \
      --corridor-top-left "$ROI_PREPROCESS_CORRIDOR_TL" \
      --corridor-top-right "$ROI_PREPROCESS_CORRIDOR_TR" \
      --corridor-bottom-right "$ROI_PREPROCESS_CORRIDOR_BR" \
      --corridor-bottom-left "$ROI_PREPROCESS_CORRIDOR_BL" \
      --outside-blur-sigma "$ROI_PREPROCESS_BLUR_SIGMA" \
      --feather-radius "$ROI_PREPROCESS_FEATHER_RADIUS" \
      --outside-blend "$ROI_PREPROCESS_BLEND" \
      $([ "$ROI_PREPROCESS_ADAPTIVE" = "1" ] && echo "--adaptive-mask") \
      $([ "$ROI_PREPROCESS_CHROMA_ONLY" = "1" ] && echo "--chroma-only") \
      $([ -n "$ROI_PREPROCESS_MASK_FILE" ] && echo "--mask-file $ROI_PREPROCESS_MASK_FILE")
    in_path="$preprocessed_path"
  fi

  if [ "$ROI_ENABLE" = "1" ]; then
    out_dir="$ARCHIVE_DIR/$stem"
    mkdir -p "$out_dir"
    if [ "$VIDEO_CODEC" = "libsvtav1" ]; then
      _codec_base_crf="$SVT_AV1_CRF"
    else
      _codec_base_crf="$X265_CRF"
    fi
    base_crf=$((_codec_base_crf + ROI_BASE_CRF_DELTA))
    roi_crf=$((_codec_base_crf + ROI_CRF_DELTA))
    roi2_crf=$((_codec_base_crf + ROI2_CRF_DELTA))

    if [ "$ROI_METADATA_ENABLE" = "1" ]; then
      metadata_path="$out_dir/roi_metadata.json"
      "$UV_BIN" run python "$ROI_SCRIPT_PY" analyze \
        --video "$in_path" \
        --ffmpeg-bin "$FFMPEG_BIN" \
        --ffprobe-bin "$FFPROBE_BIN" \
        --source-color-range "$SOURCE_COLOR_RANGE" \
        --source-color-matrix "$SOURCE_COLOR_MATRIX" \
        --source-color-primaries "$SOURCE_COLOR_PRIMARIES" \
        --source-color-trc "$SOURCE_COLOR_TRC" \
        --scale-w "$SCALE_W" \
        --scale-h "$SCALE_H" \
        --window-frames "$ROI_METADATA_WINDOW_FRAMES" \
        --sample-step "$ROI_METADATA_SAMPLE_STEP" \
        --tile-cols "$ROI_METADATA_TILE_COLS" \
        --tile-rows "$ROI_METADATA_TILE_ROWS" \
        --out "$metadata_path"
      encode_cmd=("$UV_BIN" run python "$ROI_SCRIPT_PY" encode-metadata
        --video "$in_path"
        --metadata "$metadata_path"
        --out-dir "$out_dir"
        --ffmpeg-bin "$FFMPEG_BIN"
        --scale-w "$SCALE_W"
        --scale-h "$SCALE_H"
        --downscale-flags "$DOWNSCALE_FLAGS"
        --source-color-range "$SOURCE_COLOR_RANGE"
        --source-color-matrix "$SOURCE_COLOR_MATRIX"
        --source-color-primaries "$SOURCE_COLOR_PRIMARIES"
        --source-color-trc "$SOURCE_COLOR_TRC"
        --codec "$VIDEO_CODEC"
        --base-crf "$base_crf"
        --roi-crf "$roi_crf"
        --roi2-crf "$roi2_crf")
      if [ "$VIDEO_CODEC" = "libsvtav1" ]; then
        encode_cmd+=(--svtav1-preset "$SVT_AV1_PRESET")
        encode_cmd+=(--svtav1-crf "$SVT_AV1_CRF")
        encode_cmd+=(--svtav1-params "$SVT_AV1_PARAMS")
      else
        x265_params="keyint=${X265_GOP}:min-keyint=${X265_GOP}:scenecut=0:bframes=${X265_BFRAMES}:ref=${X265_REF}"
        encode_cmd+=(--preset "$X265_PRESET")
        encode_cmd+=(--x265-params "$x265_params")
      fi
      if [ "$ROI2_ENABLE" = "1" ]; then
        encode_cmd+=(--roi2-enable)
      fi
      "${encode_cmd[@]}"
    else
      base_path="$out_dir/base.mkv"
      roi_path="$out_dir/roi.mkv"
      roi2_path="$out_dir/roi2.mkv"
      roi_w="$(calc_even_dim "$SCALE_W" "$ROI_W_FRAC")"
      roi_h="$(calc_even_dim "$SCALE_H" "$ROI_H_FRAC")"
      roi_x="$(calc_even_origin "$SCALE_W" "$ROI_X_FRAC" "$roi_w")"
      roi_y="$(calc_even_origin "$SCALE_H" "$ROI_Y_FRAC" "$roi_h")"

      if [ "$ROI2_ENABLE" = "1" ]; then
        roi2_w="$(calc_even_dim "$SCALE_W" "$ROI2_W_FRAC")"
        roi2_h="$(calc_even_dim "$SCALE_H" "$ROI2_H_FRAC")"
        roi2_x="$(calc_even_origin "$SCALE_W" "$ROI2_X_FRAC" "$roi2_w")"
        roi2_y="$(calc_even_origin "$SCALE_H" "$ROI2_Y_FRAC" "$roi2_h")"
      fi

      # Build codec-specific args for each CRF level
      read -ra base_codec_args <<< "$(codec_encode_args "$base_crf")"
      read -ra roi_codec_args  <<< "$(codec_encode_args "$roi_crf")"

      color_args=(-color_range "$SOURCE_COLOR_RANGE" -colorspace "$SOURCE_COLOR_MATRIX" -color_primaries "$SOURCE_COLOR_PRIMARIES" -color_trc "$SOURCE_COLOR_TRC")

      if [ "$ROI2_ENABLE" = "1" ]; then
        read -ra roi2_codec_args <<< "$(codec_encode_args "$roi2_crf")"
        echo "Encoding ROI two-pass+aux $in_path -> $base_path + $roi_path + $roi2_path"
        "$FFMPEG_BIN" -y -i "$in_path" \
          -filter_complex "[0:v]$(downscale_filter "$SCALE_W" "$SCALE_H" "$DOWNSCALE_FLAGS"),split=3[base][crop1][crop2];[crop1]crop=${roi_w}:${roi_h}:${roi_x}:${roi_y}[roi1];[crop2]crop=${roi2_w}:${roi2_h}:${roi2_x}:${roi2_y}[roi2]" \
          -map "[base]" -an "${base_codec_args[@]}" "${color_args[@]}" "$base_path" \
          -map "[roi1]" -an "${roi_codec_args[@]}" "${color_args[@]}" "$roi_path" \
          -map "[roi2]" -an "${roi2_codec_args[@]}" "${color_args[@]}" "$roi2_path"
      else
        echo "Encoding ROI two-pass $in_path -> $base_path + $roi_path"
        "$FFMPEG_BIN" -y -i "$in_path" \
          -filter_complex "[0:v]$(downscale_filter "$SCALE_W" "$SCALE_H" "$DOWNSCALE_FLAGS"),split=2[base][crop];[crop]crop=${roi_w}:${roi_h}:${roi_x}:${roi_y}[roi]" \
          -map "[base]" -an "${base_codec_args[@]}" "${color_args[@]}" "$base_path" \
          -map "[roi]" -an "${roi_codec_args[@]}" "${color_args[@]}" "$roi_path"
      fi
    fi
  else
    out_rel="${stem}.mkv"
    out_path="$ARCHIVE_DIR/$out_rel"
    mkdir -p "$(dirname "$out_path")"

    echo "Encoding $in_path -> $out_path"
    encode_video "$in_path" "$out_path" "$(downscale_filter "$SCALE_W" "$SCALE_H" "$DOWNSCALE_FLAGS")"
  fi
done < "$VIDEO_NAMES_FILE"

# Bundle neural network artifact inside archive (contest rules require it)
if [ -f "$SELF_DIR/postfilter_int8.pt" ]; then
  cp "$SELF_DIR/postfilter_int8.pt" "$ARCHIVE_DIR/postfilter_int8.pt"
  echo "Bundled postfilter_int8.pt ($(stat -f%z "$SELF_DIR/postfilter_int8.pt" 2>/dev/null || stat -c%s "$SELF_DIR/postfilter_int8.pt") bytes) into archive"
fi

# Extract and bundle PoseNet targets for supervised TTO at inflate time
# This pre-computes PoseNet(original) outputs so inflate can optimize against them
POSENET_TARGETS_ENABLE="${POSENET_TARGETS_ENABLE:-0}"
if [ "$POSENET_TARGETS_ENABLE" = "1" ]; then
  POSENET_TARGETS_OUT="$ARCHIVE_DIR/posenet_targets.bin"
  GT_VIDEO_PATH="${UPSTREAM_ROOT}/videos/0.mkv"
  POSENET_PATH="${UPSTREAM_ROOT}/models/posenet.safetensors"
  if [ -f "$GT_VIDEO_PATH" ] && [ -f "$POSENET_PATH" ]; then
    echo "Extracting PoseNet targets for supervised TTO ..."
    "$UV_BIN" run python -m tac.scorer_targets \
      --gt-video "$GT_VIDEO_PATH" \
      --posenet "$POSENET_PATH" \
      --upstream "$UPSTREAM_ROOT" \
      --output "$POSENET_TARGETS_OUT" \
      --device "${POSENET_TARGETS_DEVICE:-cpu}"
    if [ -f "$POSENET_TARGETS_OUT" ]; then
      echo "Bundled posenet_targets.bin ($(stat -f%z "$POSENET_TARGETS_OUT" 2>/dev/null || stat -c%s "$POSENET_TARGETS_OUT") bytes) into archive"
    else
      echo "WARNING: PoseNet target extraction failed" >&2
    fi
  else
    echo "WARNING: Cannot extract PoseNet targets - GT video or PoseNet model not found" >&2
  fi
elif [ -f "$SELF_DIR/posenet_targets.bin" ]; then
  # If pre-computed targets exist alongside the submission, bundle them
  cp "$SELF_DIR/posenet_targets.bin" "$ARCHIVE_DIR/posenet_targets.bin"
  echo "Bundled pre-computed posenet_targets.bin ($(stat -f%z "$SELF_DIR/posenet_targets.bin" 2>/dev/null || stat -c%s "$SELF_DIR/posenet_targets.bin") bytes) into archive"
fi

(
  cd "$ARCHIVE_DIR"
  zip -9 -r "$ARCHIVE_ZIP_TMP" .
)

mv "$ARCHIVE_ZIP_TMP" "$ARCHIVE_ZIP"

echo
echo "Wrote $ARCHIVE_ZIP"
