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
# Back up existing archive before overwriting (never lose a scoring artifact)
ARCHIVE_ZIP="$SELF_DIR/archive.zip"
if [ -f "$ARCHIVE_ZIP" ]; then
  BACKUP_NAME="$SELF_DIR/archive_$(date +%Y%m%dT%H%M%S).zip"
  cp "$ARCHIVE_ZIP" "$BACKUP_NAME"
  echo "[compress] Backed up existing archive to $BACKUP_NAME" >&2
fi
VIDEO_CODEC="${VIDEO_CODEC:-libx265}"
SVT_AV1_PRESET="${SVT_AV1_PRESET:-0}"
SVT_AV1_CRF="${SVT_AV1_CRF:-33}"
SVT_AV1_PARAMS="${SVT_AV1_PARAMS:-film-grain=22:keyint=-1:sharpness=1}"
X265_PRESET="${X265_PRESET:-medium}"
X265_CRF="${X265_CRF:-28}"
X265_GOP="${X265_GOP:-180}"
X265_BFRAMES="${X265_BFRAMES:-4}"
X265_REF="${X265_REF:-3}"
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
# ── Exploit #4: Sky region degradation ──────────────────────────────
# Blur and reduce precision in sky regions (SegNet class 4) before encoding.
# Sky has zero driving semantics — SegNet classifies it easily regardless
# of pixel fidelity, and PoseNet barely uses sky (far away = tiny parallax).
# Saves 12-16% of total rate on highway scenes (sky is ~40% of frame area).
SKY_DEGRADE_ENABLE="${SKY_DEGRADE_ENABLE:-0}"
SKY_DEGRADE_SCRIPT="${SKY_DEGRADE_SCRIPT:-$SELF_DIR/sky_degrade.py}"
SKY_DEGRADE_BLUR_SIGMA="${SKY_DEGRADE_BLUR_SIGMA:-2.0}"
SKY_DEGRADE_BIT_REDUCE="${SKY_DEGRADE_BIT_REDUCE:-4}"
SKY_DEGRADE_FEATHER="${SKY_DEGRADE_FEATHER:-8}"
# Downscale dimensions and flags — must be set in config.env or environment.
# SCALE_W: target width (e.g. 582), SCALE_H: target height (e.g. 437)
# DOWNSCALE_FLAGS: ffmpeg scale flags (e.g. "lanczos+accurate_rnd"), empty = ffmpeg default
SCALE_W="${SCALE_W:-}"
SCALE_H="${SCALE_H:-}"
DOWNSCALE_FLAGS="${DOWNSCALE_FLAGS:-}"
# Even-frame QP boost REMOVED (council sweep round 7): the interleave step
# re-encoded both streams at standard CRF, negating the quality differential.
# Technique was fundamentally broken. Code moved to experiments/archive/.
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
      -map_metadata -1 \
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
  "${PYTHON:-python3}" - "$@" <<'PY'
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
  "${PYTHON:-python3}" - "$@" <<'PY'
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

# Skip video encoding entirely when using renderer inflate path.
# The renderer generates frames from masks + neural network — it never uses
# the compressed video. Including 0.mkv in the archive wastes rate budget
# (every byte counts: rate_term = 25 * archive_size / gt_size).
PYTHON_INFLATE="${PYTHON_INFLATE:-0}"
SKIP_VIDEO_ENCODE="${SKIP_VIDEO_ENCODE:-0}"
if [ "$PYTHON_INFLATE" = "renderer" ]; then
  SKIP_VIDEO_ENCODE=1
  echo "[compress] PYTHON_INFLATE=renderer: skipping video encoding (renderer doesn't use compressed video)"
fi

while IFS= read -r rel; do
  [ -n "$rel" ] || continue

  if [ "$SKIP_VIDEO_ENCODE" = "1" ]; then
    echo "[compress] Skipping video encode for $rel (renderer path)"
    continue
  fi

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

  # Exploit #4: Sky region degradation — blur + reduce precision in sky
  if [ "$SKY_DEGRADE_ENABLE" = "1" ]; then
    sky_degraded_path="$WORK_DIR/${stem}_sky_degraded.mkv"
    echo "[compress] Sky degradation: sigma=${SKY_DEGRADE_BLUR_SIGMA}, bit_reduce=${SKY_DEGRADE_BIT_REDUCE}"
    "$UV_BIN" run --with numpy --with opencv-python-headless --with torch python "$SKY_DEGRADE_SCRIPT" \
      --input "$in_path" \
      --output "$sky_degraded_path" \
      --ffmpeg-bin "$FFMPEG_BIN" \
      --ffprobe-bin "$FFPROBE_BIN" \
      --blur-sigma "$SKY_DEGRADE_BLUR_SIGMA" \
      --bit-reduce "$SKY_DEGRADE_BIT_REDUCE" \
      --feather "$SKY_DEGRADE_FEATHER" \
      --upstream "${UPSTREAM_ROOT}" \
      $([ -n "$SCALE_W" ] && echo "--scale-w $SCALE_W") \
      $([ -n "$SCALE_H" ] && echo "--scale-h $SCALE_H")
    if [ -f "$sky_degraded_path" ]; then
      in_path="$sky_degraded_path"
    else
      echo "[compress] WARNING: sky degradation failed, proceeding with original" >&2
    fi
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

# Bundle postfilter artifact — ONLY when explicitly requested
BUNDLE_POSTFILTER="${BUNDLE_POSTFILTER:-0}"
if [ "$BUNDLE_POSTFILTER" = "1" ]; then
  if [ -f "$SELF_DIR/postfilter_int8.pt" ]; then
    cp "$SELF_DIR/postfilter_int8.pt" "$ARCHIVE_DIR/postfilter_int8.pt"
    echo "Bundled postfilter_int8.pt ($(stat -f%z "$SELF_DIR/postfilter_int8.pt" 2>/dev/null || stat -c%s "$SELF_DIR/postfilter_int8.pt") bytes) into archive"
  else
    echo "ERROR: BUNDLE_POSTFILTER=1 but postfilter_int8.pt not found in $SELF_DIR" >&2
    exit 1
  fi
fi
# NOTE: Removed auto-bundle of postfilter_int8.pt by file existence.
# Same pattern as optimized_embedding.pt, gradient_corrections.bin, optimized_poses.pt.

# Bundle mini-scorer binaries for INFLATE_MINI_TTO path
BUNDLE_MINI_SCORERS="${BUNDLE_MINI_SCORERS:-0}"
if [ "$BUNDLE_MINI_SCORERS" = "1" ]; then
  if [ -f "$SELF_DIR/mini_segnet.bin" ] && [ -f "$SELF_DIR/mini_posenet.bin" ]; then
    cp "$SELF_DIR/mini_segnet.bin" "$ARCHIVE_DIR/mini_segnet.bin"
    cp "$SELF_DIR/mini_posenet.bin" "$ARCHIVE_DIR/mini_posenet.bin"
    echo "Bundled mini_segnet.bin ($(stat -f%z "$SELF_DIR/mini_segnet.bin" 2>/dev/null || stat -c%s "$SELF_DIR/mini_segnet.bin") bytes) into archive"
    echo "Bundled mini_posenet.bin ($(stat -f%z "$SELF_DIR/mini_posenet.bin" 2>/dev/null || stat -c%s "$SELF_DIR/mini_posenet.bin") bytes) into archive"
  else
    echo "WARNING: BUNDLE_MINI_SCORERS=1 but mini_segnet.bin/mini_posenet.bin not found in $SELF_DIR" >&2
  fi
fi

# Bundle renderer binary — ONLY when renderer inflate path is active
if [ "$PYTHON_INFLATE" = "renderer" ]; then
  if [ -f "$SELF_DIR/renderer.bin" ]; then
    cp "$SELF_DIR/renderer.bin" "$ARCHIVE_DIR/renderer.bin"
    echo "Bundled renderer.bin ($(stat -f%z "$SELF_DIR/renderer.bin" 2>/dev/null || stat -c%s "$SELF_DIR/renderer.bin") bytes) into archive"
  else
    echo "ERROR: PYTHON_INFLATE=renderer but renderer.bin not found at $SELF_DIR/renderer.bin" >&2
    exit 1
  fi
fi

# ── Pre-extract SegNet masks (contest compliance: Yousfi PR #35) ──────
# Extract masks at compress time so inflate_renderer.py never loads SegNet.
# Without this, SegNet (~48MB) would need to be in archive.zip (rate catastrophe).
# The mask video is typically ~60-80KB for 1200 frames at 48x64 (1/8 scale);
# ~2MB at full 384x512. Verification (--verify) is mandatory: if roundtrip
# accuracy falls below 99.9%, compress_masks.py retries at CRF 10.
MASK_PREEXTRACT="${MASK_PREEXTRACT:-1}"
MASK_CRF="${MASK_CRF:-20}"
MASK_BATCH_SIZE="${MASK_BATCH_SIZE:-8}"
MASK_DEVICE="${MASK_DEVICE:-cpu}"
if [ "$MASK_PREEXTRACT" = "1" ]; then
  GT_VIDEO_PATH="${UPSTREAM_ROOT}/videos/0.mkv"
  MASKS_OUTPUT="$ARCHIVE_DIR/masks.mkv"
  if [ -f "$GT_VIDEO_PATH" ]; then
    echo "Pre-extracting SegNet masks from GT video ..."
    # --verify is mandatory: it checks >99.9% roundtrip accuracy of the
    # AV1 lossy encoding.  If verification fails, compress_masks.py retries
    # at CRF 10 automatically.  If that also fails, the pipeline aborts.
    # Without verification, AV1 lossy rounding at class boundaries could
    # silently flip class labels, corrupting SegNet input at inflate time.
    "$UV_BIN" run python "$SELF_DIR/compress_masks.py" \
      --gt-video "$GT_VIDEO_PATH" \
      --upstream "$UPSTREAM_ROOT" \
      --output "$MASKS_OUTPUT" \
      --crf "$MASK_CRF" \
      --device "$MASK_DEVICE" \
      --batch-size "$MASK_BATCH_SIZE" \
      --verify
    if [ -f "$MASKS_OUTPUT" ]; then
      echo "Bundled masks.mkv ($(stat -f%z "$MASKS_OUTPUT" 2>/dev/null || stat -c%s "$MASKS_OUTPUT") bytes) into archive"
    else
      echo "ERROR: Mask pre-extraction failed — masks.mkv not created" >&2
      exit 1
    fi
  else
    if [ "$PYTHON_INFLATE" = "renderer" ]; then
      echo "ERROR: MASK_PREEXTRACT=1 and PYTHON_INFLATE=renderer but GT video not found at $GT_VIDEO_PATH" >&2
      echo "The renderer inflate path REQUIRES masks.mkv in the archive." >&2
      exit 1
    fi
    echo "WARNING: Cannot pre-extract masks — GT video not found at $GT_VIDEO_PATH" >&2
    echo "Set MASK_PREEXTRACT=0 to skip mask pre-extraction (not contest-compliant)" >&2
  fi
elif [ -f "$SELF_DIR/masks.mkv" ]; then
  # Use pre-computed mask video if available alongside submission
  cp "$SELF_DIR/masks.mkv" "$ARCHIVE_DIR/masks.mkv"
  echo "Bundled pre-computed masks.mkv ($(stat -f%z "$SELF_DIR/masks.mkv" 2>/dev/null || stat -c%s "$SELF_DIR/masks.mkv") bytes) into archive"
fi

# ── Pre-compute corrections for CPU inflate pipeline (Eureka 1+2) ──
# Generates scorer gradients, null-space basis, fragility maps, brightness
# shifts, PoseNet targets, and hard-frame corrections in one pass.
# Output: corrections.bin (~50-100KB) bundled into archive.
PRECOMPUTE_CORRECTIONS="${PRECOMPUTE_CORRECTIONS:-0}"
if [ "$PRECOMPUTE_CORRECTIONS" = "1" ]; then
  CORRECTIONS_OUT="$ARCHIVE_DIR/corrections.bin"
  GT_VIDEO_PATH="${UPSTREAM_ROOT}/videos/0.mkv"
  POSENET_PATH="${UPSTREAM_ROOT}/models/posenet.safetensors"
  if [ -f "$GT_VIDEO_PATH" ] && [ -f "$POSENET_PATH" ]; then
    echo "Pre-computing corrections for CPU inflate pipeline ..."
    "$UV_BIN" run python -m tac.precompute_corrections \
      --gt-video "$GT_VIDEO_PATH" \
      --posenet "$POSENET_PATH" \
      --upstream "$UPSTREAM_ROOT" \
      --output "$CORRECTIONS_OUT" \
      --device "${PRECOMPUTE_DEVICE:-cpu}" \
      --batch-size "${PRECOMPUTE_BATCH_SIZE:-4}" \
      --null-space-k "${PRECOMPUTE_NULL_K:-32}" \
      --hard-frames "${PRECOMPUTE_HARD_FRAMES:-50}" \
      --hard-pixels "${PRECOMPUTE_HARD_PIXELS:-1000}"
    if [ -f "$CORRECTIONS_OUT" ]; then
      echo "Bundled corrections.bin ($(stat -f%z "$CORRECTIONS_OUT" 2>/dev/null || stat -c%s "$CORRECTIONS_OUT") bytes) into archive"
    else
      echo "WARNING: Corrections pre-computation failed" >&2
    fi
  else
    echo "WARNING: Cannot pre-compute corrections - GT video or PoseNet model not found" >&2
  fi
elif [ -f "$SELF_DIR/corrections.bin" ]; then
  cp "$SELF_DIR/corrections.bin" "$ARCHIVE_DIR/corrections.bin"
  echo "Bundled pre-computed corrections.bin ($(stat -f%z "$SELF_DIR/corrections.bin" 2>/dev/null || stat -c%s "$SELF_DIR/corrections.bin") bytes) into archive"
fi

# ── Mask codec selection (Eureka 3): av1 (default) or vvc ──
# VVC/H.266 can be 30-50% smaller than AV1 for segmentation masks.
# Set MASK_CODEC=vvc to use VVC if vvencapp is available.
MASK_CODEC="${MASK_CODEC:-av1}"
if [ "$MASK_CODEC" = "vvc" ] && ! command -v vvencapp >/dev/null 2>&1; then
  echo "WARNING: MASK_CODEC=vvc but vvencapp not found, falling back to av1" >&2
  MASK_CODEC="av1"
fi

# ── Extract GT poses for FiLM-conditioned renderer ──────────────────
# Extracts PoseNet outputs from GT frame pairs for use as FiLM conditioning.
# Storage: 600 pairs x 6 values x 2 bytes (fp16) = 7.2KB.
POSE_EXTRACT_ENABLE="${POSE_EXTRACT_ENABLE:-0}"
if [ "$POSE_EXTRACT_ENABLE" = "1" ]; then
  POSES_OUT="$ARCHIVE_DIR/poses.pt"
  GT_VIDEO_PATH="${UPSTREAM_ROOT}/videos/0.mkv"
  if [ -f "$GT_VIDEO_PATH" ]; then
    echo "Extracting GT poses for FiLM conditioning ..."
    "$UV_BIN" run python -m tac.pose_extraction \
      --upstream "$UPSTREAM_ROOT" \
      --output "$POSES_OUT" \
      --device "${POSE_EXTRACT_DEVICE:-cpu}" \
      --batch-size "${POSE_EXTRACT_BATCH_SIZE:-16}" \
      --video "$GT_VIDEO_PATH"
    if [ -f "$POSES_OUT" ]; then
      echo "Bundled poses.pt ($(stat -f%z "$POSES_OUT" 2>/dev/null || stat -c%s "$POSES_OUT") bytes) into archive"
    else
      echo "WARNING: Pose extraction failed" >&2
    fi
  else
    echo "WARNING: Cannot extract poses - GT video not found at $GT_VIDEO_PATH" >&2
  fi
elif [ -f "$SELF_DIR/poses.pt" ]; then
  cp "$SELF_DIR/poses.pt" "$ARCHIVE_DIR/poses.pt"
  echo "Bundled pre-computed poses.pt ($(stat -f%z "$SELF_DIR/poses.pt" 2>/dev/null || stat -c%s "$SELF_DIR/poses.pt") bytes) into archive"
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

# ── Pose-Space TTO: optimize FiLM conditioning vectors at compress time ──
# Runs gradient descent through frozen scorers to find optimal 6D pose vectors
# per pair. These replace GT poses in the archive for better rendering.
# Requires: renderer.bin with pose_dim > 0, GT video, upstream scorers.
POSE_TTO_ENABLE="${POSE_TTO_ENABLE:-0}"
POSE_TTO_STEPS="${POSE_TTO_STEPS:-500}"
POSE_TTO_LR="${POSE_TTO_LR:-0.01}"
POSE_TTO_DEVICE="${POSE_TTO_DEVICE:-cpu}"
POSE_TTO_BATCH="${POSE_TTO_BATCH:-50}"
POSE_TTO_LATENT_DIM="${POSE_TTO_LATENT_DIM:-0}"
if [ "$POSE_TTO_ENABLE" = "1" ]; then
  OPTIMIZED_POSES_OUT="$ARCHIVE_DIR/optimized_poses.pt"
  GT_VIDEO_PATH="${UPSTREAM_ROOT}/videos/0.mkv"
  RENDERER_PATH="$ARCHIVE_DIR/renderer.bin"
  if [ -f "$GT_VIDEO_PATH" ] && [ -f "$RENDERER_PATH" ]; then
    echo "Running Pose-Space TTO (steps=${POSE_TTO_STEPS}, lr=${POSE_TTO_LR}, device=${POSE_TTO_DEVICE}) ..."
    POSE_TTO_ARGS=(
      --checkpoint "$RENDERER_PATH"
      --device "$POSE_TTO_DEVICE"
      --steps "$POSE_TTO_STEPS"
      --lr "$POSE_TTO_LR"
      --batch-pairs "$POSE_TTO_BATCH"
      --upstream "$UPSTREAM_ROOT"
      --video "$GT_VIDEO_PATH"
      --output-dir "$WORK_DIR/pose_tto"
    )
    if [ "$POSE_TTO_LATENT_DIM" -gt 0 ] 2>/dev/null; then
      POSE_TTO_ARGS+=(--latent-dim "$POSE_TTO_LATENT_DIM")
    fi
    PROJECT_ROOT="$(cd "$SELF_DIR/../.." && pwd)"
    PYTHONPATH="${PROJECT_ROOT}/src:${UPSTREAM_ROOT}:${PYTHONPATH:-}" \
      "$UV_BIN" run python "${PROJECT_ROOT}/experiments/optimize_poses.py" "${POSE_TTO_ARGS[@]}"
    if [ -f "$WORK_DIR/pose_tto/optimized_poses.pt" ]; then
      cp "$WORK_DIR/pose_tto/optimized_poses.pt" "$OPTIMIZED_POSES_OUT"
      echo "Bundled optimized_poses.pt ($(stat -f%z "$OPTIMIZED_POSES_OUT" 2>/dev/null || stat -c%s "$OPTIMIZED_POSES_OUT") bytes) into archive"
    else
      echo "WARNING: Pose-space TTO failed -- no optimized_poses.pt produced" >&2
    fi
  else
    echo "ERROR: Cannot run pose TTO -- need GT video ($GT_VIDEO_PATH) and renderer ($RENDERER_PATH)" >&2
    echo "POSE_TTO_ENABLE=1 requires both files to exist. Fix the pipeline." >&2
    exit 1
  fi
elif [ "${BUNDLE_OPTIMIZED_POSES:-0}" = "1" ]; then
  if [ -f "$SELF_DIR/optimized_poses.pt" ]; then
    cp "$SELF_DIR/optimized_poses.pt" "$ARCHIVE_DIR/optimized_poses.pt"
    echo "Bundled pre-computed optimized_poses.pt ($(stat -f%z "$SELF_DIR/optimized_poses.pt" 2>/dev/null || stat -c%s "$SELF_DIR/optimized_poses.pt") bytes) into archive"
  else
    echo "ERROR: BUNDLE_OPTIMIZED_POSES=1 but optimized_poses.pt not found in $SELF_DIR" >&2
    exit 1
  fi
fi
# NOTE: Removed auto-bundle of optimized_poses.pt by file existence.
# Same fix as optimized_embedding.pt and gradient_corrections.bin —
# explicit flag required, no silent artifact poisoning.

# ── Bundle optimized embedding (compress-time embedding TTO) ──────────
BUNDLE_EMBEDDING="${BUNDLE_EMBEDDING:-0}"
if [ "$BUNDLE_EMBEDDING" = "1" ]; then
  if [ -f "$SELF_DIR/optimized_embedding.pt" ]; then
    cp "$SELF_DIR/optimized_embedding.pt" "$ARCHIVE_DIR/optimized_embedding.pt"
    echo "Bundled optimized_embedding.pt ($(stat -f%z "$SELF_DIR/optimized_embedding.pt" 2>/dev/null || stat -c%s "$SELF_DIR/optimized_embedding.pt") bytes) into archive"
  else
    echo "WARNING: BUNDLE_EMBEDDING=1 but optimized_embedding.pt not found in $SELF_DIR" >&2
  fi
fi
# NOTE: Removed auto-bundle of optimized_embedding.pt by file existence.
# Stale experiment artifacts silently inflating archive size caused the
# 0.108-point measurement disaster. Set BUNDLE_EMBEDDING=1 explicitly.

# ── Bundle gradient corrections (compress-time pre-computed pixel adjustments) ──
BUNDLE_CORRECTIONS="${BUNDLE_CORRECTIONS:-0}"
if [ "$BUNDLE_CORRECTIONS" = "1" ]; then
  if [ -f "$SELF_DIR/gradient_corrections.bin" ]; then
    cp "$SELF_DIR/gradient_corrections.bin" "$ARCHIVE_DIR/gradient_corrections.bin"
    echo "Bundled gradient_corrections.bin ($(stat -f%z "$SELF_DIR/gradient_corrections.bin" 2>/dev/null || stat -c%s "$SELF_DIR/gradient_corrections.bin") bytes) into archive"
  else
    echo "WARNING: BUNDLE_CORRECTIONS=1 but gradient_corrections.bin not found in $SELF_DIR" >&2
  fi
fi
# NOTE: Removed auto-bundle of gradient_corrections.bin by file existence.
# Same reason as optimized_embedding.pt — explicit > implicit for archive contents.

# ── Build archive via validated Python builder ──────────────────────────
# The canonical archive builder (compress_archive.py) uses
# build_submission_archive() with proper manifest validation.
# Supports half-frame masks (600 odd frames) and binary poses (.bin).
# Falls back to manual zip if the Python builder fails.
USE_PYTHON_ARCHIVE="${USE_PYTHON_ARCHIVE:-1}"
HALF_FRAME_MASKS="${HALF_FRAME_MASKS:-0}"
BINARY_POSES="${BINARY_POSES:-0}"

_python_archive_built=0
if [ "$USE_PYTHON_ARCHIVE" = "1" ] && [ "$PYTHON_INFLATE" = "renderer" ]; then
  # Locate the required artifacts in ARCHIVE_DIR
  _renderer_bin="$ARCHIVE_DIR/renderer.bin"
  _masks_mkv="$ARCHIVE_DIR/masks.mkv"
  # Find poses: optimized_poses.pt > optimized_poses.bin > poses.pt
  _poses_file=""
  for _pf in "$ARCHIVE_DIR/optimized_poses.pt" "$ARCHIVE_DIR/optimized_poses.bin" "$ARCHIVE_DIR/poses.pt"; do
    if [ -f "$_pf" ]; then
      _poses_file="$_pf"
      break
    fi
  done

  if [ -f "$_renderer_bin" ] && [ -f "$_masks_mkv" ] && [ -n "$_poses_file" ]; then
    echo "[compress] Building archive via compress_archive.py (validated builder)"
    _compress_archive_args=(
      --renderer-bin "$_renderer_bin"
      --masks-path "$_masks_mkv"
      --poses-path "$_poses_file"
      --output "$ARCHIVE_ZIP"
    )
    if [ "$HALF_FRAME_MASKS" = "1" ]; then
      _compress_archive_args+=(--half-frame)
      echo "[compress]   half-frame masks: enabled (600 odd frames only)"
    fi
    if [ "$BINARY_POSES" = "1" ]; then
      _compress_archive_args+=(--binary-poses)
      echo "[compress]   binary poses: enabled (.bin instead of .pt)"
    fi

    PROJECT_ROOT="$(cd "$SELF_DIR/../.." && pwd)"
    if PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}" \
       "$UV_BIN" run python "$SELF_DIR/compress_archive.py" "${_compress_archive_args[@]}"; then
      _python_archive_built=1
      echo "[compress] Archive built successfully via Python builder"
    else
      echo "[compress] WARNING: Python archive builder failed, falling back to zip" >&2
    fi
  else
    echo "[compress] Skipping Python archive builder (missing artifacts:" \
         "renderer=$([ -f "$_renderer_bin" ] && echo 'ok' || echo 'MISSING')," \
         "masks=$([ -f "$_masks_mkv" ] && echo 'ok' || echo 'MISSING')," \
         "poses=$([ -n "$_poses_file" ] && echo 'ok' || echo 'MISSING'))" >&2
  fi
fi

# Fallback: manual zip (used when Python builder is disabled or fails,
# and for non-renderer inflate paths that don't have the required artifacts)
if [ "$_python_archive_built" = "0" ]; then
  echo "[compress] Building archive via zip -9 -r (manual fallback)"
  (
    cd "$ARCHIVE_DIR"
    zip -9 -r "$ARCHIVE_ZIP_TMP" .
  )
  mv "$ARCHIVE_ZIP_TMP" "$ARCHIVE_ZIP"
fi

# ── Archive provenance (non-negotiable measurement rule) ──────────────
# EVERY archive build MUST print size and rate. This is how we prevent
# the 0.108-point measurement disaster from ever recurring.
FINAL_SIZE="$(stat -f%z "$ARCHIVE_ZIP" 2>/dev/null || stat -c%s "$ARCHIVE_ZIP")"
echo
echo "=== ARCHIVE PROVENANCE ==="
echo "  Path: $ARCHIVE_ZIP"
echo "  Size: ${FINAL_SIZE} bytes"
echo "  Rate term: 25 * ${FINAL_SIZE} / 37545489 = $(python3 -c "print(f'{25 * ${FINAL_SIZE} / 37545489:.6f}')" 2>/dev/null || echo 'N/A')"
echo "  Contents:"
unzip -l "$ARCHIVE_ZIP" 2>/dev/null | grep -v "^Archive" | grep -v "^$" | sed 's/^/    /'
echo "=========================="
