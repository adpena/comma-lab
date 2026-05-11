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

INFLATE_BROTLI_SPEC="${INFLATE_BROTLI_SPEC:-brotli==1.2.0}"
INFLATE_AV_SPEC="${INFLATE_AV_SPEC:-av==17.0.1}"
INFLATE_TORCH_SPEC="${INFLATE_TORCH_SPEC:-torch==2.11.0}"
INFLATE_NUMPY_SPEC="${INFLATE_NUMPY_SPEC:-numpy==2.4.4}"

UV_WITH_BROTLI=(--with "$INFLATE_BROTLI_SPEC")
UV_WITH_AV_TORCH_NUMPY=(
  --with "$INFLATE_AV_SPEC"
  --with "$INFLATE_TORCH_SPEC"
  --with "$INFLATE_NUMPY_SPEC"
)
UV_WITH_RENDERER_DEPS=(
  "${UV_WITH_BROTLI[@]}"
  "${UV_WITH_AV_TORCH_NUMPY[@]}"
)

echo "[inflate] uv dependency specs: brotli=$INFLATE_BROTLI_SPEC av=$INFLATE_AV_SPEC torch=$INFLATE_TORCH_SPEC numpy=$INFLATE_NUMPY_SPEC" >&2

# ============================================================
# Stage 0: brotli decompression (codex R5-3 fix, 2026-04-27)
# Centralized .br -> sibling decompression BEFORE PYTHON_INFLATE
# branch dispatch. Previously --with brotli + the actionable
# ImportError diagnostic existed only on the PYTHON_INFLATE=renderer
# arm; any other branch (1 / postfilter / grain_mask / future) that
# encountered a .br archive would fail later as a missing
# renderer.bin / masks.mkv with no actionable hint.
#
# OPTION A (centralized): we decompress all .br -> sibling files in
# $ARCHIVE_DIR up front using a small inline python that imports
# only `brotli` (NOT `tac`, so it works in any clean contest env
# where `uv run` resolves the contest's pyproject, not ours).
# After this step every downstream branch sees the archive in its
# fully-decompressed form regardless of build mode.
#
# This is a no-op when no .br files exist (the `if` guard avoids the
# uv-cold-start cost on the common Lane A path). The renderer branch
# keeps its inline _decompress_brotli_in_archive() call as defense-
# in-depth for direct python invocation paths.
# ============================================================
if compgen -G "$ARCHIVE_DIR"/*.br > /dev/null 2>&1 \
    || compgen -G "$ARCHIVE_DIR"/**/*.br > /dev/null 2>&1; then
  echo "[brotli stage 0] .br files detected in $ARCHIVE_DIR; decompressing before inflate dispatch..." >&2
  if ! "$UV_BIN" run "${UV_WITH_BROTLI[@]}" python - "$ARCHIVE_DIR" <<'PY'
import sys
from pathlib import Path

archive_dir = Path(sys.argv[1])
if not archive_dir.is_dir():
    print(f"FATAL: archive dir does not exist: {archive_dir}", file=sys.stderr)
    sys.exit(2)

br_files = sorted(archive_dir.rglob("*.br"))
if not br_files:
    print("[brotli stage 0] no .br files at decompress time (race?)", file=sys.stderr)
    sys.exit(0)

try:
    import brotli
except ImportError:
    listing = ", ".join(str(p.relative_to(archive_dir)) for p in br_files)
    print(
        "FATAL: Archive contains Brotli-compressed files (.br) but the "
        "'brotli' package is not installed in the active Python "
        f"environment.\n  Files needing decompression: {listing}\n"
        "  Fix: `pip install brotli` (or `uv pip install brotli`) in "
        "the same env that runs inflate.sh.\n"
        "  Note: `brotli` is declared as a mandatory dependency of the "
        "`tac` package (pyproject.toml [project].dependencies). The "
        "inflate.sh Stage 0 invocation also passes `--with brotli` to "
        "`uv run` for clean contest envs; if you see this in such an env, "
        "uv may have failed to install the wheel — re-run with "
        "`UV_BIN=$(which uv) bash inflate.sh ...` and inspect uv output.",
        file=sys.stderr,
    )
    sys.exit(1)

n = 0
for br_file in br_files:
    out_path = br_file.with_suffix("")  # strip .br
    data = br_file.read_bytes()
    decompressed = brotli.decompress(data)
    out_path.write_bytes(decompressed)
    ratio = (len(data) / len(decompressed) * 100) if len(decompressed) else 0.0
    print(
        f"  {br_file.relative_to(archive_dir)} -> {out_path.relative_to(archive_dir)}: "
        f"{len(data):,}B -> {len(decompressed):,}B ({ratio:.1f}%)",
        file=sys.stderr,
    )
    br_file.unlink()
    n += 1
print(f"[brotli stage 0] decompressed {n} file(s)", file=sys.stderr)
PY
  then
    echo "FATAL: brotli stage 0 decompression failed; refusing to dispatch inflate." >&2
    echo "       See diagnostic above. This is the codex R5-3 centralized" >&2
    echo "       brotli step; without it any non-renderer PYTHON_INFLATE branch" >&2
    echo "       would fail later with a less actionable 'missing file' error." >&2
    exit 1
  fi
fi

if [ -f "$ARCHIVE_DIR/renderer_payload.bin" ] \
    || [ -f "$ARCHIVE_DIR/renderer_payload.bin.br" ] \
    || [ -f "$ARCHIVE_DIR/p" ]; then
  echo "[inflate] renderer payload detected; expanding into renderer members" >&2
  if ! "$UV_BIN" run "${UV_WITH_BROTLI[@]}" python \
      "$SELF_DIR/unpack_renderer_payload.py" "$ARCHIVE_DIR" \
      --summary-json "$ARCHIVE_DIR/renderer_payload_unpack_summary.json"; then
    echo "FATAL: renderer payload unpack failed; refusing to dispatch inflate." >&2
    exit 1
  fi
fi

if [ -f "$ARCHIVE_DIR/jcsp.bin" ]; then
  JCSP_RUNTIME_BRIDGE_MODE="${JCSP_RUNTIME_BRIDGE_MODE:-consume-real-raw-outputs}"
  JCSP_RUNTIME_PROBE_MANIFEST="$INFLATED_DIR/jcsp_runtime_probe_manifest.json"
  JCSP_RUNTIME_PARITY_MANIFEST="$INFLATED_DIR/jcsp_runtime_raw_output_parity_manifest.json"
  echo "[inflate] jcsp.bin detected; running JCSP runtime bridge mode=$JCSP_RUNTIME_BRIDGE_MODE" >&2
  JCSP_RUNTIME_BRIDGE_ARGS=(
      "$SELF_DIR/jcsp_runtime_bridge.py" "$ARCHIVE_DIR"
      --mode "$JCSP_RUNTIME_BRIDGE_MODE"
      --inflated-dir "$INFLATED_DIR" \
      --output-dir "$INFLATED_DIR" \
      --video-names-file "$VIDEO_NAMES_FILE" \
      --manifest-json "$JCSP_RUNTIME_PROBE_MANIFEST" \
      --parity-manifest-json "$JCSP_RUNTIME_PARITY_MANIFEST"
  )
  if [ -n "${JCSP_REFERENCE_RAW_DIR:-}" ]; then
    JCSP_RUNTIME_BRIDGE_ARGS+=(--reference-raw-dir "$JCSP_REFERENCE_RAW_DIR")
  fi
  if ! "${PYTHON:-python3}" "${JCSP_RUNTIME_BRIDGE_ARGS[@]}"; then
    echo "FATAL: JCSP runtime bridge refused jcsp.bin; refusing to dispatch inflate." >&2
    echo "       Probe manifest: $JCSP_RUNTIME_PROBE_MANIFEST" >&2
    exit 44
  fi
  echo "[inflate] jcsp.bin consumed by JCSP runtime bridge; emitted contest .raw outputs" >&2
  exit 0
fi

if [ "$PYTHON_INFLATE" = "renderer" ] \
    && [ -f "$ARCHIVE_DIR/grayscale.mkv" ] \
    && [ ! -f "$ARCHIVE_DIR/masks.mkv" ]; then
  echo "[inflate] auto-selecting PYTHON_INFLATE=renderer_grayscale for grayscale-only archive" >&2
  PYTHON_INFLATE="renderer_grayscale"
fi


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
  ${PYTHON:-python3} - "$@" <<'PY'
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
  ${PYTHON:-python3} - "$@" <<'PY'
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

# 2026-04-28: defense-in-depth root-cause fix for the Lane RM-d 0.mkv crash.
# If config.env was somehow not sourced (operator bug, deploy regression, or
# a fresh contest env that doesn't ship config.env), PYTHON_INFLATE defaults
# to "0" → the ffmpeg branch fires → tries to read $ARCHIVE_DIR/0.mkv →
# crashes because a renderer archive contains renderer.bin + masks.mkv +
# optimized_poses.pt, NOT a per-video 0.mkv. Auto-detect this archive shape
# and refuse to dispatch to ffmpeg with an actionable error pointing at the
# canonical fix (Codex F5 + Check 64 E2E smoke).
if [ "$PYTHON_INFLATE" = "0" ] && [ "$ROI_ENABLE" = "0" ]; then
  if [ -f "$ARCHIVE_DIR/renderer.bin" ] || [ -f "$ARCHIVE_DIR/renderer.bin.br" ]; then
    echo "FATAL: renderer.bin* present in $ARCHIVE_DIR but PYTHON_INFLATE != renderer." >&2
    echo "       This is a renderer archive — the ffmpeg branch would crash trying" >&2
    echo "       to read \$ARCHIVE_DIR/<video>.mkv (Lane RM-d's 0.mkv crash, 2026-04-28)." >&2
    echo "       Likely cause: config.env missing or PYTHON_INFLATE=renderer not set." >&2
    echo "       Fix: source $SELF_DIR/config.env OR export PYTHON_INFLATE=renderer." >&2
    echo "       Permanent guard: experiments/canonical_local_auth_eval_smoke.py" >&2
    echo "                        (Check 64 — written by canonical_local_auth_eval_smoke.py)." >&2
    exit 4
  fi
fi

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
    if [ "$PYTHON_INFLATE" = "renderer" ]; then
      echo "Inflating (neural renderer: mask→frame) $ARCHIVE_DIR -> $INFLATED_DIR"
      # Codex R5-2 #3 fix (2026-04-27, Lane B-alt deployment blocker):
      # `--with brotli==...` makes the inflate path dependency-complete in a
      # CLEAN CONTEST ENVIRONMENT. The contest evaluator invokes inflate.sh
      # from inside the contest's own pyproject (upstream/pyproject.toml),
      # so `uv run` discovers THAT project's deps — NOT our `tac` package.
      # Without --with brotli, any Lane B-alt archive (renderer.bin.br,
      # masks.mkv.br, …) would fatal-exit at _decompress_brotli_in_archive
      # the first time .br files are seen. brotli is a tiny pure-C wheel;
      # the cost of always-pulling is sub-second on T4. Pinned `--with`
      # specs prevent silent PyPI resolver drift across eval machines.
      "$UV_BIN" run "${UV_WITH_RENDERER_DEPS[@]}" python "$SELF_DIR/inflate_renderer.py" \
        "$ARCHIVE_DIR" "$INFLATED_DIR" "$VIDEO_NAMES_FILE"
      if [ -f "$ARCHIVE_DIR/qpost.bin" ]; then
        echo "[inflate] qpost.bin detected; applying counted QZS3 postprocess atoms" >&2
        "$UV_BIN" run "${UV_WITH_RENDERER_DEPS[@]}" python "$SELF_DIR/apply_qzs3_postprocess.py" \
          "$ARCHIVE_DIR/qpost.bin" "$INFLATED_DIR" "$VIDEO_NAMES_FILE"
      fi
      break
    elif [ "$PYTHON_INFLATE" = "postfilter" ]; then
      echo "Inflating (canonical + learned post-filter) $ARCHIVE_DIR -> $INFLATED_DIR"
      # POSTFILTER_PATH must resolve from ARCHIVE_DIR (contest rules: neural
      # artifacts inside archive.zip). NEVER fall back to SELF_DIR — that hides
      # packaging bugs where compress.sh forgot to bundle the checkpoint.
      "$UV_BIN" run python "$SELF_DIR/inflate_postfilter.py" \
        "$ARCHIVE_DIR" "$INFLATED_DIR" "$VIDEO_NAMES_FILE" \
        "${POSTFILTER_PATH:-$ARCHIVE_DIR/postfilter_int8.pt}"
      break
    elif [ "$PYTHON_INFLATE" = "grain_mask" ]; then
      echo "Inflating (saliency-masked grain) $ARCHIVE_DIR -> $INFLATED_DIR"
      "$UV_BIN" run "${UV_WITH_AV_TORCH_NUMPY[@]}" python "$SELF_DIR/inflate_grain_mask.py" \
        "$ARCHIVE_DIR" "$INFLATED_DIR" "$VIDEO_NAMES_FILE" \
        "${GRAIN_MASK_SALIENCY:-experiments/masks/posenet_saliency.npy}" \
        "${GRAIN_MASK_STRENGTH:-8.0}"
      break
    elif [ "$PYTHON_INFLATE" = "renderer_grayscale" ]; then
      # Lane MM (2026-04-29): Selfcomp grayscale-LUT mask decode -> existing
      # MaskRenderer. Same renderer.bin as Lane A; the only delta is that
      # masks.mkv is replaced by grayscale.mkv (1-channel AV1 monochrome
      # with Selfcomp class targets [0, 255, 64, 192, 128] + sigma=15
      # Gaussian softmax LUT). Predicted [0.65, 0.85] [contest-CUDA].
      echo "Inflating (Lane MM: grayscale-LUT mask + existing renderer) $ARCHIVE_DIR -> $INFLATED_DIR"
      "$UV_BIN" run "${UV_WITH_RENDERER_DEPS[@]}" python "$SELF_DIR/inflate_renderer_grayscale.py" \
        "$ARCHIVE_DIR" "$INFLATED_DIR" "$VIDEO_NAMES_FILE"
      break
    elif [ "$PYTHON_INFLATE" = "segmap" ]; then
      # Selfcomp paradigm dispatch arm (Lane SA / SC++ / SO / future).
      # Decodes grayscale.mkv via Gaussian softmax LUT -> SegMap renderer
      # (tac.segmap_renderer) -> bicubic upsample to camera resolution.
      # SegMap weights live in payload.tar.xz (block-FP per-channel, see
      # tac.block_fp_codec.pack_payload_tar_xz) OR a raw segmap_inference.pt.
      # CLAUDE.md strict-scorer-rule honored: inflate_segmap.py loads ONLY the SegMap renderer (no PoseNet, no SegNet). # SCORER_AT_INFLATE_WAIVED: documentation-line referencing strict-scorer-rule by name; no actual scorer load happens here.
      echo "Inflating (Selfcomp SegMap: grayscale-LUT -> SegMap) $ARCHIVE_DIR -> $INFLATED_DIR"
      "$UV_BIN" run "${UV_WITH_RENDERER_DEPS[@]}" python "$SELF_DIR/inflate_segmap.py" \
        "$ARCHIVE_DIR" "$INFLATED_DIR" "$VIDEO_NAMES_FILE"
      break
    elif [ "$PYTHON_INFLATE" = "segmap_film_canvas" ]; then
      # Lane FC dispatch arm (EUREKA #5, 2026-04-29).
      # Same archive layout as segmap, but the state_dict carries a
      # `film_table.weight` Embedding key so SegMap is loaded as
      # SegMapFilmCanvas with per-frame FiLM modulation on layer_in.
      # Auto-falls back to vanilla SegMap if the film_table key is absent.
      # CLAUDE.md strict-scorer-rule honored: inflate_segmap_film_canvas.py loads ONLY the SegMap renderer. # SCORER_AT_INFLATE_WAIVED: documentation-line referencing strict-scorer-rule by name; no actual scorer load happens here.
      echo "Inflating (Lane FC: SegMap+FiLM-Canvas) $ARCHIVE_DIR -> $INFLATED_DIR"
      "$UV_BIN" run "${UV_WITH_RENDERER_DEPS[@]}" python "$SELF_DIR/inflate_segmap_film_canvas.py" \
        "$ARCHIVE_DIR" "$INFLATED_DIR" "$VIDEO_NAMES_FILE"
      break
    elif [ "$PYTHON_INFLATE" = "segmap_arithmetic" ]; then
      # Lane SH dispatch arm (Shannon arithmetic coding, 2026-04-29).
      # Same archive layout as segmap, except the SegMap weights are
      # arithmetic-coded into payload.bin instead of tar.xz; the payload is
      # decompressed via tac.arithmetic_qint_codec, then handed to the
      # standard SegMap renderer.
      # CLAUDE.md strict-scorer-rule honored: inflate_segmap_arithmetic.py loads ONLY the SegMap renderer. # SCORER_AT_INFLATE_WAIVED: documentation-line referencing strict-scorer-rule by name; no actual scorer load happens here.
      echo "Inflating (Lane SH: arithmetic-coded SegMap weights) $ARCHIVE_DIR -> $INFLATED_DIR"
      "$UV_BIN" run "${UV_WITH_RENDERER_DEPS[@]}" python "$SELF_DIR/inflate_segmap_arithmetic.py" \
        "$ARCHIVE_DIR" "$INFLATED_DIR" "$VIDEO_NAMES_FILE"
      break
    elif [ "$PYTHON_INFLATE" = "1" ]; then
      echo "Inflating (canonical PyAV + torch bicubic) $ARCHIVE_DIR -> $INFLATED_DIR"
      "$UV_BIN" run "${UV_WITH_AV_TORCH_NUMPY[@]}" python "$SELF_DIR/inflate.py" \
        "$ARCHIVE_DIR" "$INFLATED_DIR" "$VIDEO_NAMES_FILE"
      break  # Python script handles all videos in one call
    else
      echo "Inflating $in_path -> $out_path"
      "$FFMPEG_BIN" -y -i "$in_path" -vf "$(upscale_filter "$SOURCE_W" "$SOURCE_H" "$UPSCALE_FLAGS")" -an -sn -pix_fmt rgb24 -f rawvideo "$out_path"
    fi
  fi
done < "$VIDEO_NAMES_FILE"
