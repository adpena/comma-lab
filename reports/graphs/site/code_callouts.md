# code callouts

Small, measured implementation details tied to the major score and rigor changes.

## Byte-layout fix

- file: `submissions/robust_current/inflate.sh`
- why it matters: The flat path forces rawvideo output to `rgb24`.

```bash
        "$FFMPEG_BIN" -y -i "$base_path" -i "$roi_path" -i "$roi2_path" \
          -filter_complex "[0:v]$(upscale_rgb_base_filter "$SOURCE_W" "$SOURCE_H" "$UPSCALE_FLAGS")[base];[1:v]$(upscale_rgb_base_filter "$roi_w" "$roi_h" "$UPSCALE_FLAGS")[roi1];[2:v]$(upscale_rgb_base_filter "$roi2_w" "$roi2_h" "$UPSCALE_FLAGS")[roi2];[base][roi1]overlay=${roi_x}:${roi_y}[tmp];[tmp][roi2]overlay=${roi2_x}:${roi2_y}$(if [ -n "$INFLATE_POSTFILTER" ]; then printf ',format=rgb24,%s' "$INFLATE_POSTFILTER"; else printf ',format=rgb24'; fi)[out]" \
```

## Explicit color contract

- file: `submissions/robust_current/compress.sh`
- why it matters: The encoded AV1 stream now carries explicit `tv/bt709` metadata.

```bash
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
      "$out_path"
```

## Rule-faithful payload accounting

- file: `src/comma_lab/install.py`
- why it matters: The honest payload under test is explicit and small.

```bash
INSTALL_PAYLOADS: dict[str, tuple[str, ...]] = {
    "exact_current": (
        "archive.zip",
        "inflate.sh",
        "inflate.py",
    ),
    "robust_current": (
        "archive.zip",
        "inflate.sh",
        "config.env",
        "analyze_roi.py",
    ),
```

## AV1 + ROI fail-fast guard

- file: `submissions/robust_current/compress.sh`
- why it matters: Unsupported AV1+ROI combinations fail loudly instead of silently drifting into x265-only behavior.

```bash
if [ "$ROI_ENABLE" = "1" ] && [ "$VIDEO_CODEC" != "libx265" ]; then
  echo "ERROR: ROI packaging currently supports VIDEO_CODEC=libx265 only." >&2
  echo "Disable ROI or switch VIDEO_CODEC to libx265 before packaging." >&2
  exit 1
```
