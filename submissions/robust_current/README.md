# robust_current submission

An honest patched-world submission lane built around stock ffmpeg codecs and a lightweight inflator.

## current promoted floor

- `524x394 / libsvtav1 / preset0 / crf34 / film-grain22 / lanczos / unsharp=0.35`
- explicit encoded stream tags: `tv / bt709 / bt709 / bt709`
- explicit decode conversion: `rgb24(pc)`
- `current_workflow`: `2.12`
- local `rule_faithful` estimate: `2.142`

## design

- standard codec path using ffmpeg + x265 or SVT-AV1
- no heavy decoder dependencies
- inflator uses ffmpeg to decode the archived stream and upscale to raw RGB frames
- rawvideo outputs are explicitly forced to `rgb24`
- the flat AV1 path now carries explicit color assumptions instead of implicit ffmpeg defaults:
  - encoded stream tagged `tv / bt709 / bt709 / bt709`
  - decoded raw RGB path explicitly converts to `rgb24(pc)`
- installed runtime payload is intentionally minimal:
  - `archive.zip`
  - `inflate.sh`
  - `config.env`
  - `analyze_roi.py`
- every future promotion should come with a written promotion review and bug audit

## bug / rigor guardrails

- local packaging now cleans transient `archive/` scratch instead of leaving it in the submission tree
- local packaging now falls back to `/tmp` if `TMPDIR` is invalid instead of failing
- ROI branches are currently guarded to `libx265` only; AV1 + ROI is not silently allowed
- ROI metadata analysis now honors both `FFMPEG_BIN` and `FFPROBE_BIN`
- `ROI_X_FRAC` now actually affects the static ROI branch
- `INFLATE_POSTFILTER` now applies on ROI inflate paths too
- evaluation clears stale `inflated/` output before scorer runs so smoke/eval share the same clean-run assumption

## pre-scorer smoke gate

Run before relying on a candidate:

```bash
python3 -m src.comma_lab.cli smoke-submission robust_current --package
```

This checks raw output existence, file cardinality, exact frame count, exact geometry-derived byte size, and sampled semantic sanity before a full scorer run.

## preserved comparison configs

- `config.env` — live promoted floor
- `config.av1-2.12.env` — named current AV1 snapshot
- `config.av1-2.18.env` — previous AV1 floor for comparison
- `config.av1-2.19.env` — previous AV1 floor for comparison
- `config.x265-3.25.env` — preserved x265 floor
