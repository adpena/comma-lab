# findings

## 2026-04-06 colorspace-hardening AV1 promotion

### new authoritative floor

- Track B now has a new promoted honest floor: **`2.12`**
- Config: `524x394 / libsvtav1 / preset0 / crf34 / film-grain22 / lanczos / unsharp=0.35 / explicit bt709/tv encode tags / explicit rgb24(pc) decode`
- Current-workflow bytes: `864486`
- Rule-faithful estimate: `2.1418040615200598` at `897745` bytes

### hypothesis and result

- Hypothesis: explicit colorspace/range handling would reduce evaluator mismatch on the flat AV1 path.
- Result: hypothesis held.
- Byte delta vs prior floor: `+31` (`+0.0036%`)
- Score delta vs prior floor: `-0.0600`
- Pose delta: `-0.01272625`
- Seg delta: `+0.00005696`

### interpretation

- This is a production-hardening win, not just a tuning win.
- The score improved materially even though bytes barely changed, which means the evaluator cared about the explicit color conversion contract.
- PoseNet appears much more sensitive to this conversion path than SegNet at the current operating point.

## local frontier shape

The current AV1 frontier now has:
- a compression-side loss (`crf35`)
- a softer-reconstruction loss (`unsharp 0.30`)
- a synthesis-removal loss (`film-grain 0`)
- a geometry loss (`522x392`)
- an upscale-kernel win (`lanczos`)

That is excellent writeup material.

## 2026-04-06 comprehensive bug / rigor pass

### fixed execution-contract bugs

- `robust_current` packaging now honors the requested upstream root
- `--package` without sync is now rejected because it would package bytes different from the bytes under test
- evaluation now clears stale `inflated/` raws before scorer runs
- rule-faithful accounting now charges the installed runtime payload under test

### fixed ROI-path bugs

- AV1 + ROI now fails fast instead of silently drifting into x265-only logic
- ROI metadata analysis now honors both `FFMPEG_BIN` and `FFPROBE_BIN`
- `ROI_X_FRAC` now actually changes static ROI placement
- `INFLATE_POSTFILTER` now applies on ROI inflate paths
- source dimensions are now explicit config values rather than scattered literals

### verified evidence

- AV1 ROI guard returns a deliberate failure
- metadata ROI wrapper log shows both `ffprobe` and `ffmpeg`
- `ROI_X_FRAC=0.05` and `ROI_X_FRAC=0.45` produce different `roi.mkv` artifacts
- ROI inflate output changes under `INFLATE_POSTFILTER=hflip`
- encoded AV1 stream now probes as `tv / bt709 / bt709 / bt709`
- inflated raw path now probes as `rgb24(pc, gbr/bt709/bt709, progressive)`

### repo hygiene

- root `.gitignore` now covers caches and scratch artifacts
- transient `archive/` scratch is no longer left in `submissions/robust_current`
- git history cleanup was not possible because this workspace is not a git repository

## speculative next lane recorded

If AV1 + ROI is revisited, the required implementation plan is now explicitly recorded as:

1. codec-agnostic ROI encode abstraction
2. AV1 params for base/ROI/ROI2 streams
3. matching AV1-aware metadata ROI path
4. matching inflate/smoke/scorer parity checks
5. fresh scorer-backed evidence that it actually helps

This remains speculative until those steps are complete and measured.

## 2026-04-06 writeup system / frontend pass

### what changed

- Added a generated experiment manifest for durable reuse.
- Added generated code callouts tied to measured findings.
- Added reproducibility commands in `justfile` plus `docs/repro_checklist.md`.
- Added browser-preview comparison media with synced full-frame and crop-zoom playback controls.
- Added top-of-page contest context, repo identity, GitHub link, and localized last-updated metadata.
- Added poster images for the comparison videos and mobile-safe horizontal scrolling for the local-frontier table.

### interpretation

- The writeup is now easier to audit because the evidence surfaces and reproduction path are generated, not hand-maintained.
- The landing page is now closer to a technical brief than a generic dashboard.
- Remaining frontend work is refinement, not missing infrastructure.
