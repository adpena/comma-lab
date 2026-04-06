# latest report

## current state — 2026-04-06

The promoted honest Track B floor is now **`2.12`**.

### upstream snapshot

- Verified commit: `ec82c291ffeae5212e9a38253791d58995518a80`
- Pinned-file digests: all matched `workspace/upstream_snapshot.json`

## track a — exact_current

### current_workflow

- Status: alive
- Archive size: `167` bytes
- Final score: `0.00`
- Device: `cpu`

### rule_faithful

- Status: invalid for promotion
- Reason: inflate reconstructs from repo-side public videos and helper code outside `archive.zip`

## track b — robust_current

### promoted current_workflow floor

- Status: promoted
- Config: `524x394`, `libsvtav1 preset0`, `crf=34`, `film-grain=22`, `lanczos` upscale, `unsharp=9:9:0.35:9:9:0.0`, explicit `tv/bt709` encode tags, explicit `rgb24(pc)` decode
- Archive size: `864486`
- PoseNet distortion: `0.09384175`
- SegNet distortion: `0.00575313`
- Rate: `0.02302503`
- Final score: **`2.12`**
- Device: `cpu`
- Delta vs prior AV1 floor `2.18`: **`-0.06`**

### rule_faithful

- Status: local estimate available
- Honest bundle bytes: `897745`
- Estimated rate: `0.02391086183482655`
- Estimated score: **`2.1418040615200598`**

### rigor / bug pass

- packaging now honors the requested upstream root
- evaluation clears stale `inflated/` raws before scoring
- rule-faithful accounting now charges the installed runtime payload under test
- ROI branches are guarded to x265 instead of silently mixing AV1 and x265 assumptions
- ROI metadata analysis now honors `FFMPEG_BIN` and `FFPROBE_BIN`
- `ROI_X_FRAC` and ROI-side `INFLATE_POSTFILTER` now both work
- root `.gitignore` now covers cache and scratch artifacts
- git history cleanup was not possible here because this workspace is not a git repository
- explicit color hardening now closes the remaining flat-path colorspace/range-default risk

## latest winning probe

- Candidate: explicit `tv/bt709` encode tags + explicit `rgb24(pc)` decode
- Estimate before run: modest gain from lower evaluator mismatch
- Measured: **`2.12`** at `864,486` bytes
- Reflection: bytes barely moved, SegNet worsened slightly, but PoseNet improved sharply enough to produce a materially better total score.

## writeup / reproducibility status

- The site now includes explicit challenge context, repo identity (`adpena/comma-lab`), and a localized last-updated stamp.
- Rebuild path is now scripted via `just rebuild-site` / `python3 reports/graphs/refresh_site.py`.
- Durable generated artifacts now include:
  - `reports/graphs/experiment_manifest.json`
  - `reports/graphs/code_callouts.md`
  - `reports/graphs/media/*`
- Browser-preview comparison media is present for both full-frame and crop-zoom inspection.
- Desktop and mobile local screenshots were used to verify the current layout.
