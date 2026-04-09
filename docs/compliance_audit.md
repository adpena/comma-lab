# compliance audit

## purpose

This document is an end-to-end compliance review of the repo against the currently published challenge rules and evaluation path.

## scope rule

- Default assumption: all lanes should be treated as rule-faithful and compliance-checked.
- The only explicitly non-rule-faithful lane is Track A / `submissions/exact_current`.

## repo compliance status

### track a — `exact_current`

- status: intentionally non-rule-faithful research / transparency lane
- reason: inflate depends on repo-side original public videos and helper code outside the archive
- implication: useful for documenting the current workflow behavior, not for honest promotion

### track b — `robust_current`

- status: compliant with the repo's honest / rule-faithful direction
- current promoted floor:
  - `522x392 / libsvtav1 / preset0 / crf34 / film-grain22 / lanczos / sharpness=1 / learned int8 post-filter`
  - `current_workflow`: `2.05`
  - local `rule_faithful` estimate: `2.0778631822069484`
- inflate path:
  - no GPU requirement
  - tiny shipped learned post-filter after upscale
  - explicit `rgb24` rawvideo output on the AV1 path
  - stale `inflated/` output is cleared before evaluation
- installed runtime payload under test:
  - `archive.zip`
  - `inflate.sh`
  - `inflate.py`
  - `inflate_postfilter.py`
  - `inflate_grain_mask.py`
  - `postfilter_int8.pt`
  - `config.env`
  - `analyze_roi.py`
- current rule-faithful payload bytes:
  - `896432`

## corrected rigor findings

- rule-faithful accounting now charges the installed runtime payload under test instead of the broader repo-local submission tree
- `robust_current` packaging now honors the requested challenge root
- evaluation now removes stale `inflated/` raws before scorer runs so smoke and eval share the same clean-run contract
- the smoke gate now checks all three:
  - raw file cardinality / presence
  - exact frame-count / geometry-derived byte size
  - sampled RGB semantic sanity on frames `0`, midpoint, and last
- package / smoke / eval now use a submission lock so concurrent runs cannot silently overlap on the same upstream submission directory
- robust_current shell paths now fail fast on incompatible ffmpeg toolchains instead of silently drifting across hosts
- AV1 + ROI is now explicitly rejected instead of silently drifting into x265-only logic
- ROI metadata analysis now honors `FFMPEG_BIN` and `FFPROBE_BIN`
- static ROI placement now honors `ROI_X_FRAC`
- ROI inflate now honors `INFLATE_POSTFILTER`
- packaging now survives a stale/invalid `TMPDIR` by falling back to `/tmp`

## promotion rule

Every future baseline promotion should come with:
1. scorer-backed result
2. pre-scorer smoke gate including sampled RGB semantic checks
3. canonical default-config regression
4. written promotion review
5. current_workflow vs rule_faithful separation
6. bug-audit notes


## 2026-04-07 promotion update

- Current Track B promoted floor: **2.05**
- Installed runtime payload now includes the tiny learned post-filter assets (`inflate_postfilter.py`, `postfilter_int8.pt`) in addition to the prior runtime payload entries.
- Current local rule-faithful payload bytes: `896432`
