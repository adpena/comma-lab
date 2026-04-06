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
  - `524x394 / libsvtav1 / preset0 / crf34 / film-grain22 / lanczos / unsharp`
  - explicit color contract: encoded `tv / bt709 / bt709 / bt709`, decoded `rgb24(pc)`
  - `current_workflow`: `2.12`
  - local `rule_faithful` estimate: `2.1418040615200598`
- inflate path:
  - no heavy decoder-side model
  - stock ffmpeg decode / upscale path only
  - explicit `rgb24` rawvideo output on the AV1 path
  - explicit color contract on the flat path:
    - encoded archive stream tagged `tv / bt709 / bt709 / bt709`
    - inflator converts explicitly to `rgb24(pc)`
  - stale `inflated/` output is cleared before evaluation
- installed runtime payload under test:
  - `archive.zip`
  - `inflate.sh`
  - `config.env`
  - `analyze_roi.py`
- current rule-faithful payload bytes:
  - `897745`

## corrected rigor findings

- rule-faithful accounting now charges the installed runtime payload under test instead of the broader repo-local submission tree
- `robust_current` packaging now honors the requested challenge root
- evaluation now removes stale `inflated/` raws before scorer runs so smoke and eval share the same clean-run contract
- AV1 + ROI is now explicitly rejected instead of silently drifting into x265-only logic
- ROI metadata analysis now honors `FFMPEG_BIN` and `FFPROBE_BIN`
- static ROI placement now honors `ROI_X_FRAC`
- ROI inflate now honors `INFLATE_POSTFILTER`
- packaging now survives a stale/invalid `TMPDIR` by falling back to `/tmp`

## promotion rule

Every future baseline promotion should come with:
1. scorer-backed result
2. pre-scorer smoke gate
3. canonical default-config regression
4. written promotion review
5. current_workflow vs rule_faithful separation
6. bug-audit notes
