# AV1 colorspace hardening promotion review

## candidate

- Run id: `robust_current-av1-524x394-colorspace-hardening-promoted-cpu-2026-04-06`
- Config: `524x394 / libsvtav1 / preset0 / crf34 / film-grain22 / lanczos / unsharp`
- Change from prior floor:
  - explicit encoded stream tags: `tv / bt709 / bt709 / bt709`
  - explicit decode conversion: `rgb24(pc)`
- Prior honest floor: `2.18` at `864,455` bytes

## hypothesis before the run

- The prior path still relied on implicit ffmpeg color defaults.
- Hypothesis: explicit bt709/tv tagging on encode and explicit rgb24(pc) decode would reduce evaluator mismatch.
- Expected upside: modest but real score improvement if PoseNet was more sensitive to color conversion than SegNet.

## measured result

- Candidate run: `2.12` at `864,486` bytes
- Byte delta vs prior floor: `+31` bytes (`+0.0036%`)
- Pose delta vs prior floor: `-0.01272625`
- Seg delta vs prior floor: `+0.00005696`
- Score delta vs prior floor: `-0.06`

## smoke gate

- Candidate passed pre-scorer smoke gate:
  - raw file cardinality
  - exact frame count
  - exact geometry-derived byte size
  - sampled semantic sanity
- Evidence: `reports/raw/2026-04-06-hardening/robust_current-hardening-smoke.json`

## metadata / path hardening evidence

- Encoded stream ffprobe:
  - `tv / bt709 / bt709 / bt709`
  - evidence: `reports/raw/2026-04-06-hardening/encoded-ffprobe.json`
- Inflated raw path now logs as:
  - `rgb24(pc, gbr/bt709/bt709, progressive)`

## official-formula check

Using the contest formula

`100 * segnet_distortion + 25 * rate + sqrt(10 * posenet_distortion)`

with:
- seg = `0.00575313`
- pose = `0.09384175`
- rate = `0.02302503`

gives `2.1243573373034`, which is consistent with the reported `2.12` after normal rounding.

## current_workflow vs rule_faithful

- current_workflow: `2.12` at `864,486` bytes
- corrected installed-runtime-payload estimate: `2.1418040615200598` at `897745` bytes
- charged payload:
  - `archive.zip`
  - `inflate.sh`
  - `config.env`
  - `analyze_roi.py`

## reflection

The hypothesis held strongly. Archive bytes barely changed, SegNet worsened slightly, but PoseNet improved enough to dominate the total score. This is exactly the kind of production-hardening win the lab should prize: cleaner execution semantics and a better measured result at the same operating point.

## decision

**PROMOTE**

Reason:
- scorer-backed candidate beat the prior floor
- smoke gate passed
- colorspace/range handling is now explicit instead of implicit
- package/eval/reporting contracts remain explicit and reproducible
