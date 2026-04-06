# latest promotion review

## candidate

- `robust_current-av1-524x394-colorspace-hardening-promoted-cpu-2026-04-06`
- Prior floor: `2.18` (implicit color handling)
- Candidate: `2.12` (explicit bt709/tv encode tags + explicit rgb24(pc) decode)

## estimate before the run

- expected lower evaluator mismatch from explicit color handling
- expected score improvement: modest but real if PoseNet was color-conversion sensitive

## what happened

- bytes increased by only `31`
- pose improved by `-0.01272625`
- seg worsened by `+0.00005696`
- score improved by `-0.06` to `2.12`

## review verdict

- smoke gate: passed
- encoded ffprobe tags: passed
- formula check: passed
- decision: **PROMOTE**
