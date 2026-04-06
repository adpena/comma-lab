# final submission notes

## what to emphasize

- Track A is the only intentionally non-rule-faithful lane
- every honest promotion is scorer-backed and review-gated
- the writeup shows hypotheses, estimates, measurements, and reflections
- the AV1 byte-layout bug is a core mechanism story
- the latest AV1 win is a hardening win: explicit color handling reduced evaluator mismatch and improved score

## best concise summary

The honest lane started from `4.06`, reached an x265 floor at `3.25`, repaired a misleading AV1 evaluator bug to reach `2.20`, improved to `2.19` with a one-axis CRF move, improved again to `2.18` by switching the upscale kernel from bicubic to Lanczos at unchanged bytes, and then improved again to **`2.12`** by making the AV1 color contract explicit (`tv/bt709` encode tags plus explicit `rgb24(pc)` decode).
