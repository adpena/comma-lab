# final writeup draft

## opening

This repo follows two explicit tracks. Track A preserves the currently published workflow exploit for transparency and remains the only intentionally non-rule-faithful lane. Track B is the honest scorer-backed lane. The honest lane improved from `4.06` to a promoted **current_workflow** floor of **`2.12`** at **`864,486` bytes**. The corresponding local **rule_faithful estimate** is `2.142` at `897,745` bytes.

## why this writeup is stronger now

The work is not only a sequence of scores. It is a sequence of explanations. The search moved through three distinct modes:
1. honest x265 bitrate-placement and resolution improvements
2. a catastrophic AV1 evaluator-boundary bug (`97.45`)
3. a repaired AV1 path plus disciplined one-axis probes that pushed the honest floor from `2.20` to `2.19`, then `2.18`, and finally `2.12`

## latest experiment: explicit colorspace/range hardening

### prior baseline

- `2.18` at `864,455` bytes

### hypothesis

The prior AV1 path still relied on implicit ffmpeg color defaults. Making the encode/decode color contract explicit might reduce evaluator mismatch enough to improve score without materially changing rate.

### estimate before the run

Expected improvement: modest but real if PoseNet was more color-conversion-sensitive than SegNet.

### measured result

- candidate run: `2.12` at `864,486` bytes
- byte delta: `+31` (`+0.0036%`)
- pose delta: `-0.01272625`
- seg delta: `+0.00005696`

### reflection

The hypothesis held strongly. The bytes barely moved, SegNet worsened slightly, but PoseNet improved enough to dominate the total score. This is a hardening result with a measurable payoff.
