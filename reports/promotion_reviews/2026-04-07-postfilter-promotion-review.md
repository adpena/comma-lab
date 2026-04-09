# promotion review — 2026-04-07 learned post-filter

## candidate

- `robust_current-av1-522x392-postfilter-promoted-cpu-2026-04-07`
- Prior floor: `2.08` (`sharpness=1`)
- Candidate: `2.05` (`522x392 + sharpness=1 + tiny learned post-filter`)

## evidence

- local smoke:
  - `reports/raw/2026-04-07-postfilter/robust_current-exp-postfilter-smoke.json`
- authoritative scorer summary:
  - `reports/raw/2026-04-07-postfilter/robust_current-exp-postfilter-current_workflow-cpu-summary.json`
- authoritative scorer report:
  - `reports/raw/2026-04-07-postfilter/robust_current-exp-postfilter-current_workflow-cpu-report.txt`

## estimate before the run

- expected score improvement: real but uncertain
- reason: this was the first decode-side lane that improved the semantic smoke proxy materially while keeping bytes near the floor

## measured result

- current-workflow score: **`2.05`**
- current-workflow bytes: `861,986`
- rule-faithful estimate: `2.0778631822069484` at `896,432` bytes
- PoseNet: `0.07996829`
- SegNet: `0.00586716`

## comparison against the prior floor

- bytes decreased by `-2,182` vs the `2.08` sharpness=1 floor
- PoseNet improved materially
- SegNet worsened slightly but stayed within a winning range
- net result: the score improved enough to establish a new honest floor

## risks checked

- smoke gate: passed locally
- smoke cross-check: passed on BAT00
- scorer-backed local eval: passed
- honest payload ships the learned post-filter assets in the installed runtime payload
- grain-mask remained a non-competitive research lane at `2.30`
- the first post-filter variant was already rejected at `2.35` for training-distribution mismatch, so this promotion is not relying on that failed setup

## review verdict

- decision: **PROMOTE**
- rationale: this is the first scorer-backed decode-side learned lane that beat the prior `2.08` floor honestly
- next work: finish the senior-engineer and senior-editor review loop, then run the next capped three-experiment post-filter cycle
