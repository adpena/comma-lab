# Codex Findings: Local CPU Contest Drift Eureka Trigger

Date: 2026-05-22T19:49:25Z

## Verdict

Within the current DQS1/FEC6-like same-archive trust region, local macOS CPU is
reliably worse than Linux contest-CPU by a small SegNet-only offset. It is not
universal across older MPS, nonlocal-bias, or different-substrate artifacts.

The useful abstraction is therefore a trust-region calibration plus an eureka
trigger, not a global constant score correction and not score authority.

## Current Calibration

Artifact:
`.omx/research/local_cpu_contest_drift_calibration_dqs1_fec6_20260522T194800Z.json`

Paired same-archive anchors used:

- PR101 FEC6 archive `6bae0201fb08`: local minus contest `+0.000010`
- FEC3 compact archive `8866ebb655e9`: local minus contest `+0.000010`
- DQS1 top32 raw-u16 archive `3c4e15bfe7ae`: local minus contest `+0.000010`
- DQS1 diversity k002 archive `4432525de41c`: local minus contest `+0.000012`
- DQS1 drop rank020/pair0430 archive `088c17e2b6ac`: local minus contest
  `+0.000011`

All five anchors have identical rate and PoseNet between local and contest.
The offset is entirely SegNet: local SegNet distortion is higher by
`1.0e-7` to `1.2e-7`, which contributes `+0.000010` to `+0.000012` score.

Fitted stable-core model:

- Trust region: `dqs1_fec6_like_same_archive_segnet_rounding`
- Anchor count: `5`
- Median bias local-minus-contest: `+0.000010000000000010001`
- Mean bias local-minus-contest: `+0.000010600000000010602`
- Min/max: `+0.000010000000000010001` / `+0.000012000000000012001`
- Guard band: `0.000003`
- Confidence: `stable_core`

## Eureka Rule

Artifact:
`.omx/research/local_cpu_contest_drift_eureka_example_20260522T194800Z.json`

For this trust region only:

`conservative_projected_contest_score = local_score - median_bias + guard_band`

If that conservative projected score beats the current auth frontier, the row
should trigger exact auth anchoring spend. It still remains:

- `score_claim=false`
- `promotion_eligible=false`
- `rank_or_kill_eligible=false`
- `ready_for_exact_eval_dispatch=false`

This is an exact-eval spend trigger, not a score claim.

## Out-Of-Class Caveat

Older paired rows include very different local artifacts and substrate classes;
some show much larger offsets such as `+0.000136` and `+0.000288`. Those should
be treated as `wide_or_mixed` or out-of-trust-region until separately
calibrated by class, raw-output identity, scorer path, and component
decomposition.

## Code Landed

- `src/tac/optimization/local_cpu_contest_drift.py`
- `tools/calibrate_local_cpu_contest_drift.py`
- `src/tac/tests/test_local_cpu_contest_drift.py`

The helper dedupes same-archive paired anchors, fits the stable-core bias band,
and emits false-authority eureka signals for exact-eval spend triage.
