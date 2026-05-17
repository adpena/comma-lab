# Scorer-Response Probe Byte-Matched Ablation Hardening

Date: 2026-05-17
Lane: `lane_scorer_response_probe_rule6_20260517`
Author: codex

Authority:
- score_claim: false
- promotion_eligible: false
- ready_for_exact_eval_dispatch: false
- dispatch_attempted: false

## Finding

The Rule #6 score-response comparator correctly fail-closed on missing axis,
runtime mismatch, sample-count mismatch, and malformed exact-eval custody, but
one ablation-control gap remained: default `mode="ablation"` did not require
the baseline and candidate archive byte counts to match.

That could let a higher-capacity candidate archive be compared as an
"ablation" and receive `SCORER_RESPONSE_POSITIVE` from extra payload capacity
rather than from the declared distinguishing feature. This is exactly the
meta-assumption failure the 2026-05-17 adversarial review warned about:
feature liveness or feature capacity can masquerade as scorer-visible response
when the control is not byte-matched.

## Patch

`src/tac/scorer_response_probe.py` now makes byte matching part of the
ablation control surface:

- `compare_score_response(..., mode="ablation")` defaults to
  `max_ablation_archive_bytes_delta=0`.
- If archive byte counts differ beyond that tolerance, the report returns
  `BLOCKED_CONTROL_MISMATCH` with an `archive_bytes_mismatch:<delta>><limit>`
  blocker.
- Byte-changing scouting comparisons remain supported through
  `mode="candidate"`, where the tool still separates scorer-term movement from
  rate-only movement.
- The threshold is serialized into the report under
  `thresholds.max_ablation_archive_bytes_delta`.
- `tools/probe_substrate_score_response.py` exposes
  `--max-ablation-archive-bytes-delta` for rare deliberate ablations with a
  known fixed byte slack.

## Why This Matters For Score Lowering

Rule #6 is only useful if it tells us whether a distinguishing feature moves
SegNet/PoseNet terms under matched controls. Without byte matching, the probe
could bless a payload-size change as a feature-response result. That would push
the operator back into the same local-minimum loop: dispatching lifted-trainer
form variants because a probe measured capacity rather than mechanism.

This patch tightens the split:

- **Ablation mode**: same axis, same runtime tree, same sample count, and now
  same archive byte count by default.
- **Candidate mode**: byte/runtime changes are allowed for scouting, but the
  verdict remains non-promotional and explicitly distinguishes
  `RATE_ONLY_IMPROVEMENT` from `SCORER_RESPONSE_POSITIVE`.

## Regression Coverage

Added focused regressions in `src/tac/tests/test_scorer_response_probe.py`:

- default ablation rejects archive byte mismatch;
- explicit byte tolerance permits a bounded ablation;
- report JSON carries the byte-tolerance threshold;
- byte-overpay and rate-only examples use `candidate` mode instead of silently
  relying on ablation semantics.

## Verification

Commands run:

```bash
.venv/bin/python -m pytest src/tac/tests/test_scorer_response_probe.py -q
.venv/bin/python -m py_compile src/tac/scorer_response_probe.py tools/probe_substrate_score_response.py src/tac/tests/test_scorer_response_probe.py
```

Results:

- `19 passed`
- `py_compile` clean

No provider dispatch, no archive build, and no score claim were made.
