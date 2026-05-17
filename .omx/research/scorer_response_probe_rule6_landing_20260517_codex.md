# Scorer-Response Probe Rule #6 Landing

Date: 2026-05-17
Lane: `lane_scorer_response_probe_rule6_20260517`
Author: codex

Authority:
- score_claim: false
- promotion_eligible: false
- ready_for_exact_eval_dispatch: false
- dispatch_attempted: false

## Why this exists

The 2026-05-17 substrate-design meta-assumption review found that byte
liveness and distinguishing-feature integration are necessary but not
sufficient. ATW, Wunderkind, Z6, NSCS06, and NSCS01 each exposed a different
way for a feature to exist without becoming a scorer-visible score movement.

This landing converts that review into a concrete mechanism:

- `src/tac/scorer_response_probe.py` compares baseline/candidate exact-eval
  evidence and separates scorer-term deltas from rate-only deltas.
- `tools/probe_substrate_score_response.py` emits JSON and Markdown reports
  for substrate ablations or candidate-vs-baseline scouting.
- `src/tac/tests/test_scorer_response_probe.py` pins the verdict taxonomy:
  scorer-positive, scorer-present-but-rate-negative, rate-only, no-response,
  regression, custody-blocked, and control-mismatch.

## Rule #6 operational form

Rule #6 is now executable:

> A substrate distinguishing feature is not promotion-relevant until a
> matched baseline/candidate comparison shows scorer-visible component
> movement under the same evidence axis and matched controls.

Default mode is `ablation`, which requires the same runtime tree and sample
count. `candidate` mode allows runtime-tree differences for scouting, but it
does not turn the result into a promotion claim.

## Canonical usage

```bash
.venv/bin/python tools/probe_substrate_score_response.py \
  --baseline-json experiments/results/<lane>/baseline_contest_auth_eval.json \
  --candidate-json experiments/results/<lane>/candidate_contest_auth_eval.json \
  --axis contest_cpu \
  --mode ablation \
  --output-json experiments/results/<lane>/score_response_probe.json \
  --output-md .omx/research/<lane>_score_response_probe_YYYYMMDD.md
```

The output is non-promotional by design. It can support a later dispatch or
promotion decision only when paired with the normal exact-eval custody,
dispatch-claim, archive/runtime identity, and axis-label rules.

## Verdict taxonomy

- `SCORER_RESPONSE_POSITIVE`: scorer terms improve and total score improves.
- `SCORER_RESPONSE_PRESENT_RATE_NEGATIVE`: scorer terms improve, but rate cost
  erases the total-score win. This is a byte-optimization target, not a method
  failure.
- `RATE_ONLY_IMPROVEMENT`: total improves from bytes/rate only; this does not
  prove the distinguishing feature is scorer-visible.
- `NO_MEASURABLE_RESPONSE`: deltas are below configured thresholds.
- `SCORE_REGRESSION`: total or scorer terms regress above threshold.
- `BLOCKED_CUSTODY`: exact-eval evidence is too weak or malformed.
- `BLOCKED_CONTROL_MISMATCH`: ablation controls do not match.

## Score-lowering consequence

This should be used before spending full-run budget on the high-risk
per-pair-conditioning cluster. It is especially relevant for:

- NSCS03: use it after the next minimal Ballé on/off ablation to confirm
  that the learned transform moves SegNet/PoseNet terms, not only bytes.
- Z6/Z7/Z8 and ATW-like lanes: require score response before treating
  conditioning liveness as frontier signal.
- PR101-style bolt-ons: separate rate-only wins from genuine scorer response
  so byte wins are celebrated for the correct reason.

## 2026-05-17 adversarial follow-up: axis missing fail-closed

The first post-landing bug hunt found a false-authority edge: if both evidence
rows omitted an axis and the caller did not pass `--axis`, the comparator could
classify a baseline/candidate delta under an empty axis. That violates the
axis-labelled evidence discipline. The comparator now fail-closes with
`BLOCKED_CUSTODY` on missing axis even when `expected_axis=None`, and normalizes
common bracketed labels such as `[contest-CPU]` and `[macOS-CPU advisory]`
without promoting advisory axes to contest axes.

## Verification

Focused verification run in this landing:

```bash
.venv/bin/python -m pytest src/tac/tests/test_scorer_response_probe.py
.venv/bin/python -m py_compile src/tac/scorer_response_probe.py tools/probe_substrate_score_response.py
git diff --check -- src/tac/scorer_response_probe.py tools/probe_substrate_score_response.py src/tac/tests/test_scorer_response_probe.py .omx/research/scorer_response_probe_rule6_landing_20260517_codex.md
```

No provider dispatch was attempted.
