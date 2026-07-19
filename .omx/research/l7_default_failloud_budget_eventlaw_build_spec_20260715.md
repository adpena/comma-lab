# L7 fail-loud default + event-conditional wall-clock budget law — build spec

`research_only=false` · delegated lane `l7_default_failloud_budget_eventlaw` ·
MAIN landing review required · no launch/score/pointer authority.

## Objective

1. Make the canonical level-set trainer's inherited l7 stage disabled unless the
   caller explicitly opts in. Preserve explicit historical values byte-for-byte
   and emit a conspicuous structured warning whenever an explicit positive
   `--l7-start-epoch` can fire within the run.
2. Replace the flat pre-event budget derivation with a piecewise event-stage law:
   total minutes are the sum of `expected_epochs_in_stage * stage_min_per_ep`.
   Stage anchors must carry executable `LawRef` custody. Missing/unusable stage
   telemetry falls back to the old flat anchor and the launcher must log why.

## Allowed surfaces

- `experiments/train_levelset_witness_realized_through_R_mlx.py`
- `src/tac/local_acceleration/scorer_throughput_gate.py`
- `tools/launch_witness_run.py`
- focused tests under `src/tac/tests/`
- one small canonical-equation/evaluator surface and one machine-readable anchor
  receipt only if required for genuine `LawRef` custody

Do not touch experiment result/run directories, witness-DSL hot files, score
pointers, provider dispatch, or unrelated formatting.

## Required behavior

### L7

- Argparse default is a disabled sentinel, not epoch 800.
- Resolve the disabled sentinel after parsing to the existing `epochs + 1`
  never-runs representation so downstream schedule/resume logic remains additive
  and legacy-compatible.
- Explicit positive values remain unchanged. If `0 < l7_start_epoch <= epochs`,
  emit one JSON log row naming l7 a measured defect and declaring explicit opt-in.
- Explicit disabled/parked values do not warn.
- Add focused pure/parser tests for default disabled, explicit 800 preserved and
  warned, and explicit parked values not warned.

### Wall-clock law

- Introduce a typed stage-anchor/profile representation and a derivation receipt
  exposing: total days, per-stage epoch counts/min-per-ep/source, LawRef manifest,
  whether flat fallback was used, and the fallback reason.
- Default canonical profile uses the measured C0 telemetry in
  `.omx/research/v9_missing_signal_constants_audit_20260715.md`: lane-band fires at
  epoch 33; pre-event median 251.6 s/ep; post-event observed 325–333 s/ep. If the
  range requires one scalar, derive it transparently (for example midpoint), do
  not silently pick a bound.
- Compute `sum(stage_epochs * stage_min_per_ep) / 1440 * slack`; truncate stage
  counts correctly for short runs.
- Keep the existing flat formula as an explicit fallback for absent/invalid
  telemetry, retaining backward-compatible `derive_wall_clock_budget_days(...) ->
  float` for callers.
- The launcher fallback path consumes the new derivation receipt and includes
  `event-conditional` or `flat fallback: <reason>` in its logged budget source.
- Existing operator override/config-declared priority remains unchanged.

## Acceptance

```bash
.venv/bin/ruff check \
  experiments/train_levelset_witness_realized_through_R_mlx.py \
  src/tac/local_acceleration/scorer_throughput_gate.py \
  tools/launch_witness_run.py \
  src/tac/tests/test_scorer_throughput_gate.py \
  src/tac/tests/test_wallclock_default_on_perfenv_guard.py

.venv/bin/pytest -q \
  src/tac/tests/test_scorer_throughput_gate.py \
  src/tac/tests/test_wallclock_default_on_perfenv_guard.py \
  src/tac/tests/test_seg_form_unify_tau.py \
  src/tac/tests/test_curriculum_epoch_budget_guard_20260713.py \
  src/tac/tests/test_launch_witness_run.py
```

Also re-run a repo grep over checked-in `launch.sh` files and report any
curriculum launch that has neither explicit l7 parking nor
`--seg-form-unify-tau`.

## Landing protocol

The implementation executor must leave changes uncommitted for parent review.
The parent performs real three-way diff review, marks every changed `.py` through
`tools/review_tracker.py`, computes post-edit hashes, and commits only through
`tools/subagent_commit_serializer.py`. Final verdict remains pending MAIN review.
