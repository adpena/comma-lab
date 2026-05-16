# L5 v2 Measurement Schedule Side-Info Curve Transition

Date: 2026-05-16
Author: Codex
Scope: L5 v2 staircase measurement lattice

## Verdict

`score_claim=false`; `promotion_eligible=false`; `ready_for_exact_eval_dispatch=false`.

The L5 v2 measurement schedule no longer has a dead terminal rule for
`prepare_paired_anchor_packet`. It can now advance from the side-info effect
curve step to paired-anchor packet preparation, but only when an explicit
paired CPU/CUDA side-info effect-curve summary satisfies a fail-closed
contract.

## Failure Class

`l5_v2_measurement_schedule_terminal_rule_unreachable`

Before this change:

- missing or incomplete probe intake routed correctly to C1/Z5/TT5L probe
  filling;
- eligible probe intake routed to `measure_tt5l_sideinfo_effect_curve`;
- `prepare_paired_anchor_packet` had `matches=false` hard-coded, so the
  first-match lattice could not advance after the effect curve existed.

That could keep L5 v2 stuck in a local planning loop even after the intended
paired side-info evidence was available.

## Landed Fix

Added a side-info effect-curve summary contract in
`src/tac/optimization/l5_v2_measurement_schedule.py`:

- schema: `l5_v2_sideinfo_effect_curve_v1`
- required axes: `contest_cpu`, `contest_cuda`
- required variants: `zero`, `random_lsb`, `shuffled`, `trained`, `ablated`
- no score/promotion/dispatch authority flags at top level or cell level
- no per-cell blockers
- complete axis x variant observed-cell grid
- `predicate_passed=true`

`tools/build_l5_v2_lattice_measurement_schedule.py` now accepts
`--sideinfo-effect-curve-json` and passes the artifact into the schedule
builder. Missing or unpaired artifacts keep the active rule at
`measure_tt5l_sideinfo_effect_curve`; complete paired artifacts advance the
active rule to `prepare_paired_anchor_packet`.

## Verification

```bash
.venv/bin/python -m ruff check \
  src/tac/optimization/l5_v2_measurement_schedule.py \
  tools/build_l5_v2_lattice_measurement_schedule.py \
  src/tac/tests/test_l5_v2_measurement_schedule.py

.venv/bin/python -m pytest \
  src/tac/tests/test_l5_v2_measurement_schedule.py \
  src/tac/tests/test_l5_staircase_v2.py \
  src/tac/tests/test_all_lanes_operator_briefing_gate.py -q
```

Observed:

- `ruff`: all checks passed
- `pytest`: `117 passed in 0.74s`

## Next Work

The effect-curve producer still needs to emit the paired summary contract from
actual CPU/CUDA measurements. Until then, this is a readiness transition and
not empirical score evidence.

## Addendum: Architecture-Lock Wiring

Follow-up hardening in the same L5 v2 contract makes the side-info effect curve
an explicit Time-Traveler L5 architecture-lock blocker, not only a measurement
schedule transition:

- `tac.optimization.l5_staircase_v2.tt5l_sideinfo_effect_curve_status(...)`
  validates the public effect-curve artifact path.
- `l5_v2_tt5l_campaign_readiness` now sets
  `sideinfo_effect_curve_artifact_valid`,
  `sideinfo_effect_curve_status`, and `architecture_lock_allowed`.
- the next non-PR106 L5 action is
  `measure_tt5l_sideinfo_effect_curve` until the paired CPU/CUDA artifact is
  present and valid.
- `tools/operator_briefing.py` now surfaces both
  `TT5L side-info curve artifact` and
  `TT5L architecture lock allowed`, so the operator view cannot skip this
  blocker.

Additional verification:

```bash
.venv/bin/python -m ruff check \
  src/tac/optimization/l5_staircase_v2.py \
  src/tac/optimization/l5_v2_measurement_schedule.py \
  src/tac/tests/test_l5_staircase_v2.py \
  src/tac/tests/test_l5_v2_measurement_schedule.py \
  tools/operator_briefing.py

.venv/bin/python -m pytest \
  src/tac/tests/test_l5_v2_measurement_schedule.py \
  src/tac/tests/test_l5_staircase_v2.py \
  src/tac/tests/test_remote_lane_time_traveler_l5_script.py -q

.venv/bin/python -m pytest \
  src/tac/tests/test_operator_briefing.py \
  src/tac/tests/test_all_lanes_operator_briefing_gate.py -q
```

Observed:

- L5 v2 local slice: `104 passed in 0.88s`
- operator briefing slice: `43 passed in 68.40s`
- `lane_maturity.py validate`: `767 lane(s) validated cleanly`
