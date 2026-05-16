# L5 v2 TT5L artifact refresh and readiness guard

Date: 2026-05-16

Scope: TT5L / L5 v2 staircase, operator briefing, measurement schedule

## Summary

Follow-up read-only adversarial review found three live L5 v2 control-plane
risks:

1. The TT5L Dykstra artifact in `.omx/state/` was stale relative to the current
   provenance validator.
2. The top-level L5 `ready_for_gate_probe_dispatch` flag could become true from
   the four gate artifacts while TT5L cargo-cult unwind preconditions were
   still false.
3. The durable L5 v2 measurement schedule still contained a CUDA-only side-info
   effect-curve row.

This landing fixes the executable/readiness surfaces and refreshes the durable
schedule artifact. The Dykstra state artifact was re-emitted locally and its
hash is recorded here rather than committed as raw `.omx/state` churn.

## Changes

- Added `tools/build_tt5l_first_anchor_timing_smoke_artifact.py` so the TT5L
  timing-smoke custody artifact is materializable from a measured result file
  with provider, hardware, call id, exact command argv, elapsed time, rate
  metric, and result-artifact SHA-256.
- Exposed `tt5l_first_anchor_timing_smoke_status()` from
  `tac.optimization.l5_staircase_v2` for tools/tests.
- Suppressed top-level `ready_for_gate_probe_dispatch`,
  `ready_for_score_or_rank_dispatch`, and `ready_for_dispatch` unless TT5L
  Dykstra and move-level feasibility preconditions are valid.
- Added an operator-briefing preflight guard against a top-level L5 gate-probe
  ready flag when TT5L cargo-cult preconditions are false.
- Regenerated `.omx/research/l5_v2_lattice_measurement_schedule_20260516_codex.{json,md}`;
  `measure_tt5l_sideinfo_effect_curve` now requires both `contest_cpu` and
  `contest_cuda`.

## Refreshed Dykstra artifact

Command:

```bash
.venv/bin/python tools/check_substrate_dykstra_feasibility.py \
  --substrate-id time_traveler_l5_5move \
  --predicted-band-lo 0.150 \
  --predicted-band-hi 0.170 \
  --archive-size-bytes 34603 \
  --tt5l-five-move-polytope \
  --output-json .omx/state/dykstra_feasibility_time_traveler_l5.json
```

Result:

- verdict: `FEASIBLE`
- artifact_sha256: `226c227c1c08b25ea7208c6ee774f7621b25c25929870c28535a1f8896504b60`
- score_claim: `false`
- promotion_eligible: `false`
- ready_for_exact_eval_dispatch: `false`

Readiness after refresh:

- `dykstra_feasibility_artifact_valid=true`
- `move_level_feasibility_artifact_valid=false`
- `ready_for_gate_probe_dispatch=false`
- next TT5L action: `materialize_tt5l_move_level_feasibility_proof`

## Refreshed schedule artifacts

- JSON SHA-256: `d3e3b123e45aee684c1e6facd6c24ffdeea4d9026e805f13c55665c4808b47af`
- Markdown SHA-256: `a80a2b4abee78d0b6461f9faa24b5128ef290eb78338c86dbc747178f03c8179`
- side-info effect curve required axes: `["contest_cpu", "contest_cuda"]`

## Verification

- `.venv/bin/python -m ruff check src/tac/optimization/l5_staircase_v2.py tools/build_tt5l_first_anchor_timing_smoke_artifact.py tools/all_lanes_preflight.py src/tac/tests/test_l5_staircase_v2.py src/tac/tests/test_build_tt5l_first_anchor_timing_smoke_artifact.py src/tac/tests/test_all_lanes_operator_briefing_gate.py`
- `.venv/bin/python -m pytest src/tac/tests/test_l5_staircase_v2.py src/tac/tests/test_build_tt5l_first_anchor_timing_smoke_artifact.py src/tac/tests/test_all_lanes_operator_briefing_gate.py -q`

This is not score evidence and does not authorize dispatch. It prevents stale
state, stale schedules, and top-level dashboard booleans from outrunning the
TT5L staircase's concrete custody requirements.
