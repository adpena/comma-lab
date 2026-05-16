# L5 v2 TT5L Dykstra Formula And Constraint Hardening

Date: 2026-05-16
Author: Codex
Axis: TT5L / L5 v2 cargo-cult-unwind gate
Evidence grade: source-and-test hardening; no score claim

## Finding

The Dykstra planning helper documented and computed the feasible score
projection with a stale scalar formula:

`seg + sqrt(10 * pose) + rate`

The canonical contest scorer is:

`100 * seg + sqrt(10 * pose) + 25 * archive_bytes / 37,545,489`

The same artifact was also too easy for TT5L to satisfy with scalar
`FEASIBLE` fields only. It did not force the five Time-Traveler design moves
to be present as declared constraint axes, which preserved the cargo-cult risk
that the L5 v2 gate was supposed to remove.

## Change

`tools/check_substrate_dykstra_feasibility.py` now:

- uses the canonical `100 * seg` multiplier in the feasible upper bound;
- emits `score_formula`;
- emits `contest_seg_multiplier`;
- emits non-authority flags `score_claim=false`,
  `promotion_eligible=false`, and `ready_for_exact_eval_dispatch=false`;
- emits `polytope_projection_kind=score_axis_projection_with_declared_constraints`;
- accepts repeated `--constraint-set-id` values;
- adds `--tt5l-five-move-polytope` to append the five TT5L design-move
  constraint ids.

`src/tac/optimization/l5_staircase_v2.py` now refuses TT5L Dykstra artifacts
that are missing:

- the exact contest formula string;
- `contest_seg_multiplier=100.0`;
- the projection kind;
- all three contest-axis constraint ids;
- all five TT5L design-move constraint ids.

## Live Artifact Refresh

Refreshed ignored live state with:

```bash
.venv/bin/python tools/check_substrate_dykstra_feasibility.py \
  --substrate-id time_traveler_l5_5move \
  --predicted-band-lo 0.150 \
  --predicted-band-hi 0.170 \
  --archive-size-bytes 34603 \
  --tt5l-five-move-polytope \
  --output-json .omx/state/dykstra_feasibility_time_traveler_l5.json
```

Readiness check after refresh:

- `dykstra_valid=True`
- `sideinfo_valid=False`
- `timing_allowed=False`
- `next_action=materialize_tt5l_contest_full_frame_sideinfo_consumption_proof`

This is the desired fail-closed sequence: corrected Dykstra planning evidence
unblocks the side-info proof action but still does not unlock timing or
promotion.

## Verification

- `.venv/bin/python -m pytest src/tac/tests/test_check_substrate_dykstra_feasibility.py src/tac/tests/test_l5_staircase_v2.py -q`
  - `73 passed`
- `ruff check tools/check_substrate_dykstra_feasibility.py src/tac/optimization/l5_staircase_v2.py src/tac/tests/test_check_substrate_dykstra_feasibility.py src/tac/tests/test_l5_staircase_v2.py`
  - `All checks passed`

## No Score Claim

This artifact is planning-control evidence only. It is not a contest score, not
a full-frame side-info consumption proof, and not promotion evidence.
