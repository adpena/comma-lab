# L5 v2 Dykstra Promotion-Adjacent False Authority Hardening

Date: 2026-05-17
Author: codex
Authority: implementation hardening; `score_claim=false`;
`promotion_eligible=false`; `ready_for_exact_eval_dispatch=false`;
`ready_for_provider_dispatch=false`; `dispatch_attempted=false`.

## Problem

The TT5L side-info Lightning execution bundle correctly stayed dry-run/default,
but its top-level readiness vocabulary stopped at dry-run/provider dispatch. A
bundle could therefore be `ready_for_dry_run_submit=true` while missing the
Dykstra feasibility artifact, and downstream readers had to infer that this did
not authorize side-info proof, paired-anchor readiness, timing-smoke authority,
rank/kill, or promotion-adjacent claims.

That ambiguity is a false-authority risk for the L5 v2 staircase: dry-run parse
readiness is useful launch hygiene, not scientific evidence that side-info
helps or that a paired CPU/CUDA anchor exists.

## Patch

`src/tac/optimization/l5_v2_tt5l_sideinfo_lightning_execution_bundle.py` now
emits explicit top-level promotion-adjacent readiness fields:

- `ready_for_sideinfo_effect_claim=false`
- `ready_for_timing_smoke_authority=false`
- `ready_for_paired_anchor_claim=false`
- `promotion_adjacent_readiness`
- `promotion_adjacent_blockers`

Missing or invalid
`.omx/state/dykstra_feasibility_time_traveler_l5.json` now appears as a
surface-specific blocker for all three promotion-adjacent surfaces while leaving
dry-run parsing separately available. When Dykstra is valid, those blockers
drop away, but the surfaces still remain false until harvested exact-eval cells,
passing side-info effect-curve evidence, paired CPU/CUDA runtime timing
artifacts, and adjudicated exact-eval artifacts exist.

The Markdown renderer now prints those fields so the operator-visible `.omx`
report carries the distinction directly.

## Tests

Focused verification:

```bash
.venv/bin/python -m pytest src/tac/tests/test_l5_v2_tt5l_sideinfo_lightning_execution_bundle.py -q
.venv/bin/python -m pytest src/tac/tests/test_l5_v2_tt5l_sideinfo_lightning_execution_bundle.py src/tac/tests/test_l5_v2_tt5l_sideinfo_lightning_execution_bundle_dry_run.py -q
.venv/bin/python -m py_compile src/tac/optimization/l5_v2_tt5l_sideinfo_lightning_execution_bundle.py src/tac/tests/test_l5_v2_tt5l_sideinfo_lightning_execution_bundle.py
.venv/bin/python tools/lane_maturity.py validate
git diff --check
```

Result:

- `5 passed`
- `13 passed`
- `py_compile` clean
- `794 lane(s) validated cleanly`
- `git diff --check` clean

## Evidence Discipline

This patch makes no score claim and does not dispatch provider work. It only
splits dry-run parse readiness from promotion-adjacent scientific authority so
the L5 v2 staircase cannot accidentally treat a launch bundle as side-info
effect evidence.
