# L5 v2 TT5L no-op custody gate hardening

Recorded: 2026-05-17T06:01:30Z

Classification: engineering hardening, no score claim.

## Trigger

The Volta read-only review found a real rigor gap: tracked TT5L custody
artifacts could lag the current no-op hardening, and the TT5L side-info
effect-curve dispatch planner did not require the newer source-change custody
fields before marking work units operator-ready.

## Changes

- Hardened `src/tac/optimization/l5_v2_tt5l_sideinfo_effect_curve_dispatch_plan.py`
  so variant rows fail closed if they omit:
  `generation_rule`, `variant_seed`, source archive/member/side-section SHA-256
  fields, or source-change booleans.
- Added a regression test in
  `src/tac/tests/test_l5_v2_tt5l_sideinfo_effect_curve_dispatch_plan.py` that
  removes side-section custody from `random_lsb` and expects
  `ready_for_operator_dispatch=false`.
- Regenerated `.omx/research/l5_v2_tt5l_sideinfo_variant_packets_20260517_codex.json`
  and its markdown report with the current builder, preserving the historical
  all-zero source classification while adding no-op/source-change custody fields.
- Rebuilt `.omx/research/l5_v2_tt5l_sideinfo_effect_curve_dispatch_plan_20260517_codex.json`
  under the stricter validator. The plan remains no-score/no-dispatch; all five
  work units are still only operator-ready, not provider-dispatch-ready.

## Artifact check

- Legacy all-zero manifest missing required custody fields: none.
- Current full-shape manifest missing required custody fields: none.
- Current effect-curve dispatch plan embedded variant rows missing required
  custody fields: none.
- Current effect-curve ready work units: `5`.
- Current blockers remain:
  `requires_paired_cpu_cuda_exact_eval_for_sideinfo_effect_curve`,
  `requires_dispatch_lane_claim_before_auth_eval`,
  `score_claim_forbidden_until_effect_curve_artifact_passes`,
  `requires_paired_cpu_cuda_exact_eval_before_score_claim`.

## Validation

- `.venv/bin/python -m pytest -q src/tac/tests/test_l5_v2_tt5l_sideinfo_effect_curve_dispatch_plan.py src/tac/tests/test_tt5l_sideinfo_variant_packets.py`
  -> `11 passed`
- `.venv/bin/python -m pytest -q src/tac/tests/test_l5_v2_measurement_schedule.py src/tac/tests/test_l5_v2_sideinfo_effect_curve.py`
  -> `24 passed`
- `.venv/bin/python -m ruff check src/tac/optimization/l5_v2_tt5l_sideinfo_effect_curve_dispatch_plan.py src/tac/tests/test_l5_v2_tt5l_sideinfo_effect_curve_dispatch_plan.py src/tac/optimization/tt5l_sideinfo_variant_packets.py src/tac/tests/test_tt5l_sideinfo_variant_packets.py`
  -> passed
- `.venv/bin/python -m py_compile src/tac/optimization/l5_v2_tt5l_sideinfo_effect_curve_dispatch_plan.py src/tac/tests/test_l5_v2_tt5l_sideinfo_effect_curve_dispatch_plan.py`
  -> passed

## Remaining blockers

- No full 600-pair trained TT5L archive with proven useful nonzero side-info has
  been exact-evaluated.
- TT5L side-info effect claims still require paired `[contest-CPU]` and
  `[contest-CUDA]` cells for all five current archive SHA-256 values.
- Modal remains blocked by workspace billing limits; Lightning current plan is
  CUDA dry-run only and needs identity/teamspace/source-staging/remote-CUDA
  preflights before non-dry-run.
