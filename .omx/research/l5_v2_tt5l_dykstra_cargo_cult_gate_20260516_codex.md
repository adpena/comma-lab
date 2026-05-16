# L5 v2 TT5L Dykstra Cargo-Cult Gate - 2026-05-16

## Scope

Operator asked to review the cargo-cult assumption documents as another example
of the local-minimum failure. This pass focused on the L5/L5 v2 staircase and
the Time-Traveler L5 v2 path, per the standing operator priority.

Reviewed documents:

- `.omx/research/time_traveler_l5_cargo_cult_unwind_design_20260516.md`
- `.omx/research/high_risk_substrate_cargo_cult_unwind_audit_20260516.md`
- `.omx/research/meta_assumption_backfill_audit_all_staircase_substrates_20260516.md`
- `.omx/research/c6_e4_mdl_ibps_cargo_cult_unwind_design_20260516.md`
- `.omx/research/nscs02_downsampled_renderer_cargo_cult_unwind_design_20260516.md`
- `.omx/research/atw_codec_v1_cargo_cult_unwind_design_20260516.md`
- `.omx/research/sane_hnerv_cargo_cult_unwind_design_20260516.md`
- Claude memory addenda on hard-earned-vs-cargo-culted assumption
  classification and the assumptions-challenge audit.

## Finding

The TT5L cargo-cult is not any one paper or ingredient. The failure mode is
treating five individually plausible moves as additive score movement:

```text
old: ΔS_TT5L = ΔS_pc + ΔS_AR + ΔS_fov + ΔS_wm + ΔS_tik
new: ΔS_TT5L = projection(⋂ per-move polytopes ∩ rate/distortion budget)
```

The reviewed TT5L design memo already retired the old `[0.150, 0.170]`
additive band and set the score band to null pending Dykstra feasibility. The
remaining control-plane gap was that `l5_v2_dispatch_readiness()` still made the
full-frame side-info proof the first TT5L action. That could preserve the old
local-minimum behavior: useful engineering work would proceed before the
retired additive claim was forced through the feasibility intersection.

## Landed Change

`src/tac/optimization/l5_staircase_v2.py` now requires a planning-only TT5L
Dykstra feasibility artifact before side-info proof, timing smoke, paired anchor
planning, or stack-of-stacks actions are surfaced as the next TT5L step.

Required artifact:

```text
.omx/state/dykstra_feasibility_time_traveler_l5.json
```

Required tool:

```text
tools/check_substrate_dykstra_feasibility.py
```

The readiness payload records:

- `dykstra_feasibility_artifact_valid`
- `dykstra_feasibility_status`
- `first_anchor_timing_smoke_allowed = dykstra_valid and sideinfo_valid`

The parser fails closed on malformed planning authority. In particular, an
empty `{}` Dykstra artifact is rejected with
`tt5l_dykstra_feasibility_artifact_empty`; required fields must be present
before the next TT5L action can advance to the side-info proof.

The Dykstra artifact is explicitly not score authority:

```text
score_claim=false
promotion_eligible=false
ready_for_exact_eval_dispatch=false
```

## Next TT5L Action

Run the planning-control artifact before the full-frame side-info proof:

```text
.venv/bin/python tools/check_substrate_dykstra_feasibility.py \
  --substrate-id time_traveler_l5_5move \
  --predicted-band-lo 0.150 \
  --predicted-band-hi 0.170 \
  --archive-size-bytes <tt5l_target_or_candidate_archive_bytes> \
  --output-json .omx/state/dykstra_feasibility_time_traveler_l5.json
```

This command is intentionally a cargo-cult-unwind gate, not a promotion gate.
It prevents the retired additive five-move score band from quietly re-entering
operator briefing, cathedral autopilot, or dispatch planning.

## Verification

```text
.venv/bin/ruff check src/tac/optimization/l5_staircase_v2.py src/tac/tests/test_l5_staircase_v2.py
# All checks passed

PYTHONPATH=src .venv/bin/python -m pytest \
  src/tac/tests/test_l5_staircase_v2.py::test_l5_v2_dispatch_readiness_prioritizes_tt5l_campaign_action \
  src/tac/tests/test_l5_staircase_v2.py::test_l5_v2_tt5l_dykstra_artifact_unblocks_sideinfo_next_action \
  src/tac/tests/test_l5_staircase_v2.py::test_l5_v2_valid_gates_do_not_unlock_tt5l_timing_without_dykstra \
  src/tac/tests/test_l5_staircase_v2.py::test_l5_v2_tt5l_timing_requires_dykstra_and_sideinfo_evidence -q
# 4 passed

PYTHONPATH=src .venv/bin/python -m pytest \
  src/tac/tests/test_l5_staircase_v2.py \
  src/tac/tests/test_operator_briefing.py::test_briefing_runs_all_three_phases \
  src/tac/tests/test_operator_briefing.py::test_l5_v2_briefing_suppresses_packetir_targets_on_matrix_sha_mismatch \
  src/tac/tests/test_cathedral_autopilot.py::test_validation_queue_surfaces_l5_v2_packetir_stack_state -q
# 65 passed

PYTHONPATH=src .venv/bin/python tools/operator_briefing.py \
  --json --skip-pareto --skip-dashboard --skip-reconciler
# primary_staircase=tt5l_first_non_pr106_l5_v2
# next_non_pr106_l5_action=run_tt5l_dykstra_feasibility_polytope
# dykstra_feasibility_artifact_valid=false
# first_anchor_timing_smoke_allowed=false
# score_claim=false
# ready_for_exact_eval_dispatch=false
```
