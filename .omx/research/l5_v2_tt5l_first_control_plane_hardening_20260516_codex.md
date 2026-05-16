# L5 v2 TT5L-First Control Plane Hardening - 2026-05-16

## Scope

Operator correction: L5/L5 v2 staircase is the non-negotiable priority. This
landing removes PR106 PacketIR as the default face of the L5 v2 control plane
and promotes TT5L-first actions as the primary visible staircase path.

## Landed changes

- Added a contest-full-frame TT5L side-info consumption proof builder:
  `tools/build_tt5l_contest_sideinfo_consumption_proof.py`.
- Added `build_tt5l_contest_full_frame_sideinfo_consumption_proof(...)` so the
  L5 v2 side-info gate can accept contest-shaped 600-pair / 1200-frame
  mutation evidence without a score claim.
- Added `l5_v2_tt5l_campaign_readiness(...)` and wired it into
  `l5_v2_dispatch_readiness(...)`.
- Updated `tools/operator_briefing.py` Phase 9 to show
  `L5-v2 TT5L-first frontier readiness` and expose
  `next_non_pr106_l5_action`.
- Updated `tools/cathedral_autopilot.py` to queue a non-PR106
  `time_traveler_l5_v2_tt5l_first_anchor` row before optional PacketIR stack
  rows.
- Made `contest_exact_eval` a legal substrate target mode and aligned the TT5L
  registered contract with its dispatch recipe.
- Corrected substrate inventory visibility: the toy TT5L side-info proof is no
  longer treated as `sideinfo_consumed=true`; the full-frame proof remains the
  gate.

## Evidence discipline

All new surfaces are planning/custody surfaces:

- `score_claim=false`
- `promotion_eligible=false`
- `rank_or_kill_eligible=false`
- `ready_for_exact_eval_dispatch=false`

No score movement is claimed from this landing. The next TT5L action is to
materialize the contest-full-frame side-info proof from real baseline/mutated
TT5L inflated raw outputs, then use the existing A100 recipe for paired
CPU/CUDA first-anchor work after lane claim.

## Verification

```text
.venv/bin/ruff check src/tac/substrate_registry/contract.py src/tac/tests/test_substrate_registry.py src/tac/optimization/l5_staircase_v2.py tools/operator_briefing.py tools/cathedral_autopilot.py src/tac/optimization/substrate_composition_matrix.py src/tac/substrates/time_traveler_l5_autonomy/registered_substrate.py src/tac/tests/test_l5_staircase_v2.py src/tac/tests/test_operator_briefing.py src/tac/tests/test_cathedral_autopilot.py src/tac/substrates/time_traveler_l5_autonomy/tests/test_registered_substrate.py src/tac/tests/test_autopilot_dispatch_ranking.py tools/build_tt5l_contest_sideinfo_consumption_proof.py src/tac/substrates/time_traveler_l5_autonomy/consumption_proof.py src/tac/substrates/time_traveler_l5_autonomy/tests/test_consumption_proof.py
# All checks passed

PYTHONPATH=src .venv/bin/python -m pytest src/tac/substrates/time_traveler_l5_autonomy/tests/test_consumption_proof.py src/tac/tests/test_l5_staircase_v2.py::test_l5_v2_dispatch_readiness_prioritizes_tt5l_campaign_action src/tac/tests/test_l5_staircase_v2.py::test_l5_v2_sideinfo_consumption_rejects_toy_manifest_scope src/tac/tests/test_l5_staircase_v2.py::test_l5_v2_sideinfo_consumption_binds_mutation_to_parsed_section_range src/tac/tests/test_operator_briefing.py::test_briefing_runs_all_three_phases src/tac/tests/test_operator_briefing.py::test_l5_v2_briefing_suppresses_packetir_targets_on_matrix_sha_mismatch src/tac/tests/test_cathedral_autopilot.py::test_validation_queue_surfaces_l5_v2_packetir_stack_state src/tac/substrates/time_traveler_l5_autonomy/tests/test_registered_substrate.py src/tac/tests/test_substrate_registry.py::test_contest_exact_eval_is_legal_target_mode src/tac/tests/test_autopilot_dispatch_ranking.py::test_uncustodied_prediction_bands_do_not_receive_autopilot_rank_reward src/tac/tests/test_autopilot_dispatch_ranking.py::test_time_traveler_l5_source_scope_flows_to_autopilot_candidate_row -q
# 20 passed

tools/operator_briefing.py --json --skip-pareto --skip-dashboard --skip-reconciler
# primary_staircase=tt5l_first_non_pr106_l5_v2
# next_non_pr106_l5_action=materialize_tt5l_contest_full_frame_sideinfo_consumption_proof
# ready_for_exact_eval_dispatch=false

tools/build_tt5l_contest_sideinfo_consumption_proof.py --help
# direct executable path works and re-enters the repo venv before importing TT5L deps
```
