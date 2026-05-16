# L5 scaffold dispatch-claim and lattice blocker hardening

Date: 2026-05-16

## Objective

Preserve partner WIP for the L5/L5 v2 scaffold surface while closing two
false-authority paths found during adversarial review:

1. Remote L5 scaffold drivers could carry a `DISPATCH_INSTANCE_JOB_ID` and set
   `CLAIM_VERIFIED=1` without proving the corresponding active row existed in
   `.omx/state/active_lane_dispatch_claims.md`.
2. The Cathedral autopilot's compressive-sensing lattice diagnostic could be
   visible in JSON while a non-EXACT recovery regime was not also converted
   into a dispatch blocker on candidate rows.
3. A broader scan found the same remote-worker claim custody pattern in
   `scripts/remote_lane_substrate_nscs01_nullspace_split_renderer.sh`: it
   force-created `active_nscs01_remote_driver` from inside the worker and used
   the invalid `--instance-id` flag for terminal rows instead of verifying the
   pre-existing dispatch claim.

This ledger records hardening only. It makes no score claim and does not
promote any scaffold to paid dispatch or exact eval.

## Changes

- `scripts/remote_lane_substrate_rudin_floor_interpretable_ml.sh`
  now runs `tools/claim_lane_dispatch.py summary --live-only --format json`
  and verifies an active `(lane_id, instance_job_id)` row before bootstrap or
  trainer invocation. Missing helper, summary failure, or missing active row
  fail closed before any work proceeds. Terminal-claim cleanup remains in the
  `EXIT` trap and records claim-verification failure separately from later
  remote-driver failure.

- `scripts/remote_lane_substrate_time_traveler_l5_z6.sh` receives the same
  active-claim verification and restores the full-run default `Z6_EPOCHS=300`;
  the trainer-side smoke cap remains the authority that turns smoke requests
  into at most three epochs.

- `src/tac/optimization/l5_staircase_v2.py` marks already-present L1 scaffold
  actions as `recommended_next_action_status="completed_or_superseded"` and
  sets `ready_for_recommended_next_action=false`. This prevents the operator
  briefing from repeatedly recommending the already-landed Z6/Rudin/Tishby L1
  scaffold build actions.

- `tools/cathedral_autopilot_autonomous_loop.py` converts a lattice diagnostic
  with `recovery_regime != "EXACT"` into a candidate blocker named
  `compressive_sensing_lattice_recovery_regime_<REGIME>_operator_review_required`.
  The lattice record remains `score_claim=false`,
  `promotion_eligible=false`, and `ready_for_exact_eval_dispatch=false`.

- `tools/operator_briefing.py` now renders the effective L5 v2 next action and
  its completed-or-superseded status, including L1 blocker detail, rather than
  hiding the reason the original build action is no longer actionable.

- `scripts/remote_lane_substrate_nscs01_nullspace_split_renderer.sh` now uses
  the same active-claim summary verification pattern; terminal rows use
  `--instance-job-id` plus explicit `--claims-path` and `--agent`. The remote
  worker no longer mints its own forced active dispatch claim.

## Verification

Commands run:

```bash
bash -n \
  scripts/remote_lane_substrate_rudin_floor_interpretable_ml.sh \
  scripts/remote_lane_substrate_time_traveler_l5_z6.sh \
  scripts/remote_lane_substrate_nscs01_nullspace_split_renderer.sh
.venv/bin/python -m pytest \
  src/tac/substrates/rudin_floor_interpretable_ml/tests/test_rudin_floor_l1_scaffold.py::test_remote_driver_requires_and_closes_dispatch_claim \
  src/tac/substrates/time_traveler_l5_z6/tests/test_z6.py::test_remote_driver_verifies_active_claim_and_preserves_full_epoch_default \
  src/tac/substrates/nscs01_nullspace_split_renderer/tests/test_nscs01_substrate.py::test_remote_driver_verifies_existing_active_claim_not_force_creates_one \
  src/tac/tests/test_l5_staircase_v2.py::test_l5_v2_asymptotic_pursuit_candidates_are_source_backed \
  src/tac/tests/test_operator_briefing.py::test_briefing_json_composite_has_all_three_keys \
  src/tac/tests/test_cathedral_autopilot_autonomous_loop.py::test_main_surfaces_compressive_sensing_lattice_diagnostic -q
.venv/bin/python -m ruff check \
  src/tac/optimization/l5_staircase_v2.py \
  src/tac/substrates/rudin_floor_interpretable_ml/tests/test_rudin_floor_l1_scaffold.py \
  src/tac/substrates/time_traveler_l5_z6/tests/test_z6.py \
  src/tac/substrates/nscs01_nullspace_split_renderer/tests/test_nscs01_substrate.py \
  src/tac/tests/test_cathedral_autopilot_autonomous_loop.py \
  src/tac/tests/test_l5_staircase_v2.py \
  src/tac/tests/test_operator_briefing.py \
  tools/cathedral_autopilot_autonomous_loop.py \
  tools/operator_briefing.py
.venv/bin/python -m py_compile \
  src/tac/optimization/l5_staircase_v2.py \
  tools/cathedral_autopilot_autonomous_loop.py \
  tools/operator_briefing.py
git diff --check
```

Observed evidence:

- Shell syntax passed for all three changed remote drivers.
- Focused regression tests passed after the NSCS01 sister terminal-claim
  tightening: `6 passed in 8.07s`.
- Ruff passed on changed Python files.
- py_compile passed on changed Python entry points.
- `git diff --check` passed.

Expanded final greenup after integrating the broader Tishby/Z6/L5 v2 worktree:

- Focused L5/L5 v2/autopilot/Tishby/Rudin/Z6 suite passed:
  `248 passed in 18.58s`.
- `tac.preflight --no-codebase` passed.
- `tools/lane_maturity.py validate` passed:
  `773 lane(s) validated cleanly`.
- Z6 diagnostic smoke with default full-run epoch value still capped correctly:
  `requested_epochs=300`, `epochs=3`, `smoke_epoch_cap=3`.
- Tishby diagnostic smoke remained non-claiming:
  `score_claim=false`, `research_only=true`, `score_axis=diagnostic_cpu`,
  `roundtrip_ok=true`.
- Cathedral autopilot lattice smoke emitted
  `compressive_sensing_lattice_recovery_regime_FAILED_operator_review_required`
  into the dispatch event blockers.
- Operator briefing JSON rendered all three L5 v2 sample next actions as
  `completed_or_superseded:*`.

## Evidence boundary

This is a custody and false-authority hardening change. It does not produce a
candidate archive, score, promotion, paid dispatch, or exact-eval result. The
next score-lowering work remains the L5/L5 v2 paired measurement and scaffold
lift path, but these guards prevent stale scaffold actions and lattice
diagnostics from masquerading as launch authority.
