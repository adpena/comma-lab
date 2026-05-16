# L5 v2 asymptotic lane-registry adversarial audit

- schema: `l5_v2_asymptotic_lane_registry_adversarial_audit_v1`
- created_at_utc: `2026-05-16T22:26:12Z`
- source: `read_only_subagent_audit_nietzsche_the_3rd`
- score_claim: `false`
- promotion_eligible: `false`
- ready_for_exact_eval_dispatch: `false`

## Finding

The L5 v2 TT5L core readiness surface is currently fail-closed for architecture
lock: Dykstra and move-level artifacts are valid, but probe evidence, paired
CPU/CUDA planning, side-info effect curve, timing smoke custody, and anchor-pair
evidence still block lock authority. This matches the intended L5 v2 posture.

The highest-risk 12-month annoyance is separate: asymptotic candidate surfaces
can drift from lane-registry truth. A Z6/Z7/Z8 asymptotic candidate can appear
as code-level candidate state and as a completed dispatch-claim row while its
lane id is absent from `.omx/state/lane_registry.json`. AGENTS.md treats the
lane registry as the deduplication layer, so this mismatch can cause duplicate
or stale Z6 work later.

The asymptotic candidate rows also mark `ready_for_l1_build` from ledger
presence alone. That is defensible only if the field means "ready to start the
first build task"; it must not be read as scaffold readiness. The scoping memos
still require first-artifact gates: Z6 needs scaffold and identity-predictor
probe evidence, Rudin needs Dykstra feasibility, and Tishby IB-pure needs D4
and IB-tractability gates.

## Recommended patch

Add an `l5_v2_asymptotic_next_action_status` artifact or payload section with
per-candidate fields:

- `lane_registry_registered`
- `ledger_present`
- `ledger_sha256`
- `expected_first_artifact_status`
- `next_prerequisite_status`
- `ready_for_l1_build_semantics`

Wire the status into `l5_v2_asymptotic_pursuit_candidates()`,
`tools/operator_briefing.py`, and `tools/all_lanes_preflight.py`. Add a
regression test that fails if a candidate ledger is present but its lane id is
absent from `.omx/state/lane_registry.json`, unless the row explicitly declares
a canonical replacement lane id.

The first data fix should register the Z6 scoping lane or intentionally map it
to an existing canonical lane id.
