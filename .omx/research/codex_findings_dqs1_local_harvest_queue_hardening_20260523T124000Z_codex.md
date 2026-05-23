# Codex Findings - DQS1 Local Harvest Queue Hardening

- Timestamp: 2026-05-23T12:40:00Z
- Lane: `lane_dqs1_local_harvest_observation_canonicalization_20260523`
- Scope: DQS1 local-first harvest canonicalization, queue/autopilot safety, and disk-pressure retention.

## Findings

1. The DQS1 local-first queue builder consumed only `pairset_component_marginal_canonicalization_summary.v1`, while the current cross-family planner emits `cross_family_candidate_portfolio_action_summary.v1`. This broke the intended planner -> queue path with `missing portfolio_json`. Fixed by accepting both summary schemas and resolving `portfolio_json` or `json_out` under the same false-authority checks.

2. `tools/plan_cross_family_candidate_portfolio.py` omitted `dispatch_attempted=false` and `gpu_launched=false` from action-summary false-authority payloads. Downstream queue safety gates require those fields. Fixed in the shared false-authority payload and covered by CLI tests.

3. A broad harvest glob, `.omx/research/dqs1_local_first_harvest_*.json`, matched generated observation-summary artifacts and failed closed. Fixed the observation-builder CLI to ignore generated `dqs1_local_first_harvest_observations_*` artifacts for glob matches while preserving explicit-path failure semantics.

4. `run_dqs1_local_first_autopilot.py --max-candidates 1` did not bound candidate starts when the YAML contained multiple experiments; it allowed the worker to complete candidate A and start candidate B before harvest. Fixed by adding a reusable `max_experiments` worker window and wiring DQS1 autopilot to one experiment per harvest boundary.

5. The local run produced a valid macOS CPU advisory for `pairset_drop_two_r010_009_p0376_0459`: `0.19203961709818362`, eureka `observe_only`, conservative projected contest CPU `0.1920321170981836`. This is planning signal only, not score/rank/promotion authority.

6. The interrupted multi-candidate run left certified rebuildable raw scratch. Executed retention with a durable plan and journal, deleting 25,637,045,658 bytes from the completed first candidate and partial second candidate while preserving manifests/evidence.

## Artifacts

- New harvest: `.omx/research/dqs1_local_first_harvest_pairset_drop_two_r010_009_p0376_0459_20260523T123100Z.json`
- New eureka signal: `.omx/research/local_cpu_contest_drift_eureka_pairset_drop_two_r010_009_p0376_0459_20260523T121625Z.json`
- New canonical observations: `.omx/research/dqs1_local_first_harvest_observations_20260523T123200Z.jsonl`
- Retention plan/journal: `.omx/research/dqs1_artifact_retention_drop_two_r010_queue_stop_20260523T123000Z.json`
- Checked-in queue now points at the next single candidate: `pairset_drop_two_r010_005_p0376_0467`

## Status

All new signals are false-authority local advisory artifacts. Exact contest CPU/CUDA auth eval remains required before any frontier or promotion claim.
