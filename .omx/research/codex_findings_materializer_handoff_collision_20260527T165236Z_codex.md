# Codex Findings: Materializer Handoff Collision Guard

Generated at: 2026-05-27T16:52:36Z

## Scope

Queue-owned final-rate attack and feedback closure for the current
`[contest-CPU]` frontier archive:

- archive SHA-256: `0a3abfe645c4fac0df9ea89237f25dd9bfc6b2471b897c36d7437795d27d1403`
- archive bytes: `178546`
- score axis: `[contest-CPU]`
- score: `0.19202062679074616`
- local queue artifact:
  `.omx/research/frontier_final_rate_attack_all_exec_20260527Tlocal1/`
- external materializer results:
  `/Volumes/VertigoDataTier/experiments/results/frontier_final_rate_attack/all_exec_20260527Tlocal1/`

## Empirical Findings

The all-exec final-rate attack ran three executable materializer families:

- `archive_zip_repack_v1`
- `packet_member_zip_header_elide_v1`
- `packet_member_recompress_v1`

All three completed with zero failed queue commands and no exact-dispatch-ready
candidate. Each produced `candidate_not_rate_positive` / zero-delta evidence
against the current single-member FECa frontier archive.

Three requested targets were correctly omitted with typed blockers:

- `packet_member_merge_v1`: `packet_member_merge_requires_at_least_two_members`
- `renderer_payload_dfl1_v1`:
  `renderer_payload_dfl1_missing_members:renderer.bin,masks.mkv,optimized_poses.pt`
- `archive_section_entropy_recode_v1`:
  derived section parser hit `PR101 FEC6 selector magic mismatch: b'FECa'`, then
  omitted as lacking a usable section manifest

## Bug Class Fixed

A regenerated feedback operation-materializer queue exposed a non-score bug
class: when multiple executable materializer rows write candidate manifests in
the same directory, their exact-readiness follow-up paths previously converged
on one shared `exact_eval_handoff/` directory. Later rows could overwrite
`source_queue.json`, `harvest_report.json`, `submission_closure_report.json`,
`exact_readiness_bridge_report.json`, and `dispatch_plan.json` for earlier
rows.

This did not promote a false score, but it was signal-loss risk.

## Code Guard

`build_materializer_execution_queue(...)` now detects shared materializer
manifest parent directories. When a collision is present, it emits deterministic
per-experiment handoff directories under:

`exact_eval_handoff/<experiment_id>/`

Single-row queues retain the previous compact path for operator readability.

Regression test:

- `test_materializer_execution_queue_isolates_followups_for_shared_manifest_parent`

## Verification

Commands run:

- `.venv/bin/python -m pytest src/tac/tests/test_byte_shaving_campaign_queue.py::test_materializer_execution_queue_can_append_exact_readiness_followups src/tac/tests/test_byte_shaving_campaign_queue.py::test_materializer_execution_queue_isolates_followups_for_shared_manifest_parent src/tac/tests/test_frontier_rate_attack_bootstrap.py::test_frontier_bootstrap_can_append_exact_readiness_followups -q`
- `.venv/bin/python -m ruff check src/comma_lab/scheduler/byte_shaving_campaign_queue.py src/tac/tests/test_byte_shaving_campaign_queue.py`

The regenerated isolated operation-materializer queue completed 10/10 local
steps with no failed or blocked steps. It preserved separate handoff reports for
both operation rows:

- `packet_member_recompress_v1`
- `packet_member_zip_header_elide_v1`

Both rows remained false-authority and exact-dispatch-refused because the
realized archive delta was zero bytes.

## Continual-Learning Impact

The current frontier archive shape is now learned as saturated for the basic
repack/header/recompress pass on this axis. The next high-EV path is not to
rerun these leaves; it is to spend automation on receiver/runtime proof closure
for missing higher-level materializers and on grouped inverse-steganalysis
operations that can change distortion/rate jointly.

Immediate automation target:

1. Feed this zero-delta materializer evidence into the feedback refresh and
   operation portfolio.
2. Prioritize runtime-consumption proof repairs for materializers whose blockers
   are `requires_queue_context_and_runtime_consumption_proof`.
3. Continue Cascade C work: PoseNet-null subset detection plus SegNet-region
   water filling plus per-region selector coding.

All artifacts are local/advisory only. No score, promotion, rank/kill, budget
spend, or exact-eval dispatch authority is claimed here.
