# Codex Findings - Packet Member ZIP Header Elide Materializer - 2026-05-24T22:56:23Z

## Scope

Reviewed and completed the packet-member ZIP-header elision lane as a
family-agnostic materializer for HNeRV, HNeRV bolt-ons, NeRV-family packets, and
non-NeRV ZIP archives.

## Findings

- The registry had a planning contract for `packet_member_zip_header_elide_v1`,
  but the executable path needed the same end-to-end custody as the other
  family-agnostic materializers.
- The completed materializer strips deterministic ZIP/member metadata while
  preserving member payload bytes and emitting a receiver/runtime proof.
- The queue, final-byte context compiler, CLI, postconditions, telemetry, and
  harvestable-schema allowlist now consume the target through the normal
  local-proof-chain path.
- The materializer remains false-authority by construction:
  `score_claim=false`, `promotion_eligible=false`,
  `rank_or_kill_eligible=false`, and
  `ready_for_exact_eval_dispatch=false`.

## Verification

- `.venv/bin/python -m ruff check src/tac/optimization/family_agnostic_materializers.py tools/run_family_agnostic_materializer.py src/comma_lab/scheduler/byte_shaving_campaign_queue.py src/comma_lab/scheduler/byte_shaving_materializer_registry.py src/comma_lab/scheduler/final_byte_operation_contexts.py src/tac/tests/test_family_agnostic_materializers.py src/tac/tests/test_byte_shaving_campaign_queue.py src/tac/tests/test_final_byte_operation_contexts.py`
- `.venv/bin/python -m pytest src/tac/tests/test_family_agnostic_materializers.py src/tac/tests/test_byte_shaving_campaign_queue.py::test_materializer_registry_has_family_agnostic_fail_closed_targets src/tac/tests/test_byte_shaving_campaign_queue.py::test_materializer_work_queue_wraps_packet_member_zip_header_elide src/tac/tests/test_final_byte_operation_contexts.py -q`
- `.venv/bin/python -m pytest src/tac/tests/test_byte_shaving_campaign.py::test_inverse_action_compiler_hint_lowers_registered_targets_and_aliases src/tac/tests/test_byte_shaving_campaign.py::test_inverse_action_compiler_hint_lowers_to_family_packet_ir src/tac/tests/test_byte_shaving_campaign.py::test_direct_mlx_spend_triage_compiler_hint_lowers_to_packet_ir -q`

## Remaining Gates

- No contest score authority is created by this lane.
- Real-archive empirical savings, same-runtime inflate parity, and exact
  CPU/CUDA auth anchors remain required before any promotion decision.
