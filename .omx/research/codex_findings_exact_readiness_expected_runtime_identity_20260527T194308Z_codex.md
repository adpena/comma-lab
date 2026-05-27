# Codex Findings - Exact Readiness Expected Runtime Identity

Generated: 2026-05-27T19:43:08Z

## Summary

The expected-runtime-tree identity contract now reaches exact-readiness and
submission-closure consumers instead of stopping at materializer harvest.

## Engineering Outcome

- Exact-readiness records both observed and expected runtime tree hashes from
  candidate rows and runtime-consumption proofs.
- Runtime-ready rows now fail closed when either the proof or candidate row
  omits expected runtime identity, or when expected identity disagrees with the
  observed runtime tree.
- Packet-member-merge and tensor-factorize receiver runtime manifests now emit
  `expected_runtime_tree_sha256` beside the observed runtime tree hash.
- Materializer submission closure no longer treats observed runtime-tree hashes
  as expected hashes.
- Materializer-chain harvest observer overlay now records same-value runtime
  identity fields as applied, preserving provenance without creating false
  conflicts.

## Authority

This is a custody and automation guard only. It does not claim score, promote a
candidate, rank or kill a row, launch paid work, or mark proxy/local/MLX rows
ready for exact eval dispatch.

## Validation

- `.venv/bin/python -m ruff check src/comma_lab/scheduler/materializer_chain_harvest.py src/tac/optimization/packet_member_merge_receiver.py src/tac/optimization/tensor_factorize_receiver.py src/tac/optimizer/exact_readiness.py src/tac/optimizer/materializer_submission_closure.py src/tac/tests/test_frontier_rate_attack_bootstrap.py src/tac/tests/test_materializer_submission_closure.py src/tac/tests/test_optimizer_exact_readiness.py`
- `.venv/bin/python -m py_compile src/comma_lab/scheduler/materializer_chain_harvest.py src/tac/optimization/packet_member_merge_receiver.py src/tac/optimization/tensor_factorize_receiver.py src/tac/optimizer/exact_readiness.py src/tac/optimizer/materializer_submission_closure.py src/tac/tests/test_frontier_rate_attack_bootstrap.py src/tac/tests/test_materializer_submission_closure.py src/tac/tests/test_optimizer_exact_readiness.py`
- `.venv/bin/python -m pytest src/tac/tests/test_optimizer_exact_readiness.py src/tac/tests/test_materializer_submission_closure.py src/tac/tests/test_materializer_chain_harvest_scheduler.py src/tac/tests/test_experiment_queue_observer.py src/tac/tests/test_family_agnostic_materializers.py src/tac/tests/test_frontier_rate_attack_bootstrap.py -q`

Final focused result: `239 passed`.

- `.venv/bin/python tools/lane_maturity.py validate`
  - `1443 lane(s) validated cleanly`
- `.venv/bin/python tools/review_gate_hook.py`
  - passed

## Next Integration

The next final-rate campaign run should consume this stricter exact-readiness
boundary: receiver-closed materializer rows that lack expected runtime identity
must stay in planning/repair queues until the receiver proof can declare and
verify the expected runtime tree.
