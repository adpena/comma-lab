# Codex Findings: Materializer Runtime Tree Closure

Date: 2026-05-27T16:33:04Z

## Verdict

Materializer submission closure now fails closed when a generated receiver
runtime adapter is available by path but no expected runtime-tree SHA is bound
to the row or proof. A live `inflate.sh` directory is not enough to close a
receiver adapter; the closure must match an explicit expected tree identity.

## Landed Integration

- `materializer_submission_closure` now recognizes explicit expected runtime
  tree fields when choosing the adapter runtime: `expected_runtime_tree_sha256`,
  `expected_inflate_runtime_tree_sha256`, and
  `expected_candidate_runtime_tree_sha256`.
- `_resolve_runtime_adapter_dir(...)` refuses adapter closure with
  `runtime_adapter_expected_tree_sha_missing` if no expected tree identity is
  present.
- The closure still accepts proof-backed adapter runtimes when the live tree
  hash matches a row/proof-bound expected tree hash.
- Static source-runtime closure tests now use proof fixtures with concrete
  `runtime_consumption_probe` evidence, matching the stricter exact-readiness
  proof contract instead of relying on top-level `passed=true`.

## Safeguards

- Refused runtime adapter closure emits an archive/proof-only static custody
  packet with `ready_for_exact_eval_dispatch=false`.
- The closed source queue carries the refusal blocker into both readiness and
  dispatch blockers.
- No score, promotion, rank/kill, exact-eval dispatch, or budget authority is
  granted by closure.

## Verification

- `.venv/bin/python -m ruff check src/tac/optimizer/materializer_submission_closure.py src/tac/tests/test_materializer_submission_closure.py`
- `.venv/bin/python -m py_compile src/tac/optimizer/materializer_submission_closure.py src/tac/tests/test_materializer_submission_closure.py`
- `.venv/bin/python -m pytest src/tac/tests/test_materializer_submission_closure.py -q`
  - `6 passed`
- `.venv/bin/python -m pytest src/tac/tests/test_materializer_chain_harvest_scheduler.py::test_harvest_family_agnostic_declaration_only_proof_is_not_receiver_ready src/tac/tests/test_materializer_chain_harvest_scheduler.py::test_harvest_rejects_chain_manifest_stale_runtime_tree_identity -q`
  - `2 passed`
- `.venv/bin/python -m pytest src/tac/tests/test_experiment_queue_observer.py::test_observer_rejects_materializer_with_only_proof_present_flag src/tac/tests/test_experiment_queue_observer.py::test_observer_rejects_materializer_stale_runtime_tree_identity -q`
  - `2 passed`
- `.venv/bin/python -m pytest src/tac/tests/test_experiment_queue.py::test_materializer_chain_complete_requires_runtime_identity_by_default -q`
  - `1 passed`

## Remaining Work

The closure layer now requires expected runtime identity for generated receiver
adapters. The next false-authority hardening step is to make every materializer
queue row emit the expected runtime-tree identity explicitly at the postcondition
boundary, then reject observer `postcondition_passed` for any archive/runtime
artifact that cannot prove candidate archive custody, runtime identity, receiver
proof schema, and false authority directly from the artifact bytes.
