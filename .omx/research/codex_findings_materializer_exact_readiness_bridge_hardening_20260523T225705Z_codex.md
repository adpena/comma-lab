# Codex Findings - Materializer Exact-Readiness Bridge Hardening

UTC: 2026-05-23T22:57:05Z

## Finding

Schrodinger reviewed the materializer harvest exact-readiness bridge after the
initial landing and found four authority/custody risks:

- the aggregate bridge report exposed nested
  `ready_for_exact_eval_dispatch=true` fields;
- PR101 runtime-consumption proofs were not bound to the promoted `inflate.sh`
  / `inflate.py` runtime files;
- operator-supplied extra source blockers could be cleared without a narrow
  bridge allowlist;
- duplicate `candidate_id` rows could cause the bridge to promote the wrong
  source row through the existing candidate-id lookup.

## Fix

The bridge report remains planning-only and now contains no nested truthy
authority fields. Per-candidate dispatch authority is still emitted only in
the separate exact-ready queue artifact written by the existing promoter.

The exact-readiness gate now fail-closes PR101 runtime-consumption proofs unless
the proof is bound to the candidate submission directory and actual runtime
file hashes:

- proof `packet_dir` must resolve to the candidate `submission_dir`;
- proof `manifest_path` must exist and match `manifest_sha256`;
- proof `inflate_wrapper_route_proof.inflate_sh_sha256` must match the live
  `inflate.sh`;
- proof `inflate_static_bias_patch_proof` /
  `inflate_wrapper_route_proof.packet_inflate_py_sha256` /
  `inflate_runtime_bias_logic_proof.inflate_py_sha256` must match the live
  `inflate.py`;
- proof route/runtime booleans must still prove wrapper routing and runtime
  logic execution.

The bridge also rejects nonempty `dispatch_ready` source queues, duplicate
candidate IDs, and all operator-supplied extra source blockers unless they are
explicitly allowlisted in the scheduler. The current allowlist is empty because
the bridge's known materializer blockers are already encoded.

## Verification

- `PYTHONPATH=. .venv/bin/python -m pytest src/tac/tests/test_materializer_chain_harvest_scheduler.py src/tac/tests/test_optimizer_exact_readiness.py -q`
  - 56 passed
- `PYTHONPATH=. .venv/bin/python -m pytest src/tac/tests/test_materializer_chain_harvest_scheduler.py src/tac/tests/test_optimizer_candidate_queue.py src/tac/tests/test_optimizer_exact_readiness.py src/tac/tests/test_experiment_queue.py src/tac/tests/test_byte_shaving_campaign_queue.py src/tac/tests/test_serialized_archive_economics.py -q`
  - 176 passed
- `.venv/bin/python -m ruff check src/comma_lab/scheduler/materializer_chain_harvest.py tools/harvest_materializer_chain_candidates.py src/tac/optimizer/exact_readiness.py src/tac/tests/test_materializer_chain_harvest_scheduler.py src/tac/tests/test_optimizer_exact_readiness.py`
  - passed
- `git diff --check`
  - passed

## Remaining Gap

This bridge still only converts harvested materializer source queues into
exact-readiness artifacts. It does not dispatch, claim score authority, or
replace full contest CPU/CUDA auth eval. The next useful automation step is a
queue-owned consumer that watches for per-candidate exact-ready queue artifacts,
claims the lane, and dispatches exact eval only after the existing lane-claim
and auth-axis gates accept the packet.
