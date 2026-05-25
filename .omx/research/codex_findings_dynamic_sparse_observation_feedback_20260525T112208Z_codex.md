# Codex Findings: Dynamic Sparse Observation Feedback Compiler

- UTC: 2026-05-25T11:22:08Z
- Lane: `codex_dynamic_sparse_observation_feedback_20260525`
- Status: scaffold integrated; planning-only; not score authority

## Finding

The dynamic sparse channel-gate compiler could rank source/channel coefficients,
but queue and materializer observations still had no direct path back into that
compiler surface. That left real local proof-chain outcomes as downstream prose
instead of reusable acquisition signal.

## Landed Integration

- Added `operation_set_compiler_hint_from_observation_feedback(...)` to convert
  queue/materializer observation rows into `inverse_action_operation_set_compiler_hint.v1`
  via the existing channel-gate compiler path.
- Added `tools/build_dynamic_sparse_gate_compiler_hint.py --observations` so
  observation JSON can be compiled without hand-written adapters.
- Preserved false-authority boundaries: observation feedback is candidate
  generation only, carries no score claim, and still requires byte-closed
  materialization plus exact auth eval before promotion.
- Proved the observation-feedback hint lowers through PacketIR into the
  materializer work queue.

## Verification

- `.venv/bin/python -m ruff check src/tac/optimization/dynamic_sparse_gate_oracle.py src/tac/optimization/__init__.py src/tac/tests/test_dynamic_sparse_gate_oracle.py src/tac/tests/test_byte_shaving_campaign_queue.py tools/build_dynamic_sparse_gate_compiler_hint.py`
- `.venv/bin/python -m pytest src/tac/tests/test_dynamic_sparse_gate_oracle.py src/tac/tests/test_byte_shaving_campaign_queue.py::test_dynamic_sparse_observation_feedback_hint_reaches_materializer_work_queue -q`
- `.venv/bin/python -m pytest src/tac/tests/test_dynamic_sparse_gate_oracle.py src/tac/tests/test_byte_shaving_campaign_queue.py -q`
- `git diff --check`

## Remaining Work

- Feed live materializer observation artifacts into `--observations` from the
  queue runner, then compare grouped structural priors against measured
  interaction deltas.
- Let successful queue observations update acquisition priors automatically
  instead of requiring a manual compiler invocation.
- Keep all rows planning-only until a byte-closed candidate passes exact
  readiness and contest auth eval.
