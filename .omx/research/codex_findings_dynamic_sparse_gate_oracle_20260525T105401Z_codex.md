# Codex Findings: Dynamic Sparse Gate Oracle

UTC: 2026-05-25T10:54:01Z
Agent: Codex
Lane: `codex_dynamic_sparse_gate_oracle_20260525`

## Scope

Converted the MUDDFormer / NanoGPT speedrun signal into a reusable Pact
planning primitive without importing transformer-specific architecture
assumptions. The implementation is a NumPy, false-authority oracle that can
generate dynamic sparse source-operation coefficients and lower selected
operations into existing inverse-action compiler hints.

Primary sources checked:

- `https://arxiv.org/abs/2502.12170`
- `https://github.com/KellerJordan/modded-nanogpt/pull/259`
- `https://x.com/classiclarryd/status/2058486428255035457`

## Findings And Fixes

- Added `tac.optimization.dynamic_sparse_gate_oracle`, a planning-only
  MUDD-style gate with RMSNorm, GELU, two-matrix coefficient generation, sparse
  source mixtures, and exact zero-init no-op proof.
- Added `operation_set_compiler_hint_from_gate_scores(...)` so dynamic gate
  coefficients rank and select existing final-byte operations, then emit
  `inverse_action_operation_set_compiler_hint.v1` payloads.
- Preserved false-authority boundaries on the oracle result, each selected
  operation, and nested gate-selection metadata.
- Verified the emitted compiler hint lowers through PacketIR using the existing
  deterministic operation-set compiler path.
- Exported the oracle and hint builder from `tac.optimization` lazily for normal
  operator use.

## Verification

- `.venv/bin/python -m ruff check src/tac/optimization/dynamic_sparse_gate_oracle.py src/tac/tests/test_dynamic_sparse_gate_oracle.py src/tac/optimization/__init__.py`
  - Result: pass
- `.venv/bin/python -m pytest src/tac/tests/test_dynamic_sparse_gate_oracle.py -q`
  - Result: `3 passed`
- `.venv/bin/python -m pytest src/tac/tests/test_dynamic_sparse_gate_oracle.py src/tac/tests/test_byte_shaving_campaign_queue.py::test_packet_ir_operation_set_lowers_to_materializer_backlog_rows src/tac/tests/test_byte_shaving_campaign_queue.py::test_packet_ir_operation_set_lowering_keeps_unknown_target_blocked src/tac/tests/test_queue_feedback_replan_policy.py::test_queue_observation_helpers_are_public_package_exports -q`
  - Result: `6 passed`
- `.venv/bin/python -m pytest src/tac/tests/test_dynamic_sparse_gate_oracle.py src/tac/tests/test_family_agnostic_materializers.py src/tac/tests/test_experiment_queue_observer.py src/tac/tests/test_byte_shaving_materializer_campaign_runner.py src/tac/tests/test_inverse_steganalysis_acquisition.py src/tac/tests/test_byte_shaving_campaign_queue.py src/tac/tests/test_family_agnostic_materializer_sweep.py src/tac/tests/test_inverse_steganalysis_action_functional_cli.py -q`
  - Result: `265 passed`
- `.venv/bin/python - <<'PY' ...`
  - Verified lazy `tac.optimization` exports resolve the new functions.

## Remaining Work

- Add an MLX parity implementation after the NumPy oracle is stable.
- Feed real inverse-action operation candidates through the gate as an advisory
  selector for grouped final-byte sweeps.
- If the gate proves useful, add a queue-owned materializer campaign that treats
  dynamic gate selection as candidate-generation signal only, then lets exact
  materializer observations flow back through the queue feedback bridge.
