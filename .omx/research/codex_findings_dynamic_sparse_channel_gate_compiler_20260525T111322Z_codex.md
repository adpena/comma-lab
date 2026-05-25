# Codex Findings: Dynamic Sparse Channel Gate Compiler Bridge

UTC: 2026-05-25T11:13:22Z
Agent: Codex
Lane: `codex_dynamic_sparse_channel_gate_compiler_20260525`
Authority: planning and candidate-generation only; no score claim, promotion claim, rank/kill authority, or exact-eval dispatch authority

## Scope

Converted the MUDDFormer / NanoGPT speedrun signal into a receiver/compiler
handoff instead of leaving it as a standalone oracle. The useful external signal
is not a compression score claim; it is the pattern of sparse late-layer,
data-dependent gates with a shared small projection and zero-init/no-op safety.

Primary sources checked:

- `https://arxiv.org/abs/2502.12170`
- `https://github.com/Caiyun-AI/MUDDFormer`
- `https://x.com/classiclarryd/status/2058486428255035457`
- `https://github.com/KellerJordan/modded-nanogpt/pull/259`

## Findings And Fixes

- Added channel-aware dynamic sparse gate selection that keeps source and
  channel dimensions separate, matching the practical PR #259 lesson that
  value/residual/gate channels should not be averaged away into one scalar.
- Added `operation_set_compiler_hint_from_channel_gate_scores(...)`, emitting
  existing `inverse_action_operation_set_compiler_hint.v1` payloads with nested
  false-authority metadata.
- Added `tools/build_dynamic_sparse_gate_compiler_hint.py`, an operator CLI that
  turns candidate JSON plus coefficient JSON into the same compiler-hint
  contract.
- Proved the bridge reaches automated final-byte operations:
  dynamic channel gate hint -> inverse action surface -> PacketIR operation set
  -> DQS1 materializer backlog/work queue, with exact-eval dispatch still
  blocked until context, runtime proof, byte-closed archive, and auth eval.

## Verification

- `.venv/bin/python -m ruff check src/tac/optimization/dynamic_sparse_gate_oracle.py src/tac/optimization/__init__.py src/tac/tests/test_dynamic_sparse_gate_oracle.py src/tac/tests/test_byte_shaving_campaign_queue.py tools/build_dynamic_sparse_gate_compiler_hint.py`
  - Result: pass
- `.venv/bin/python -m pytest src/tac/tests/test_dynamic_sparse_gate_oracle.py src/tac/tests/test_byte_shaving_campaign_queue.py::test_dynamic_sparse_channel_gate_hint_reaches_materializer_work_queue -q`
  - Result: `7 passed in 0.45s`
- `.venv/bin/python -m pytest src/tac/tests/test_dynamic_sparse_gate_oracle.py src/tac/tests/test_family_agnostic_materializers.py src/tac/tests/test_experiment_queue_observer.py src/tac/tests/test_byte_shaving_materializer_campaign_runner.py src/tac/tests/test_inverse_steganalysis_acquisition.py src/tac/tests/test_byte_shaving_campaign_queue.py src/tac/tests/test_family_agnostic_materializer_sweep.py src/tac/tests/test_inverse_steganalysis_action_functional_cli.py -q --durations=20`
  - Result: `269 passed in 5.23s`
- `.venv/bin/python tools/review_tracker.py policy-check src/tac/tests/test_byte_shaving_campaign_queue.py`
  - Result: pass, 94 entities compliant, 0 violations

## Remaining Work

- Feed real inverse-action/materializer observation rows into this CLI rather
  than synthetic fixture candidates.
- Add an MLX parity implementation for local advisory training once the NumPy
  oracle has stable queue-side behavior.
- Train or fit the gate on queue/materializer feedback and keep the learned
  selector advisory-only until byte-closed materialized candidates pass exact
  auth eval.
