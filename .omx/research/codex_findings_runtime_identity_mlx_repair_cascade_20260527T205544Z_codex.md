# Runtime Identity And MLX Repair Cascade Closure - Codex Findings

Date: 2026-05-27T20:55:44Z
Author: Codex
Commit under review: `dd239f0e1 Harden runtime identity and MLX repair cascade queue`

## What Landed

- Runtime-adapter identity now rejects conflicting nested runtime tree identities instead of accepting the first discovered hash. This closes the failure mode where a parent row and receiver verification could each claim a different observed or expected runtime tree while still satisfying a schema-level check.
- The byte-shaving materializer completion contract now carries the expected runtime-tree hash and tests that a nested receiver-verification mismatch fails the materializer postcondition.
- Repair-waterfill queues now emit a nested `repair_cascade_mlx_probe_queue.json` after the repair waterfill work order exists, then validate and bounded-run that queue locally. Cascade C therefore becomes a queue-owned MLX-local probe stage rather than an operator-remembered follow-up.
- The generated Cascade-C probe queue remains advisory-only: `[macOS-MLX research-signal]`, no score claim, no budget spend authority, no promotion authority, and no exact-dispatch authority.

## Authority Boundary

The repair cascade queue is intentionally downstream of the encoder-side waterfill work order. It can consume the typed response ledger, structural cascade opportunity rows, and receiver-closed rate credit context, but it cannot spend bytes, claim score, promote a candidate, or dispatch exact eval. Any useful MLX signal must still return through component-response artifacts, receiver consumption proof, and contest CPU/CUDA exact-axis gates.

## Verification

- `.venv/bin/python -m ruff check src/comma_lab/scheduler/frontier_rate_attack_feedback.py src/tac/optimization/runtime_adapter_identity.py src/tac/tests/test_frontier_rate_attack_feedback.py src/tac/tests/test_runtime_adapter_identity.py src/tac/tests/test_byte_shaving_campaign_queue.py`
- `.venv/bin/python -m pytest src/tac/tests/test_frontier_rate_attack_feedback.py -q`
- `.venv/bin/python -m pytest src/tac/tests/test_repair_cascade_mlx_probe_queue.py -q`
- `.venv/bin/python -m pytest src/tac/tests/test_runtime_adapter_identity.py src/tac/tests/test_byte_shaving_campaign_queue.py::test_materializer_chain_completion_contract_rejects_schema_only_and_failure -q`
- `.venv/bin/python tools/lane_maturity.py validate`
- `.venv/bin/python tools/review_gate_hook.py`

## Next Integration Hooks

- Feed completed `repair_cascade_mlx_probe_queue` specs into component-response harvest rows so MLX-local Cascade-C measurements update the same posterior as repair stackability probes.
- Add deterministic replay bundles for Cascade-C local MLX probes once a concrete local response artifact exists.
- Promote only byte-closed, receiver-consumed repair children that preserve the rate-only parent and pass exact CPU/CUDA component replay gates.
