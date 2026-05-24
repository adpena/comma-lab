# Codex Findings: Inverse-Action Executable Materializer Fixture

UTC: 2026-05-24T19:37:22Z
Agent: Codex
Lane: codex_inverse_action_executable_materializer_fixture_20260524

## Verdict

The inverse-steganalysis apparatus is partially implemented, not yet fully autonomous.
The immediate blocker found in this pass was concrete: a single water-bucket cell
with an `operation_set_compiler` hint could compile into a family-agnostic
operation, but the byte-shaving planner only emitted PacketIR operation sets for
multi-unit combinations. That made a strict one-cell inverse-action selection
stop at planning/bridge metadata instead of entering the executable
materializer queue.

## Fix Landed

- `src/tac/optimization/byte_shaving_campaign.py` now emits a singleton
  operation-set envelope only for compiled inverse-action compiler hints whose
  target kind is one of the registered family-agnostic compiler defaults.
- Unsupported/high-level inverse-action cells remain compiler-required and do
  not get false PacketIR readiness.
- Ordinary one-operation planner rows remain on the prior path, avoiding
  duplicate materializer backlog rows for non-inverse direct byte-range cases.
- `src/tac/tests/test_byte_shaving_campaign_queue.py` adds an end-to-end fixture:
  inverse-action functional -> signal surface -> campaign plan -> PacketIR
  bridge -> artifact-map context compiler -> work queue -> real
  `tools/run_family_agnostic_materializer.py` execution -> byte-closed archive
  + runtime-consumption proof.

## Verification

- `.venv/bin/python -m pytest src/tac/tests/test_byte_shaving_campaign.py src/tac/tests/test_byte_shaving_campaign_queue.py -q`
  passed: 79 tests.
- `.venv/bin/python -m ruff check src/tac/optimization/byte_shaving_campaign.py src/tac/tests/test_byte_shaving_campaign_queue.py`
  passed.

## Remaining Gaps

1. Convert more `high_level_operation_compiler_required` cells into concrete
   compiler hints for HNeRV/NeRV, HNeRV boltons, tensors, packets, and
   non-NeRV payloads.
2. Add proof-backed materializers for currently planning-only archive/header,
   packet, and tensor operations.
3. Extend archive-section runtime proofs beyond length-preserving Brotli
   recodes to runtime-consumed offset/layout changes.
4. Densify hydrated scorer-response fields and MLX/CPU/CUDA calibration so the
   water bucket is fed by component-local scorer evidence rather than mostly
   static/rule-based priors.
5. Wire materializer campaign feedback into automatic queue replan/resume policy
   rather than leaving failed proof-chain rows as operator-read artifacts.

## Authority Boundary

All new artifacts remain planning/proof-chain only. They do not claim score,
promotion eligibility, rank/kill authority, dispatch authority, or exact auth
readiness. Exact frontier movement still requires byte-closed archive custody,
runtime proof, exact-readiness handoff, lane claim, and contest CPU/CUDA auth
eval.
