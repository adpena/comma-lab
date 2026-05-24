# Codex Findings: Inverse-Action Compiler Registry Targets

UTC: 2026-05-24T20:39:51Z
Lane: codex_inverse_action_byte_range_compiler_20260524

## Finding

The inverse-steganalysis action surface already had a deterministic
`operation_set_compiler` handoff, but it only lowered three registered
family-operation targets. That left byte-range entropy recode and several
archive/member/tensor materializer contracts stranded as high-level advisory
work even when the action functional already selected deterministic operations.

## Landing

- Extended the inverse-action compiler defaults in
  `src/tac/optimization/byte_shaving_campaign.py` to cover the registered
  byte-range, archive-section, packet-member, and tensor target families.
- Canonicalized stale `byte_range_entropy_coder_v1` compiler hints into the
  registered `byte_range_entropy_recode_v1` target and
  `byte_range_entropy_recode_adapter` materializer.
- Added explicit lowered-operation metadata distinguishing registered
  executable materializers from registered planning/receiver contracts, so
  target-kind expansion does not masquerade as completed implementation.
- Preserved materializer context fields such as `archive_member_name`,
  `archive_byte_range`, `section_manifest`, `packet_member_manifest`,
  `tensor_manifest`, and `runtime_consumption_proof` through PacketIR params.
- Added a regression proving registered targets and aliases lower into
  PacketIR without score, promotion, dispatch, or exact-eval authority.

## Authority Boundary

This is a compiler/materialization bridge improvement, not a score claim.
Lowered rows remain planning-only until materializer contexts, runtime
consumption proof, locality/inflate parity, and exact CPU/CUDA auth-eval gates
clear.

## Xhigh Sidecar Synthesis

Kant's read-only architecture review found that the repo has real
inverse-steganalysis plumbing, but is not yet a full automated final-rate
attack. Its highest-EV implementation recommendation was to land a concrete
registry-backed `operation_set_compiler` slice while keeping executable versus
planning-only materializer status explicit. This patch implements that slice;
the next recommendation remains moving materializer feedback replan from an
opt-in CLI path into queue/DAG-owned local policy.

## Verification

- `.venv/bin/python -m ruff check src/tac/optimization/byte_shaving_campaign.py src/tac/tests/test_byte_shaving_campaign.py`
- `.venv/bin/python -m pytest src/tac/tests/test_byte_shaving_campaign.py -q`
  passed with `21 passed`.
- `.venv/bin/python -m pytest src/tac/tests/test_inverse_steganalysis_acquisition.py src/tac/tests/test_byte_shaving_campaign.py src/tac/tests/test_byte_shaving_campaign_queue.py -q`
  passed with `111 passed`.
- `.venv/bin/python -m pytest src/tac/tests/test_byte_range_entropy_recode_materializer.py src/tac/tests/test_materializer_chain_harvest_scheduler.py -q`
  passed with `38 passed`.
- `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/review_tracker.py policy-check src/tac/optimization/byte_shaving_campaign.py src/tac/tests/test_byte_shaving_campaign.py`
  reported `0 violations`.

## Next Gap

The compiler now recognizes the registry-wide deterministic operation surface.
The next frontier-relevant gap is to run the materializer campaign on real
inverse-action/MLX artifacts and emit queue-owned materializer contexts for the
byte-range and family-agnostic archive operations, then feed measured execution
outcomes back into the action-functional replan loop.
