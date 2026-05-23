# Codex Findings - Inverse Scorer Exact-Readiness Boundary

UTC: 2026-05-23T19:22:21Z
Lane: `lane_inverse_scorer_exact_readiness_boundary_20260523`

## Finding

IAS1 chain rows need a dedicated exact-readiness boundary. Receiver evidence and
full-frame inflate parity can make an archive a better exact-eval input, but
they must not erase the contest-auth boundary or become score/promotion
authority.

## Fix

`tac.optimizer.exact_readiness` now recognizes
`inverse_scorer_cell_candidate_chain_v1` rows and blocks exact-eval promotion
unless both conditions hold:

- strict full-frame inflate parity is satisfied by the chain parity step and
  backed by a non-symlink proof artifact whose SHA-256 matches the row;
- an explicit exact-auth boundary remains present via blockers or
  `next_required_gates`.

The guard also rejects IAS1 chain rows with truthy proxy-authority fields
defined by the canonical proxy-candidate contract, while allowing the local
non-authority blockers to clear only after the strict parity proof and exact
auth boundary both hold.

## Verification

- `src/tac/tests/test_optimizer_exact_readiness.py`
  `src/tac/tests/test_inverse_scorer_cell_materializer.py`: 61 passed.
- Integrated focused slice
  `src/tac/tests/test_inverse_scorer_cell_materializer.py`
  `src/tac/tests/test_artifact_retention.py`
  `src/tac/tests/test_optimizer_exact_readiness.py`
  `src/tac/tests/test_exact_dispatch_authority.py`
  `src/tac/tests/test_byte_shaving_campaign_queue.py`: 127 passed.
- Adversarial follow-up after exact-readiness bridge review:
  `src/tac/tests/test_optimizer_exact_readiness.py`: 34 passed.
- `git diff --check`: passed.

## Authority

This is dispatch-boundary hardening only. It does not claim score or authorize
contest eval by itself; it decides whether a candidate row has enough local
custody to be handed to the existing exact-eval dispatch machinery.
