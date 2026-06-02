# Codex Findings: NeRV Modelsize Budget Receiver Split

UTC: 2026-06-02T12:53:33Z

## Landed

- Hardened `modelsize_budget_plan` to schema
  `compact_carrier_modelsize_budget_plan.v2`.
- Split modelsize ladder rows into receiver-closed measured bytes, advisory
  measured bytes without receiver proof, and projected/lower-bound bytes.
- Reserved `receiver_closed_modelsize_budget_selected` for ladders with at
  least two receiver-proofed archive-byte rows.
- Routed projected/advisory ladders to
  `advisory_or_projected_modelsize_budget_selected` with explicit blockers:
  `receiver_closed_modelsize_ladder_has_fewer_than_two_points`,
  `modelsize_budget_selection_is_advisory_or_projected`,
  `projected_or_lower_bound_archive_bytes_not_receiver_closed`, and
  `receiver_closed_byte_proof_missing`.
- Added `receiver_closed_selected_archive_bytes`,
  `receiver_closed_points`, `point_count_by_evidence`, and `decision_basis`
  so final-rate, bit allocator, Cathedral, and carrier-training consumers can
  tell planning rows apart from receiver-closed byte rows without heuristics.
- Updated `carrier_training_plan` to consume the v2 schema and expose the
  receiver-closed selected byte target in `evidence_summary`.

## Artifact

- `.omx/research/nerv_modelsize_budget_receiver_split_proof_20260602T125333Z.json`
  - SHA-256: `78fd6c6d6983ba7e097d71b962c5ae84232809b27b7dc5f191ed527c65096d1a`
  - schema: `codex_nerv_modelsize_budget_receiver_split_proof.v1`
  - receiver-closed plan status: `receiver_closed_modelsize_budget_selected`
  - receiver-closed decision basis: `receiver_closed_rows`
  - receiver-closed selected archive bytes: `72,000`
  - receiver-closed measured points: `3`
  - projected/advisory plan status:
    `advisory_or_projected_modelsize_budget_selected`
  - projected/advisory all points: `3`
  - projected/advisory measured points: `0`
  - projected/advisory receiver-closed selection: `null`

## Verification

- `PYTHONPATH=src .venv/bin/python ...` generated the proof artifact from the
  patched SSD worktree source.
- Initial ad-hoc generation without `PYTHONPATH=src` imported the dirty shared
  checkout at `/Users/adpena/Projects/pact/src/tac/__init__.py`; the artifact
  was corrected by overwrite-with-expected-SHA before landing.
- `.venv/bin/pytest src/tac/substrates/_shared/mlx_score_aware/tests/test_modelsize_budget_plan.py src/tac/substrates/_shared/mlx_score_aware/tests/test_carrier_training_plan.py -q`
  passed: 10 tests.
- `.venv/bin/pytest src/tac/tests/test_nerv_control_surfaces.py src/tac/tests/test_nerv_top_priority_stack_seam.py src/tac/cathedral_consumers/pareto_carrier_fit_consumer/tests/test_pareto_carrier_fit_consumer.py -q`
  passed: 49 tests.
- `.venv/bin/ruff check ...` passed on the touched files.

## Verdict

GO for local modelsize/rate planning with explicit evidence class separation.
NO-GO for score claims, rank/kill, promotion, exact/full-video/CUDA dispatch,
or PR95 beat claims. Source-formula modelsize curves and advisory byte rows
still require trained receiver archive bytes before they can become production
evidence.

## Next Work

SNeRV and HiNeRV should consume `receiver_closed_selected_archive_bytes` only
when present. If it is `null`, the next action is a receiver-closed
modelsize/fc_dim ladder, not a full training launch or exact dispatch.
