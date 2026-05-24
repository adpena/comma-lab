# Codex Findings - Inverse Action Portfolio Water-Fill And Explicit-Target Compiler Fallback

UTC: 2026-05-24T21:54:14Z

## Scope

Codex replaced the inverse-steganalysis water-bucket selector's scalar greedy
choice with a bounded portfolio search over selectable action cells. Codex also
added a narrow fallback compiler that can synthesize operation-set compiler
provenance only when an inverse-action cell explicitly names one of the
registered executable family targets. These are planning/control-plane
improvements only: the artifacts remain false-authority until byte-closed
candidates and exact auth eval exist.

## Why This Matters

The previous `_water_bucket_fill(...)` implementation sorted cells by local
Euler-Lagrange residual and greedily accepted the next cell that fit the byte
budget. That could leave better combinations unselected when one high-utility
cell consumed budget that two slightly lower-utility cells could use for higher
total Lagrangian gain.

The new implementation keeps the greedy baseline for observability, but selects
the final bucket using bounded Lagrangian portfolio search:

- objective per cell: `expected_score_gain - lambda_rate * water_fill_cost_bytes`;
- budget constraint: `total_byte_budget`;
- only positive-residual/selectable cells are candidates;
- frontier pruning keeps the search bounded and deterministic;
- output records selected gain/cost plus the greedy baseline it beat.

The compiler fallback removes a different local minimum: when a cell already
names a concrete executable family target such as
`archive_section_entropy_recode_v1`, `packet_member_recompress_v1`, or
`tensor_factorize_v1`, it no longer falls all the way back to
`high_level_operation_compiler_required` solely because the full
`operation_set_compiler` object is missing. It still refuses vague cells and
unsupported targets.

## Landed Behavior

- `src/tac/optimization/inverse_steganalysis_acquisition.py`
  - Added `_water_bucket_portfolio_search(...)`.
  - Added `_water_bucket_pruned_frontier(...)`.
  - Added `_water_bucket_state_sort_key(...)`.
  - Added `_water_bucket_selection_row(...)`.
  - Extended water-bucket JSON with `selection_strategy`,
    `candidate_pool_count`, `frontier_state_count`,
    `selected_lagrangian_gain`, and greedy-baseline fields.
  - Preserved explicit operation-set target metadata from atoms into action
    cells so downstream planning can compile only explicitly named targets.
- `src/tac/optimization/byte_shaving_campaign.py`
  - Added an implicit compiler mapping that only activates for explicit
    target/family metadata resolving to one of the registered executable
    family targets.
  - Reused the existing compiler provenance path so PacketIR, materializer
    context, runtime-proof, and exact-auth blockers remain intact.
- `src/tac/tests/test_inverse_steganalysis_acquisition.py`
  - Added a regression where scalar greedy picks one high-utility cell but the
    bounded portfolio selects two cells with higher total expected and
    Lagrangian gain.
  - Added coverage that explicit target metadata survives action-functional
    construction.
- `src/tac/tests/test_byte_shaving_campaign.py`
  - Added coverage that explicit high-level target metadata lowers to PacketIR
    for archive-section recode, packet-member recompress, and tensor factorize
    without a full explicit `operation_set_compiler` object.

## Verification

- `.venv/bin/python -m ruff check src/tac/optimization/inverse_steganalysis_acquisition.py src/tac/tests/test_inverse_steganalysis_acquisition.py`
- `.venv/bin/python -m ruff check src/tac/optimization/inverse_steganalysis_acquisition.py src/tac/tests/test_inverse_steganalysis_acquisition.py src/tac/optimization/byte_shaving_campaign.py src/tac/tests/test_byte_shaving_campaign.py`
- `.venv/bin/python -m pytest src/tac/tests/test_inverse_steganalysis_acquisition.py::test_water_bucket_uses_portfolio_search_not_scalar_greedy -q`
  - `1 passed`
- `.venv/bin/python -m pytest src/tac/tests/test_inverse_steganalysis_acquisition.py::test_action_functional_preserves_explicit_target_metadata_for_compiler src/tac/tests/test_inverse_steganalysis_acquisition.py::test_water_bucket_uses_portfolio_search_not_scalar_greedy src/tac/tests/test_byte_shaving_campaign.py::test_inverse_action_explicit_target_metadata_synthesizes_compiler_hint -q`
  - `5 passed`
- `.venv/bin/python -m pytest src/tac/tests/test_inverse_steganalysis_acquisition.py src/tac/tests/test_byte_shaving_campaign.py -q`
  - `55 passed`
- `.venv/bin/python -m pytest src/tac/tests/test_byte_shaving_campaign_queue.py src/tac/tests/test_staircase_dag.py src/tac/tests/test_byte_shaving_materializer_campaign_runner.py -q`
  - `126 passed`

## Authority Boundary

The selected bucket is still a false-authority action-surface plan. The search
chooses which action cells should be materialized next; it does not claim score,
promotion eligibility, rank/kill eligibility, exact-eval readiness, or dispatch
authority. The selected byte cost remains planner budget cost, not serialized
byte savings.

The compiler fallback only reduces blocker class from
`requires_candidate_family_operation_compiler` to the existing
materializer-context/runtime-proof/exact-auth blockers. It does not make the
high-level materializer executable and does not infer targets from vague
advisory scorer signal.

## Remaining Gaps

- The portfolio search optimizes per-cell Lagrangian gain after the current
  action functional has already folded first/second-order terms into each cell.
  It does not yet model explicit pairwise selection kernels across cells.
- Materializer/context breadth is still the next bottleneck after selection:
  action cells that lack compiler hints or source provenance still require
  compiler/materializer work before they become byte-closed candidates.
- MLX/Metal/Accelerate candidate-training lanes still need to emit stronger
  compiler hints or candidate packets for this selector to exploit at scale.
