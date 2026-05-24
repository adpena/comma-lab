# Codex Findings: Direct MLX Spend-Triage Compiler Hint Preservation

- Timestamp: 2026-05-24T20:07:41Z
- Lane: `codex_mlx_direct_spend_triage_compiler_hint_20260524`
- Scope: inverse-steganalysis direct MLX spend-triage rows to action functional, PacketIR, and materializer queue lowering.

## Finding

The grouped MLX acquisition batch path already preserved explicit
`operation_set_compiler` hints, but the direct strict MLX effective-spend
triage path dropped the same row-level hint before it reached the action
functional. That created an avoidable orphaned-signal path: a direct MLX row
could pass strict local/proxy gates and still lose the deterministic compiler
intent needed for PacketIR and queue-owned materializers.

Generic scorer-response rows still must not synthesize materializer hints; that
would fabricate archive-operation semantics from advisory signal. This patch
only preserves an explicit operator/model-provided compiler hint already present
on a strict MLX spend-triage row.

## Landed

- `src/tac/optimization/inverse_steganalysis_acquisition.py` now carries
  direct MLX spend-triage row `operation_set_compiler` hints into both the
  atom and false-authority provenance.
- `src/tac/tests/test_inverse_steganalysis_acquisition.py` verifies direct
  MLX hint survival to the action cell and nested truthy-authority rejection.
- `src/tac/tests/test_byte_shaving_campaign.py` verifies direct MLX selection
  to action functional to compiled PacketIR lowering.
- `src/tac/tests/test_byte_shaving_campaign_queue.py` verifies the resulting
  PacketIR reaches materializer backlog/work-queue rows while remaining blocked
  for materializer contexts and false-authority for score/promotion/dispatch.

## Authority Boundary

All emitted surfaces remain candidate-generation/planning signal only:
`score_claim=false`, `score_claim_valid=false`, `promotion_eligible=false`,
`rank_or_kill_eligible=false`, and `ready_for_exact_eval_dispatch=false`.

## Verification

- `.venv/bin/python -m pytest src/tac/tests/test_inverse_steganalysis_acquisition.py src/tac/tests/test_byte_shaving_campaign.py src/tac/tests/test_byte_shaving_campaign_queue.py -q`
  - `109 passed`
- `.venv/bin/python -m ruff check src/tac/optimization/inverse_steganalysis_acquisition.py src/tac/tests/test_inverse_steganalysis_acquisition.py src/tac/tests/test_byte_shaving_campaign.py src/tac/tests/test_byte_shaving_campaign_queue.py`
  - `All checks passed!`

## Next

- Feed real queue performance summaries back through the emitted replan request
  artifacts to continuously update the action functional denominator terms.
- Keep materializer synthesis explicit: MLX/scorer-response observations can
  rank and calibrate candidate work, but only compiler hints or byte-closed
  archive semantics should produce executable PacketIR/materializer targets.
