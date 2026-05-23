# Codex Findings: Inverse Queue Guard Resolution And Materializer Routing

UTC: 2026-05-23T20:40:50Z
Agent: Codex
Lane: codex_non_dqs1_materializer_queue_guard_20260523

## Summary

Dirac's read-only adversarial report against
`src/tac/optimization/inverse_scorer_exact_eval_queue.py` was rechecked against
the current `main` tree. The reported failure classes are already covered by
the queue builder and tests landed in `8a89380d4`:

- forged or minimal parity payloads fail closed via parity payload reload and
  full-frame output-tree checks;
- runtime coverage is linked through runtime path and `inflate.sh` SHA checks;
- unresolved chain readiness blockers fail closed;
- parent traversal in archive paths fails closed;
- stale archive-member metadata is recomputed from ZIP member bytes before
  queue construction;
- chain steps copied into queue rows are sanitized instead of preserving nested
  authority booleans.

No additional queue-builder patch was needed for those findings.

## New Guard Landed

The byte-shaving DAG compiler could previously let an executable materializer
for a non-DQS1 target kind appear as an executable DQS1 portfolio row. That is a
false-action-surface bug: inverse action functional work is useful, but it is a
materializer work-queue action, not a byte-closed DQS1 candidate archive.

The scheduler now appends
`non_dqs1_target_requires_materializer_work_queue:<target_kind>` whenever a
resolved executable materializer has a target kind other than
`dqs1_pairset_drop`. That blocker classifies to
`materializer_work_queue_required`, keeps the row out of DQS1 operator actions,
and preserves the executable adapter row in
`byte_shaving_materializer_work_queue.v1`.

## Fixture Hardening

The inverse scorer exact-queue promotion fixture used a candidate archive that
was larger than its source archive while claiming full-frame inflate parity. The
fixture now uses a larger source payload and smaller candidate payload, so the
"promotes after readiness" test represents an actual byte-shaving candidate.

## Verification

- `.venv/bin/python -m pytest src/tac/tests/test_inverse_scorer_exact_eval_queue.py src/tac/tests/test_byte_shaving_campaign_queue.py::test_non_dqs1_executable_materializers_do_not_emit_dqs1_portfolio_rows src/tac/tests/test_byte_shaving_campaign_queue.py::test_inverse_action_cells_compile_to_candidate_materializer_work_queue -q`
- `.venv/bin/python -m pytest src/tac/tests/test_byte_shaving_campaign_queue.py -q`
- `.venv/bin/python -m ruff check src/comma_lab/scheduler/byte_shaving_campaign_queue.py src/tac/tests/test_byte_shaving_campaign_queue.py src/tac/optimization/inverse_scorer_exact_eval_queue.py src/tac/tests/test_inverse_scorer_exact_eval_queue.py`
- `.venv/bin/python tools/lane_maturity.py validate`
