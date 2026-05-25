# Codex Findings: Inverse-Action Context Autowire

UTC: 2026-05-25T08:36:13Z
Lane: `codex_inverse_action_context_autowire_20260525`
Commit status at write time: pending

## Finding

The receiver/compiler bridge landed in the prior tranche, but the normal
feedback-actuation queue still risked orphaning the signal: it compiled widened
inverse-action cells into materializer backlog/work-queue state without asking
the queue CLI to emit `byte_shaving_materializer_contexts.v1`. Separately,
several real producer paths dropped `operation_set_compiler` or explicit target
metadata before the action-functional stage, even though the downstream compiler
already knew how to consume it.

## Landing

This patch closes both sides of that gap:

- `tools/build_byte_shaving_campaign_queue.py` now treats
  `--materializer-contexts-out` as the context-generation trigger. If
  `--materializer-artifact-map` is omitted, it calls
  `build_final_byte_operation_contexts(..., artifact_map=None, ...)` so inline
  backlog/compiler params can produce executable contexts.
- `src/comma_lab/scheduler/queue_feedback_replan_policy.py` now wires
  `--materializer-contexts-out` and
  `--materializer-context-default-output-root` into the paused
  candidate-actuation queue, and records contexts in metadata, telemetry, and
  JSON completion postconditions.
- `src/tac/optimization/inverse_steganalysis_acquisition.py` now preserves
  compiler hints and explicit target metadata from inverse-scorer surfaces,
  byte-shaving operation sets, and byte-shaving ranked units into normalized
  atoms/action cells/provenance.

## Proof

Durable proof artifact:
`.omx/research/codex_inverse_action_context_autowire_20260525T083613Z/proof_summary.json`

Key assertions:

- `artifact_map_omitted=true`
- `contexts_generated=true`
- `contexts_unblocked=true`
- `work_queue_has_executable_row=true`
- `score_authority_false=true`

Generated proof state:

- `materializer_context_row_count=1`
- `materializer_context_blocked_count=0`
- `materializer_work_queue_executable_row_count=1`
- `materializer_work_queue_blocked_row_count=0`

## Verification

- `.venv/bin/python -m ruff check tools/build_byte_shaving_campaign_queue.py src/comma_lab/scheduler/queue_feedback_replan_policy.py src/tac/optimization/inverse_steganalysis_acquisition.py src/tac/tests/test_queue_feedback_replan_policy.py src/tac/tests/test_inverse_steganalysis_acquisition.py src/tac/tests/test_byte_shaving_campaign.py src/tac/tests/test_byte_shaving_campaign_queue.py`
- `.venv/bin/python -m pytest src/tac/tests/test_queue_feedback_replan_policy.py::test_feedback_candidate_actuation_planning_queue_compiles_widened_cells src/tac/tests/test_inverse_steganalysis_acquisition.py::test_inverse_scorer_surface_preserves_operation_compiler_metadata src/tac/tests/test_inverse_steganalysis_acquisition.py::test_byte_shaving_plan_producer_preserves_compiler_and_target_metadata src/tac/tests/test_byte_shaving_campaign.py::test_byte_shaving_ranked_unit_target_metadata_round_trips_to_packet_ir src/tac/tests/test_byte_shaving_campaign_queue.py::test_byte_shaving_campaign_queue_cli_generates_contexts_from_inline_compiler_hints`
- `.venv/bin/python -m pytest src/tac/tests/test_inverse_steganalysis_acquisition.py src/tac/tests/test_byte_shaving_campaign.py src/tac/tests/test_byte_shaving_campaign_queue.py src/tac/tests/test_queue_feedback_replan_policy.py`

Latest result: 178 passed.

## Authority Boundary

Generated contexts and work queues remain candidate-generation/planning surfaces
only. They do not grant score, promotion, rank/kill, paid-dispatch, or exact
eval authority. Legacy widened action artifacts that predate this patch and do
not contain compiler/target metadata should be replayed or regenerated rather
than guessed into executable operations.

