# Codex Findings: Frontier Feedback Campaign Autowave

- timestamp_utc: 2026-05-25T22:38:19Z
- lane: frontier_rate_attack_feedback_cycle_autowave
- authority: false_authority_planning_and_local_advisory_only
- score_claim: false
- promotion_eligible: false
- rank_or_kill_eligible: false
- ready_for_exact_eval_dispatch: false

## What Changed

The final-rate attack path was still too manual: a bounded cycle could build and
execute the first DQS1 queue, but completed candidates in a partially active
batch could be stranded until a manual harvest, and the next queue still needed
operator stitching.

This pass makes `tools/run_frontier_rate_attack_feedback_cycle.py` the
operator-facing campaign lane. It now supports `--campaign-waves N`: each wave
executes the current queue, harvests completed local-first results, canonicalizes
observation JSONL rows, refreshes the component-marginal action model, emits a
validated next DQS1 queue, and records the wave under `campaign_execution`.

The storage-heavy `tools/run_dqs1_local_first_tranche.py` remains a specialized
legacy/tranche surface, but it now consumes local CPU eureka planning in its
pairset acquisition refresh so the same beyond-drop-two/drop-many signal is not
orphaned there.

## Empirical Local Cycle

Executed bounded local CPU advisory cycle on external storage:

- output: `.omx/research/codex_frontier_rate_attack_feedback_cycle_20260525T221326Z/frontier_rate_attack_feedback_cycle.json`
- results_root: `/Volumes/VertigoDataTier/pact_experiments/frontier_rate_attack_feedback_cycle_20260525T221326Z`
- completed candidates:
  - `pairset_drop_one_rank008_pair0496`
  - `pairset_drop_one_rank007_pair0059`
- both local macOS CPU advisory scores: `0.19204028295713674`
- current contest-CPU frontier anchor remained better: `0.19202828295713675`
- action: observe_only; no exact eval request; no dispatch; no promotion.

The completed candidates were manually recovered into:

- `.omx/research/codex_frontier_rate_attack_feedback_cycle_20260525T221326Z/manual_harvest/`

Then the recovered harvests were canonicalized into:

- `.omx/research/codex_frontier_rate_attack_feedback_cycle_harvest_closure_20260525T223819Z/dqs1_harvest_observations/dqs1_local_first_harvest_observations_20260525T223819Z.jsonl`

The post-harvest queue selected geometry-aware candidates:

- `pairset_geometry_lowimpact_k003_h9dfec80f80`
- `pairset_geometry_lowimpact_k004_h3d2fc811a7`
- `pairset_geometry_lowimpact_k006_h7ce3073a1a`
- `pairset_geometry_lowimpact_k008_hc4a555f7ab`

## Bug Classes Fixed

1. External results-root preflight no longer crashes when the requested SSD
   workload directory does not exist yet; free-space probing walks to the
   nearest existing parent.
2. Batch autopilot now harvests fully succeeded candidates even when later
   candidates remain queued/running, preventing signal loss at `max_total_steps`.
3. `tools/harvest_dqs1_local_first_result.py` accepts `--candidate-id`, so a
   specific completed candidate can be recovered from a multi-candidate queue
   without rerunning expensive local CPU advisory work.
4. Operator briefing now treats nested authority leaks inside frontier feedback
   cycle reports as blocking, and recognizes `CAMPAIGN_QUEUE_READY` reports from
   multi-wave campaign output.

## Tests

- `.venv/bin/python -m ruff check tools/run_frontier_rate_attack_feedback_cycle.py tools/run_dqs1_local_first_tranche.py tools/operator_briefing.py src/tac/tests/test_dqs1_local_first_tranche.py src/tac/tests/test_operator_briefing.py`
- `.venv/bin/python -m pytest src/tac/tests/test_frontier_rate_attack_feedback.py src/tac/tests/test_dqs1_local_first_autopilot.py src/tac/tests/test_dqs1_local_first_queue_builder.py::test_dqs1_harvest_selects_candidate_from_multi_candidate_batch src/tac/tests/test_dqs1_local_first_tranche.py src/tac/tests/test_operator_briefing.py::test_operator_briefing_surfaces_frontier_feedback_cycle_autopolicy src/tac/tests/test_operator_briefing.py::test_operator_briefing_blocks_nested_frontier_feedback_cycle_authority_leak src/tac/tests/test_operator_briefing.py::test_operator_briefing_surfaces_campaign_wave_frontier_feedback_queue -q`

## Next Integration Hooks

The next highest-leverage hook is cycle-level ingestion of
`byte_shaving_materializer_campaign_run.v1` outputs: receiver feedback,
queue-feedback replan policy, candidate widening queues, and actuation queues
should be surfaced directly in the frontier cycle report as local paused
follow-up queues. That keeps byte-shaving materializer campaigns and DQS1
frontier feedback under one operator-visible campaign lane without making Dask
or tranche scripts the source of authority.
