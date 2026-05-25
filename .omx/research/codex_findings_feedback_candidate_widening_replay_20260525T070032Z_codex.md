# Feedback Candidate Widening Replay - Codex Findings - 2026-05-25T07:00:32Z

## Summary

Codex replayed the dry feedback action-functional path from:

- `.omx/research/high_level_byte_shaving_runner_smoke_20260524T050723Z/campaign3_ias1aware_repair_feedback/followup_materializer_queue_exec2/materializer_campaign_run.json`

The original dry feedback action functional had `cell_count=1`, `selected_count=0`,
and `materializer_archive_delta_blocked_cell_count=1`. The replay policy now
refuses fake plan-only widening unless a widenable scorer/signal source is
present or rediscovered. For this run, it rediscovered:

- `.omx/research/high_level_byte_shaving_runner_smoke_20260524T050723Z/scorer_response.json`

The emitted widening queue executed locally through `tools/experiment_queue.py`
with `timeout_seconds=0` and no cloud/score authority.

The hardening added in this pass makes rediscovered scorer-response sources
custody-bearing rather than schema-only: the policy records raw SHA-256, byte
size, row count, producer, and nearest run/plan seed directory. If multiple
usable nearby scorer-response datasets are discovered, the policy refuses the
widening handoff as ambiguous instead of appending all sources.

## Durable Artifacts

- Replay follow-up queue:
  `.omx/research/high_level_byte_shaving_runner_smoke_20260524T050723Z/campaign3_ias1aware_repair_feedback/followup_materializer_queue_exec2/queue_feedback_replan_followup_queue.replay.json`
- Replayed policy:
  `.omx/research/high_level_byte_shaving_runner_smoke_20260524T050723Z/campaign3_ias1aware_repair_feedback/followup_materializer_queue_exec2/queue_feedback_replan_policy.widening_replay.json`
- Candidate widening queue:
  `.omx/research/high_level_byte_shaving_runner_smoke_20260524T050723Z/campaign3_ias1aware_repair_feedback/followup_materializer_queue_exec2/queue_feedback_candidate_widening_queue.json`
- Widened action functional:
  `.omx/research/high_level_byte_shaving_runner_smoke_20260524T050723Z/campaign3_ias1aware_repair_feedback/followup_materializer_queue_exec2/inverse_steganalysis_action_functional.feedback.widened.json`
- Widened Markdown summary:
  `.omx/research/high_level_byte_shaving_runner_smoke_20260524T050723Z/campaign3_ias1aware_repair_feedback/followup_materializer_queue_exec2/inverse_steganalysis_action_functional.feedback.widened.md`
- Queue performance:
  `.omx/research/high_level_byte_shaving_runner_smoke_20260524T050723Z/campaign3_ias1aware_repair_feedback/followup_materializer_queue_exec2/queue_feedback_candidate_widening_performance.json`
- Final paused observation:
  `.omx/research/high_level_byte_shaving_runner_smoke_20260524T050723Z/campaign3_ias1aware_repair_feedback/followup_materializer_queue_exec2/queue_feedback_candidate_widening_observation.paused.json`

## Result

- Policy decision: `widen_inverse_candidate_generation`
- Widening source mode: `discovered_nearby_scorer_response`
- Rediscovered source custody:
  `path=.omx/research/high_level_byte_shaving_runner_smoke_20260524T050723Z/scorer_response.json`,
  `sha256=8004f6da270932828f5d8d4141eb181ed5e823cd49251b39cbf800f6d1edae60`,
  `bytes=2166`, `row_count=2`,
  `producer=codex_high_level_byte_shaving_runner_smoke`
- Widened command added `--scorer-response .omx/research/high_level_byte_shaving_runner_smoke_20260524T050723Z/scorer_response.json`
- Queue execution: 1 local CPU step, 0 failures, 0.264224666 seconds elapsed
- Candidate widening queue postconditions require widened JSON existence,
  widened Markdown existence, action-functional schema, `planning_only=true`,
  `candidate_generation_only=true`, and false score/promotion/rank-kill fields
- Final queue health: healthy, paused, 0 blockers, 0 orphaned steps
- Widened action functional: `cell_count=3`, `selected_count=3`,
  `selected_water_fill_cost_bytes=41`, `selected_expected_score_gain=0.0001606`
- Authority fields remained false:
  `score_claim=false`, `promotion_eligible=false`,
  `rank_or_kill_eligible=false`, `ready_for_exact_eval_dispatch=false`

## Interpretation

This closed the fake-widening gap for plan-only dry feedback. The dry result was
not a terminal "no opportunity" signal; it was missing a live scorer-response
source. Once rediscovered, the action surface selected 3 planning cells from the
combined campaign plan plus scorer-response evidence.

The next frontier-relevant step is to route these selected planning cells into a
real automated final-rate materializer/actuator and then exact-auth calibration.
This replay is not a score claim and not dispatch authority.

Runtime SQLite state was used to produce the paused observation/performance
artifacts. Per current repository convention, the durable committed signal is
the JSON/Markdown artifact set above rather than the transient SQLite file.
