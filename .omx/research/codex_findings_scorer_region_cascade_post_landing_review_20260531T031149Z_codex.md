# Codex Findings: Scorer-Region Cascade Post-Landing Review

UTC: 2026-05-31T03:11:49Z

## Scope

Adversarial review of partner landing `d949c3f3a` (`Queue grouped
scorer-region cascade attack`) against the current `main` tree. Focus:
selected-survivor custody, false-authority semantics, MLX/local CPU split,
queue validity, and whether the first grouped P19/P18/P11/P15 cascade result
was being over-promoted.

## Verdict

The landing is structurally aligned with the score-lowering loop: it turns the
scorer-region cascade into a queue-owned grouped campaign, preserves
receiver-closed selected-survivor archive custody, and keeps MLX/local CPU rows
advisory until exact CPU/CUDA authority exists.

The first completed variant is a useful negative calibration anchor, not a
candidate promotion:

- MLX partial acquisition reported a strong local signal.
- The full local CPU gate was negative versus the current CPU frontier pointer.
- The exact-ready bridge records `local_cpu_eureka_trigger_false` and
  `local_cpu_score_not_below_auth_frontier`.
- The campaign report keeps `score_claim=false` and
  `ready_for_exact_eval_dispatch=false`.

## Fix Landed

One post-landing hygiene issue was found and fixed in `a6105a1a4`: the campaign
library imported `ScorerRegionSelectorChainQueueError` without using it. The
CLI owns that catch path and imports the symbol directly; the library import was
a ruff blocker.

## Verification

- `.venv/bin/python -m ruff check` on the scorer-region cascade queue, bridge,
  chain queue, waterfill helper, CLIs, and focused tests passed.
- `.venv/bin/python -m pytest src/tac/tests/test_scorer_region_selector_cascade_campaign_queue.py src/tac/tests/test_scorer_region_selector_chain_queue.py -q`
  passed: 14 tests.
- `.venv/bin/python tools/experiment_queue.py --queue .omx/research/scorer_region_selector_cascade_campaign_20260531T023734Z/queue.json validate`
  passed: 13 experiments, 193 steps.
- JSON authority scan over the staged/landed campaign artifacts found no truthy
  `score_claim`, `promotion_eligible`, `rank_or_kill_eligible`,
  `ready_for_exact_eval_dispatch`, `budget_spend_allowed`, or
  `ready_for_budget_spend` fields.
- `.venv/bin/python tools/lane_maturity.py validate` passed: 1557 lanes.
- `.venv/bin/python tools/review_gate_hook.py` passed.

## Follow-Up

Keep draining the grouped campaign, but treat the first local-CPU-negative row
as posterior demotion evidence for that low-amplitude RGB frame-1 operator set.
Next high-EV variants should add YUV-native receiver patches, larger grouped
region coverage, and CPU-calibrated MLX acquisition rather than raw MLX partial
score.
