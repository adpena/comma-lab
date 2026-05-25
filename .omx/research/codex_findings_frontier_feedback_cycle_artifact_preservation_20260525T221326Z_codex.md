<!-- SPDX-License-Identifier: MIT -->
---
schema: codex_findings_v1
topic: frontier_feedback_cycle_artifact_preservation
created_at_utc: 2026-05-25T22:13:26Z
author: codex
lane_id: lane_codex_drop_many_greedy_feedback_bridge_20260525
score_claim: false
promotion_eligible: false
rank_or_kill_eligible: false
ready_for_exact_eval_dispatch: false
dispatch_attempted: false
research_only: false
---

# Frontier Feedback Cycle Artifact Preservation

## Finding

The queue-owned feedback cycle now consumes the DQS1 drop-many greedy verdict
automatically, but fresh generated refresh artifacts can still be lost if they
remain untracked scratch. The 2026-05-25T22:12Z and 2026-05-25T22:13Z refreshes
both proved that `drop_many_greedy_verdict_discovery.active=true` and that the
negative independent-greedy verdict flows into the action summary.

## Landing

This commit preserves the two refresh artifact bundles under:

- `.omx/research/codex_frontier_rate_attack_feedback_cycle_20260525T221223Z/`
- `.omx/research/codex_frontier_rate_attack_feedback_cycle_20260525T221326Z/`

The latest validated queue has 4 experiments and 28 steps. It carries 20 DQS1
observations, one discovered drop-many greedy verdict, and active preferred
successors:

- `learned_component_marginal_combo`
- `pair_frame_geometry_low_impact_drop_many`
- `inverse_scorer_null_direction_masked_variant`

## Autopilot Hardening

The local-first autopilot disk-space probe now walks up to the nearest existing
ancestor before calling `shutil.disk_usage`. That keeps queue refreshes usable
when the requested external result root has not been created yet, without
weakening the retention/custody guard.

## Authority Boundary

All preserved artifacts remain planning-only. They do not claim score,
promotion, rank/kill, or exact-eval dispatch readiness, and they do not replace
contest CPU/CUDA authority.

## Verification

- `tools/experiment_queue.py --queue .../20260525T221223Z/initial_refresh/dqs1_followup_queue.json validate`: valid, 4 experiments, 28 steps
- `tools/experiment_queue.py --queue .../20260525T221326Z/initial_refresh/dqs1_followup_queue.json validate`: valid, 4 experiments, 28 steps
- artifact summary: drop-many discovery active, discovered verdict count 1,
  verdict `NEGATIVE_COLLAPSE_TO_K1_EMPIRICAL_DROP_MANY_REGRESSES`
- `pytest src/tac/tests/test_dqs1_local_first_autopilot.py -q`: 3 passed
- `ruff` on touched autopilot files: clean
- focused review policy checks: clean
