<!-- SPDX-License-Identifier: MIT -->
---
schema: codex_findings_v1
topic: dqs1_drop_many_verdict_auto_discovery
created_at_utc: 2026-05-25T22:00:26Z
author: codex
lane_id: lane_codex_drop_many_greedy_feedback_bridge_20260525
score_claim: false
promotion_eligible: false
rank_or_kill_eligible: false
ready_for_exact_eval_dispatch: false
dispatch_attempted: false
research_only: false
---

# DQS1 Drop-Many Verdict Auto-Discovery

## Finding

The cross-family portfolio already knew how to consume
`dqs1_drop_many_build_1c_greedy_independent_heuristic_verdict.v1`, but the
frontier feedback cycle only picked it up if a caller manually threaded the
verdict path through a separate CLI. In the latest component-marginal refresh,
that left the drop-many verdict model inactive even though a safe verdict
artifact existed.

## Landing

`frontier_rate_attack_feedback_cycle` now discovers safe DQS1 drop-many greedy
verdicts from canonical repo roots (`.omx/research` and `experiments/results`)
when building pairset component-marginal feedback bundles.

The bundle and action summary now preserve:

- discovered verdict paths;
- discovery/refusal metadata;
- the compiled `drop_many_greedy_verdict_model`.

That lets the queue-owned component-marginal refresh apply the existing policy:
independent greedy drop-many rows are held when the verdict says the inherited
score surface collapsed to K=1, while interaction-aware/component-model
successors remain eligible planning rows.

## Authority Boundary

Discovery requires the verdict schema and false-authority fields. The compiled
model remains planning-only and cannot claim score, promotion, rank/kill, or
exact-eval dispatch readiness.

Discovery is bounded to direct child verdict directories matching
`dqs1_drop_many_build_1c_greedy_heuristic*/verdict.json`. It does not recursively
scan arbitrary `experiments/results` descendants, preserving the no-disk-pressure
and no-slow-hidden-walk contract for large local result trees.

## Verification

The regression builds a component-marginal bundle with an auto-discovered
drop-many verdict and proves that an independent
`drop_many_beam_pairwise_interaction_waterfill` candidate is rewritten to
`hold_independent_drop_many_until_interaction_or_component_model` with the
expected planning blocker.

A second regression proves unrelated deep verdict-shaped files are ignored by
the bounded discovery pattern.
