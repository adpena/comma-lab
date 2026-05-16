# L5 v2 Literature-Scope Class-Shift Reward Suppression

Date: 2026-05-16
Author: Codex
Scope: Cathedral autopilot / L5 v2 rank reward hardening

## Verdict

`score_claim=false`; `promotion_eligible=false`; `ready_for_exact_eval_dispatch=false`.

Unscoped literature-anchor rows no longer keep a class-shift reward through
`lane_class`.

## Failure Class

`literature_source_scope_bypass_via_lane_class`

The prior source-scope guard suppressed `literature_anchor` and notes before
applying Z1 class-shift rank rewards, but it still passed `lane_class` through
the same reward function. A planning row with
`literature_anchor="Rao-Ballard predictive coding"` and
`lane_class="substrate_class_shift predictive_receiver"` could therefore miss
the required source-scope fields and still receive a class-shift bonus.

## Landed Fix

When `_candidate_literature_anchor_rank_reward_suppressed(...)` is true,
`apply_z1_empirical_revision_to_candidate_delta(...)` now suppresses
`lane_class` along with literature anchor text and notes before class-shift
reward adjustment.

Regression coverage:

- JSONL candidate ingestion: unscoped literature plus class-shift lane class
  stays at the base predicted delta; scoped rows retain both lane-class and
  literature rewards.
- Substrate-composition ranking ingestion: unscoped L5 predictive row with
  class-shift lane class gets source-scope blockers and no class-shift reward.
- Probe-disambiguator ingestion: unscoped row with class-shift lane class gets
  source-scope blockers and no class-shift reward.

## Verification

```bash
.venv/bin/python -m ruff check \
  tools/cathedral_autopilot_autonomous_loop.py \
  src/tac/tests/test_cathedral_autopilot_autonomous_loop.py \
  src/tac/tests/test_cathedral_autopilot_substrate_composition_wire.py

.venv/bin/python -m pytest \
  src/tac/tests/test_cathedral_autopilot_autonomous_loop.py \
  src/tac/tests/test_cathedral_autopilot_substrate_composition_wire.py -q
```

Observed:

- `ruff`: all checks passed
- `pytest`: `176 passed in 0.63s`
