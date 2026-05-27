# Codex Findings: Repair Byte-Closed Materialization Queue

UTC: 2026-05-27T22:16:39Z
Commit: 2104f1557 Wire repair byte-closed materialization queue

## What Landed

- Added `repair_campaign_byte_closed_materialization_queue` as the executable queue artifact targeted by repair materialization lineage.
- Added `tools/build_repair_campaign_byte_closed_materialization_queue.py`.
- Wired the repair campaign score queue to build, validate, and run this materialization queue by default after scoring.
- Each selected repair allocation now gets a queue-owned byte-closed materialization experiment with these default stages:
  1. emit repair-budget materialization plan,
  2. emit child component replay manifests,
  3. bind materializer execution evidence,
  4. audit materialization execution,
  5. emit selected-allocation materialization gate.
- MLX custody remains local advisory only.
- Exact-eval handoff remains false unless byte-closed archive, archive-bound runtime proof, receiver consumption, and component replay custody are complete.

## Verification

- `.venv/bin/python -m ruff check ...` on the new queue, score-queue wiring, CLI, and tests.
- `.venv/bin/python -m pytest src/tac/tests/test_repair_campaign_materialization_queue.py src/tac/tests/test_repair_campaign_score_queue.py -q` -> 8 passed.
- `.venv/bin/python -m pytest src/tac/tests/test_repair_campaign_materialization_queue.py src/tac/tests/test_repair_campaign_score_queue.py src/tac/tests/test_repair_campaign_scorer.py src/tac/tests/test_repair_campaign_stackability_queue.py -q` -> 31 passed.
- `.venv/bin/python -m pytest src/tac/tests/test_frontier_rate_attack_feedback.py -q` -> 57 passed.
- `.venv/bin/python tools/lane_maturity.py validate` -> 1444 lanes validated cleanly.
- `tools/review_gate_hook.py` passed on the staged repair materialization slice.

## Remaining High-EV Work

1. Replace readiness-only child repair rows with actual family-specific materializers for SegNet region waterfill, PoseNet-null bottom-decile, selector codec variants, and palette/frame-asymmetry repair.
2. Feed materialization gate blockers back into posterior acquisition so failed archive/proof/component replay attempts alter future queue routing automatically.
3. Promote successful materialization gates into exact-readiness consumers only after archive SHA, runtime proof, receiver consumption, and exact-axis component-response artifacts are all present.
4. Add per-family stackability interaction penalties keyed by interaction edge histogram and entropy-position class.
