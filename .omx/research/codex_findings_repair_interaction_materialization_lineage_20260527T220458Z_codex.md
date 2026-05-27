# Codex Findings: Repair Interaction Materialization Lineage

UTC: 2026-05-27T22:04:58Z
Commit: af638ece8 Add repair interaction materialization lineage

## What Landed

- Added `repair_campaign_cross_scale_interaction_dynamics.v1` as a typed schema on every multiscale repair action row.
- Added canonical interaction edges across bit, byte, pixel, boundary, region, frame, pair, batch, and full-video support.
- Added interaction state variables for `delta_segnet`, `delta_posenet`, `lambda_delta_bytes`, entropy-position weight, cross-scale terms, and receiver runtime constraints.
- Added interaction edge and dynamics-class histograms to the multiscale action ledger so queue consumers can rank and audit stackability patterns directly.
- Added `repair_campaign_materialization_lineage.v1` to scored rows and optimizer allocations. The lineage explicitly separates local MLX advisory readiness from byte-closed archive, archive-bound receiver proof, component replay manifest, and exact-axis response custody.
- Threaded materialization lineage and interaction dynamics into repair stackability queue metadata so local MLX probes preserve the mathematical object they are probing.

## Authority Boundary

- MLX-ready rows remain advisory only.
- The materialization lineage sets all score, promotion, budget-spend, and exact-dispatch authority fields false.
- Exact-axis handoff remains blocked until byte-closed candidate archive, receiver runtime proof, component replay manifest, and exact-axis component response artifacts are all present and revalidated.

## Verification

- `.venv/bin/python -m ruff check src/tac/optimization/repair_campaign_scorer.py src/comma_lab/scheduler/repair_campaign_stackability_queue.py src/tac/tests/test_repair_campaign_scorer.py src/tac/tests/test_repair_campaign_stackability_queue.py`
- `.venv/bin/python -m pytest src/tac/tests/test_repair_campaign_scorer.py src/tac/tests/test_repair_campaign_stackability_queue.py -q` -> 22 passed
- `.venv/bin/python -m pytest src/tac/tests/test_repair_campaign_*.py -q` -> 28 passed
- `.venv/bin/python -m pytest src/tac/tests/test_frontier_rate_attack_feedback.py -q` -> 57 passed
- `.venv/bin/python tools/lane_maturity.py validate` -> 1444 lanes validated cleanly
- `tools/review_gate_hook.py` passed on the staged repair slice
- `git diff --cached --check` passed before commit

## Remaining High-EV Work

1. Build the byte-closed repair materialization queue named by `target_queue_artifact_key=repair_campaign_byte_closed_materialization_queue`.
2. Make every selected repair allocation attempt archive materialization and archive-bound receiver proof by default, while preserving MLX-only rows as advisory/frozen.
3. Promote interaction edge histograms into the acquisition policy so negative stackability results update posterior priors by scale and entropy position.
4. Add executable materializer families for PoseNet-null bottom-decile, SegNet class-region waterfill, per-region selector codec, and palette/frame-asymmetry repair dynamics.
