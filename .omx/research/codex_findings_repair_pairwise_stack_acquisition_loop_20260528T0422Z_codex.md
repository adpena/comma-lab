# Codex Findings: Repair Pairwise Stack Acquisition Loop

Timestamp: 2026-05-28T04:22Z

## What Landed

- `repair_family_stack_search` now emits a pairwise interaction tensor over ordered family transitions, entropy-stage gaps, scope overlap, region/boundary coupling, pair/batch spillover, byte-credit pressure, negative-posterior pressure, and coupling synergy.
- Stack search now builds a primary acquisition path from that tensor and records a bounded terminal policy: strictly better archive-bound candidate blocked on exact axis, precise exact-axis blocker, or durable posterior demotion.
- The autonomous repair floor loop now surfaces the primary stack terminal outcome and stops on those bounded classes instead of a vague local-improvement condition.
- Family materializer manifests now preserve `allocated_repair_bytes`, `objective_delta_score_units`, and candidate-family identity for novel families such as `frame0_k16_palette_asymmetry` and `entropy_boundary_probe`.
- Byte-transform execution now falls back from empty MLX component deltas to the manifest objective delta, preserving local advisory acquisition signal without granting score authority.

## Evidence

- `.venv/bin/python -m ruff check src/tac/optimization/repair_family_byte_transform_executor.py src/tac/optimization/repair_family_materializers.py src/tac/optimization/repair_family_stack_search.py src/tac/tests/test_repair_family_materializers.py tools/run_repair_campaign_autonomous_floor_loop.py` passed.
- `.venv/bin/python -m pytest src/tac/tests/test_repair_family_materializers.py src/tac/tests/test_repair_campaign_materialization_queue.py -q` passed: 23 tests.
- Local MLX-advisory smoke: `.omx/research/repair_family_pairwise_autonomous_floor_loop_smoke3_20260528Tlocal/floor_loop_summary_all_harvested.json`
  - execution reports: 4
  - pairwise interaction tensor cells: 12
  - primary acquisition path: `posenet_null_bottom_decile -> entropy_boundary_probe`
  - local candidate improvement observed: true
  - exact handoff candidates: 4
  - posterior learning signals appended: 4
  - exact dispatch authority: false, blocked on contest CPU/CUDA axis
- Recursive adversarial review bundle `c991a7f473743e21` reached three successive clean passes with zero unresolved critical findings.

## Authority Boundary

All new rows remain MLX-local advisory or exact-handoff planning only. No score claim, rank/kill, promotion, budget spend, or exact dispatch authority is granted without contest CPU/CUDA custody and lane claim.
