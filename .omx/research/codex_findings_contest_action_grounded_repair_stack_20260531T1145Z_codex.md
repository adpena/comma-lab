# Codex Findings: Contest-Action-Grounded Repair Stack

UTC: 2026-05-31T11:45Z

## Finding

The repair stack search already consumed interaction tensors, byte-credit
pressure, posterior demotion, and archive-bound custody, but the row acquisition
score still treated the official contest rate term mostly as a penalty around a
local SegNet/PoseNet delta. That left a mathematical grounding gap: a repair row
could look good per byte before the official `S = 100*d_seg + sqrt(10*d_pose) +
25*archive_bytes/37545489` objective had been hydrated for that row.

## Change

- Wired `tac.optimization.contest_space_action` into
  `repair_family_stack_search`.
- Every stack row now carries a `contest_space_action_row` with:
  - raw local MLX distortion delta preserved;
  - official per-byte rate cost/credit applied;
  - rate-adjusted net delta and expected improvement exposed;
  - hydration explicitly tagged `[macOS-MLX research-signal]` and fail-closed.
- The aggregate stack plan now emits one
  `contest_space_action_functional.v1`, so acquisition consumers can read one
  contest objective surface instead of re-deriving score math.
- The existing floor-loop regression now asserts the official rate coefficient
  changes repair-row net delta.

## Live Replay

Command:

```bash
.venv/bin/python tools/run_repair_campaign_autonomous_floor_loop.py \
  --materialization-queue .omx/research/repair_multi_archive_autonomous_stage_disciplined_psv3_fec6_20260528T0648Z/repair_materialization_queue.json \
  --contract-migration-backlog .omx/research/archive_bound_contract_migration_backlog_queue_20260531T1038Z.json \
  --output-dir .omx/research/repair_floor_loop_contest_action_grounding_20260531T1145Z \
  --summary-out .omx/research/repair_floor_loop_contest_action_grounding_20260531T1145Z/floor_loop_summary.json \
  --max-iterations 1 \
  --overwrite
```

Result:

- `contest_space_action_row_count`: 10
- `contest_space_local_gate_passed_count`: 10
- `contest_space_best_observed_net_delta_score_units`: -0.0030714468157165414
- `rate_score_per_byte`: 6.658589531221714e-7
- `primary_stack_acquisition_terminal_outcome`:
  `strictly_better_archive_bound_candidate_exact_axis_blocked`
- `score_claim=false`, `ready_for_exact_eval_dispatch=false`

The replay produced byte-closed/local advisory artifacts and exact-ready bridge
inputs, but it still correctly blocks on contest CPU/CUDA authority and runtime
content-tree custody before any score claim.

## Verification

- `.venv/bin/ruff check src/tac/optimization/repair_family_stack_search.py src/tac/tests/test_repair_campaign_materialization_queue.py`
- `.venv/bin/python -m pytest src/tac/tests/test_repair_campaign_materialization_queue.py -q`

Both passed.
