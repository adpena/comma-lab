# L5-v2 Cathedral Validation Queue Wire-In - 2026-05-16

## Context

The L5-v2 staircase exposes PR106 PacketIR stack evidence through
`src/tac/optimization/l5_staircase_v2.py`, but Cathedral autopilot did not
consume that surface. This violated the mandatory wire-in discipline: L5-v2
could be visible in operator briefing while remaining absent from Cathedral's
validation queue.

## Fix

- Added a Cathedral validation-queue adapter for L5-v2 / PR106 PacketIR stack
  state in `tools/cathedral_autopilot.py`.
- The adapter preserves:
  - `lane_id`, `campaign_id`, and `subject_id`;
  - CPU/CUDA axis semantics;
  - PR106 PacketIR matrix status counts;
  - dispatch blockers from L5-v2 gates, PacketIR stack evidence, stack-cell
    candidate generation, and matrix loading;
  - exact-eval target templates when present.
- The adapter is explicitly non-promotional:
  - `score_claim=false`;
  - `promotion_eligible=false`;
  - `rank_or_kill_eligible=false`;
  - `ready_for_exact_eval_dispatch=false`;
  - `potential_score_delta_if_validated=0.0` until a composite archive and
    paired exact eval exist.

## Current State

The current PR106 PacketIR matrix has 16 rows, all blocked as
`runtime_consumption_blocked`. Cathedral now surfaces this as
`blocked_until_runtime_bound_packetir_evidence` instead of silently omitting
the L5-v2 stack path.

## Verification

```bash
.venv/bin/python -m ruff check tools/cathedral_autopilot.py \
  src/tac/tests/test_cathedral_autopilot.py

PYTHONPATH=src:. .venv/bin/pytest src/tac/tests/test_cathedral_autopilot.py -q

.venv/bin/python tools/cathedral_autopilot.py plan \
  --d-seg 0.00067082 \
  --d-pose 0.0000336 \
  --archive-bytes 185578 \
  --target-score 0.190 \
  --label l5_v2_cathedral_smoke
```

Observed: `35 passed`, ruff clean, and the smoke output contains an L5-v2
validation-queue row with `runtime_consumption_blocked: 16` and all authority
flags false.

## Next

Regenerate direct runtime-consumption manifests for the current
`submissions/pr106_latent_sidecar_r2_pr101_grammar` runtime. Once PacketIR rows
recover runtime-bound paired evidence, the same Cathedral adapter will surface
individual PR106 stack-cell candidates while still requiring composite archive
materialization, sideinfo consumption proof, and paired CPU/CUDA exact eval.
