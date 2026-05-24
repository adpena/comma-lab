# Codex Findings: PR95/HNeRV MLX Queue Plan Bridge

UTC: 2026-05-24T08:19:14Z

## Landing

- Added `--plan-only` to `tools/run_pr95_mlx_timing_smoke.py`.
- The PR95/HNeRV MLX timing-smoke runner now writes:
  - `plan.json` with `schema=representation_training_probe_plan_v1`;
  - `representation_training_plan.json` with the canonical representation-training plan sidecar;
  - `plan_summary.json` for operator/queue inspection.
- The emitted `recommended_execution` is directly consumable by
  `comma_lab.scheduler.local_training_queue.build_local_training_execution_queue`.
- `stage_count` is intentionally `1` for these timing rows; the PR95 curriculum
  identity lives in `stage_index` / `stage_module` so Stage 8 is not
  misinterpreted as an eight-stage completed curriculum.

## Why This Matters

The prior PR95 MLX lane could run local timing smokes, but the queue still needed
manual JSON construction before it could own Stage 1/5/8 work. This landing turns
the native MLX reproduction lane into queue-owned local-training work while
keeping it planning-only and non-authoritative.

## Authority Boundary

All emitted plan and sidecar rows preserve:

- `score_claim=false`
- `promotion_eligible=false`
- `rank_or_kill_eligible=false`
- `ready_for_exact_eval_dispatch=false`
- `promotable=false`

Dispatch/readiness blockers still require runtime consumption proof, receiver
proof, PyTorch export forward parity, byte-closed contest archive export, and
exact CPU/CUDA auth eval.

## Verification

- `.venv/bin/ruff check tools/run_pr95_mlx_timing_smoke.py src/tac/tests/test_run_pr95_mlx_timing_smoke_plan.py`
- `.venv/bin/python -m pytest -q src/tac/tests/test_run_pr95_mlx_timing_smoke_plan.py src/tac/tests/test_run_pr95_mlx_timing_smoke.py src/tac/tests/test_pr95_hnerv_mlx.py src/tac/tests/test_local_training_runtime_profile.py src/tac/tests/test_representation_training_probe_integration.py`
- `.venv/bin/python tools/lane_maturity.py validate`

Observed result after stage-count tightening:

- 23 focused tests passed.
- Ruff passed on the touched PR95 MLX runner/tests.
- Manual `--plan-only --allow-existing-output-dir` smoke emitted
  `schema=pr95_hnerv_mlx_timing_smoke_plan_summary_v1`, `stage_count=1`,
  `stage_index=8`, sidecar `candidate_params.stage_count=1`, and
  `resource_kind=local_mlx`.

## Remaining Gaps

- The PR95 MLX lane has source-model PyTorch-to-MLX forward parity for the
  decoder topology. It still needs public checkpoint parity and codec/runtime
  consumption proof.
- The smoke archive is byte-closed for queue plumbing only; it is not consumed by
  the PR95 contest runtime.
- Stage 1/5/8 queue fan-out can now be built, but a bounded local executor run
  should be harvested into the optimizer candidate queue before using it for
  throughput decisions.
