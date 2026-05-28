---
schema: codex_findings.v1
generated_utc: 2026-05-28T12:47:00Z
lane_id: lane_pr95_hnerv_mlx_reproduction
evidence_grade: "[repo-local/experiment_queue.v1/macOS-MLX research-signal]"
score_claim: false
promotion_eligible: false
ready_for_exact_eval_dispatch: false
---

# PR95 MLX drift-scope search is now queue-owned

The PR95 Conv2d drift scope search is no longer only a manual CLI. It can emit a
`local_mlx_drift_scope_search_plan.v1` plan declaring whether the target is
`contest_video_overfit`, `single_video`, `video_corpus`, or
`archive_generalization`. The new `comma_lab.scheduler.mlx_drift_scope_queue`
compiler turns those plans into `experiment_queue.v1` work with false-authority
postconditions and local CPU/MLX resource custody.

The design keeps the contest-video overfit explicit for the challenge while
preserving a reusable contract for other videos, HNeRV variants, BoostNeRV
bolt-ons, NeRV-family, and non-NeRV archives once adapters emit the same plan
schema.

## Proof Artifact

- `.omx/research/pr95_mlx_drift_scope_queue_baseline_cpu_20260528T124547Z/queue.json`
- `.omx/research/pr95_mlx_drift_scope_queue_baseline_cpu_20260528T124547Z/worker_result.json`
- `.omx/research/pr95_mlx_drift_scope_queue_baseline_cpu_20260528T124547Z/scope_search_summary.json`

Bounded worker execution ran the baseline-only CPU scope search through
`experiment_queue.v1`, produced one summary artifact, and satisfied all declared
postconditions. The output remains local advisory only:
`score_claim=false`, `promotion_eligible=false`,
`ready_for_exact_eval_dispatch=false`.

## Verification

- `.venv/bin/ruff check src/comma_lab/scheduler/mlx_drift_scope_queue.py src/comma_lab/scheduler/__init__.py tools/build_mlx_drift_scope_queue.py tools/run_pr95_mlx_conv2d_drift_scope_search.py src/tac/tests/test_mlx_drift_scope_queue.py`
- `.venv/bin/python -m pytest -q src/tac/tests/test_mlx_drift_scope_queue.py src/tac/tests/test_run_pr95_mlx_conv2d_drift_scope_search.py src/tac/tests/test_run_pr95_mlx_timing_smoke_plan.py`
- `tools/experiment_queue.py --queue <proof>/queue.json run-worker --execute --max-steps 1`

## Next Wiring

Use this queue compiler as the template for broader MLX drift acquisition:
module-scope Conv2d today, scorer-response calibration and substrate-training
parity tomorrow. The next adapters should emit the same optimization-target
metadata so scheduler/acquisition layers can distinguish contest-video
overfitting from corpus-generalizable training without ad hoc scripts.
