# DFL1 Exact-Handoff Overwrite Closure

UTC: 2026-05-26T18:16:13Z

## Summary

Codex preserved the targeted repair materializer handoff signal after the
receiver-closed DFL1 sidecar overlay. The first closure rerun showed a real
queue bug: exact-readiness followup steps could rerun `submission_closure` and
`exact_readiness_bridge` with overwrite enabled, but `harvest_materializer_chains`
and `build_exact_eval_dispatch_plan` still refused to replace stale handoff
JSON. That left a false failure on an otherwise receiver-closed, deterministic
encoder-side artifact.

## Fix

- Added `--overwrite` to generated materializer harvest followup commands.
- Added `--overwrite` to generated exact-eval dispatch-plan commands.
- Updated the focused DFL1 materializer execution queue artifact so reruns use
  the same overwrite contract.
- Preserved both the failed closure rerun record and the post-fix queue status
  so the stale-artifact failure mode is not lost.

## Verification

- `ruff check src/comma_lab/scheduler/byte_shaving_campaign_queue.py src/tac/tests/test_byte_shaving_campaign_queue.py`
- `.venv/bin/python -m pytest src/tac/tests/test_byte_shaving_campaign_queue.py -q`
- `.venv/bin/python tools/experiment_queue.py --queue .omx/research/frontier_mlx_paired_reference_cache_reuse_20260526T154742Z/targeted_component_correction_bound_materializer_execution_queue.json validate`
- `.venv/bin/python tools/experiment_queue.py --queue .omx/research/frontier_mlx_paired_reference_cache_reuse_20260526T154742Z/targeted_component_correction_bound_materializer_execution_queue.json status`
- `.venv/bin/python tools/review_tracker.py policy-check src/comma_lab/scheduler/byte_shaving_campaign_queue.py src/tac/tests/test_byte_shaving_campaign_queue.py`
- `.venv/bin/python tools/review_gate_hook.py`

The post-fix queue status has all six DFL1 materializer/exact-handoff steps
marked succeeded. The previously failing `build_exact_eval_dispatch_plan` step
reran with `--overwrite` and emitted a dry-run dispatch plan/queue only. No
score claim, promotion claim, rank/kill authority, paid dispatch, or receiver
optimization authority was introduced.

## Next Integration Point

Use this closure as the runnable handoff template for the 2026-05-26 paired
repair dynamics queue: the allocator/materializer side may keep rerunning local
MLX acquisition and encoder-side repair candidates, but exact CPU/CUDA authority
still requires the normal lane claim, dispatch queue authorization, and auth eval
harvest before any score wording.
