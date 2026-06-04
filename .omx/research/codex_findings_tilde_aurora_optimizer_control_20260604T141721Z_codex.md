# Codex Findings: Tilde Aurora Optimizer Control Slice

written_at_utc: 2026-06-04T14:17:21Z
lane_id: lane_tilde_aurora_optimizer_control_20260604
source_intake: .omx/research/codex_findings_tilde_research_parallax_nerv_intake_20260603T174229Z_codex.md
axis_status: false_authority_planner_control_only
score_claim: false
promotion_eligible: false
ready_for_exact_eval_dispatch: false

## What Changed

The NeRV long-training planner now accepts `aurora` as an alias for
`aurora_like`, and the shared MLX score-aware adapter now exposes a real
Pact-local Aurora-like optimizer port. It remains a timing-smoke candidate, not
the default optimizer and not PR95 source authority.

Aurora rows carry explicit launch blockers:

```json
[
  "aurora_not_pr95_source_authority",
  "aurora_requires_local_timing_convergence_smoke"
]
```

The planner renders runnable timing-smoke command skeletons for provenance, but
the launch authority contract still blocks promotion/dispatch until local timing
and convergence evidence exists. The optimizer itself is a local MLX port of the
public Aurora update shape, not a source-faithful upstream reproduction.

## Files

- `src/tac/analysis/nerv_long_training_campaign_plan.py`
- `src/tac/substrates/_shared/mlx_score_aware/adapter.py`
- `src/tac/substrates/_shared/mlx_score_aware/tests/test_wave_n11_stabilizer.py`
- `src/tac/tests/test_nerv_long_training_campaign_plan.py`
- `.omx/state/lane_registry.json`
- `.omx/state/lane_maturity_audit.log`

## Verification

- `uv run pytest src/tac/tests/test_nerv_long_training_campaign_plan.py::test_aurora_like_optimizer_row_is_native_mlx_timing_smoke_and_fail_closed src/tac/tests/test_nerv_long_training_campaign_plan.py::test_default_optimizer_kinds_cover_native_mlx_optimizer_surface src/tac/tests/test_nerv_long_training_campaign_plan.py::test_long_training_campaign_plan_builds_optimizer_matrix -q` -> 3 passed
- `uv run pytest src/tac/tests/test_nerv_long_training_campaign_plan.py -q` -> 80 passed
- `uv run pytest src/tac/substrates/_shared/mlx_score_aware/tests/test_wave_n11_stabilizer.py -q` -> included in the 266-test changed-surface pass
- `uv run ruff check src/tac/analysis/nerv_long_training_campaign_plan.py src/tac/tests/test_nerv_long_training_campaign_plan.py`
- `uv run python -m py_compile src/tac/analysis/nerv_long_training_campaign_plan.py src/tac/substrates/_shared/mlx_score_aware/adapter.py src/tac/tests/test_nerv_long_training_campaign_plan.py`

`uv run python tools/lane_maturity.py validate` still fails on pre-existing
`lane_z8_symbolic_lambda_wavelet_blob_20260601` missing evidence paths; this
landing did not touch those evidence files or that lane.

## Next Action

Run the real Aurora-like MLX optimizer smoke only after the shared
`mlx_score_aware` adapter ownership and Metal stability fixes are committed:

```json
{
  "target": "HiNeRV/HNeRV",
  "classification": "timing_smoke_candidate",
  "required_code_surface": [
    "src/tac/substrates/_shared/mlx_score_aware/adapter.py",
    "src/tac/substrates/_shared/mlx_score_aware/tests/",
    "tools/run_compact_renderer_mlx_spine_runner.py"
  ],
  "minimum_smoke": "tiny PR95/HiNeRV local advisory comparison: pact_muon_adamw vs muon vs aurora_like",
  "required_output": "seconds_per_epoch, loss telemetry, SegNet/PoseNet components, unchanged byte ceiling, false-authority flags",
  "promotion_rule": "no score or dispatch authority until receiver-closed archive/runtime and exact contest-axis replay"
}
```
