# Codex Findings: MLX Execution-Plan Bridge for LL Scorer-Response Harvest

## Verdict

PROCEED as a local-MLX candidate-generation bridge only. This landing turns the
profile-stability row selection into a reusable execution contract without
granting score, promotion, rank/kill, or exact-eval dispatch authority.

## What landed

- Added `tac.local_acceleration.mlx_execution_plan` with schema
  `mlx_scorer_response_execution_plan.v1`.
- Added `tools/plan_mlx_scorer_response_execution.py`.
- Added optional `--mlx-profile-stability` attachment to
  `tools/plan_ll_scorer_response_next.py`.
- Added a human-visible `MLX Execution Recommendation` section to the LL plan
  markdown renderer.
- Preserved `profile_summary.archive_size_bytes` in newly generated profile
  stability manifests so future execution plans do not need a manual byte count.

## Real FEC6 execution-plan artifact

Source profile-stability manifest:

`experiments/results/mlx_scorer_response_profile_fec6_20260521T2340Z_pairs16_4_cpu_gpu/stability_cpu_baseline_with_selection.json`

Generated plan artifacts:

- `experiments/results/mlx_scorer_response_profile_fec6_20260521T2340Z_pairs16_4_cpu_gpu/execution_plan_cpu_b2_pairs16_20.json`
- `experiments/results/mlx_scorer_response_profile_fec6_20260521T2340Z_pairs16_4_cpu_gpu/execution_plan_cpu_b2_pairs16_20.md`
- `experiments/results/pr110_provisional_hfv1_engineering_20260520_codex/contest_oracle_search/ll_scorer_response_next_probe_plan_null_pair4_mlx_execution_20260521_codex.json`
- `experiments/results/pr110_provisional_hfv1_engineering_20260520_codex/contest_oracle_search/ll_scorer_response_next_probe_plan_null_pair4_mlx_execution_20260521_codex.md`

Recommended execution:

```bash
.venv/bin/python tools/run_mlx_scorer_response_cache.py --reference-cache-dir experiments/results/mlx_scorer_input_cache_reference_video_20260521T2304Z_full600 --candidate-cache-dir experiments/results/mlx_scorer_input_cache_fec6_pr101_20260521T210100Z_full300pairs --archive-size-bytes 178517 --repo-root . --batch-pairs 2 --start-pair 16 --max-pairs 4 --device cpu --output experiments/results/mlx_scorer_response_profile_fec6_20260521T2340Z_pairs16_4_cpu_gpu/next_response_cpu_b2_pairs16_20.json
```

Empirical execution result:

- `avg_posenet_dist`: `6.13186800535459e-06`
- `avg_segnet_dist`: `0.0005149841381353326`
- `canonical_score`: `0.1781961810476318`
- `n_samples`: `4`
- `score_claim`: `false`

This reproduces the selected CPU row from the profile. The full source profile
still has verdict `FAIL_MLX_PROFILE_STABILITY` because the GPU rows drift; the
execution plan therefore records `profile_full_pass_required=false` and uses
only the eligible CPU row (`device=cpu`, `batch_pairs=2`, pair window
`[16, 20]`).

## Byte-count correction caught during execution

An initial manual plan used `178417` bytes, which reproduced the component
averages but not the profile row's rate term. Empirical `wc -c` on the FEC6
archive and the source profile both confirm the correct byte count is `178517`.
The generated execution-plan and LL-plan artifacts were regenerated with
`178517`, and the final runner output exactly matches the selected profile row.

## False-authority guards

The planner fails closed unless:

- profile-stability evidence grade is `macOS-MLX-research-signal`;
- `score_claim`, `score_claim_valid`, `promotion_eligible`,
  `ready_for_exact_eval_dispatch`, and `rank_or_kill_eligible` are explicit
  `false`;
- a `selection.recommended_row` exists and is listed in
  `selection.eligible_row_indices`;
- selected GPU rows are explicitly allowed with research-signal language and
  remain limited to `batch_pairs=1` until batch-invariance passes.

## Verification

```bash
.venv/bin/python -m ruff check src/tac/local_acceleration/mlx_execution_plan.py src/tac/local_acceleration/mlx_profile_stability.py src/tac/optimization/scorer_response_dataset.py tools/plan_mlx_scorer_response_execution.py tools/plan_ll_scorer_response_next.py src/tac/tests/test_mlx_execution_plan.py src/tac/tests/test_mlx_profile_stability.py
.venv/bin/python -m pytest -q src/tac/tests/test_mlx_execution_plan.py src/tac/tests/test_mlx_profile_stability.py src/tac/tests/test_plan_ll_scorer_response_next_cli.py src/tac/tests/test_scorer_response_dataset.py
.venv/bin/python -m pytest -q src/tac/tests/test_mlx_execution_plan.py src/tac/tests/test_mlx_profile_stability.py src/tac/tests/test_plan_ll_scorer_response_next_cli.py src/tac/tests/test_scorer_response_dataset.py src/tac/tests/test_profile_mlx_scorer_response_cache.py src/tac/tests/test_mlx_scorer_response.py src/tac/tests/test_mlx_batch_invariance.py
```

Results:

- Ruff: pass
- Focused pytest: `32 passed`
- Expanded MLX pytest: `50 passed`
