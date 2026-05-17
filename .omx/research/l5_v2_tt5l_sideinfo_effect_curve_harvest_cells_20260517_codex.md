# L5 v2 TT5L side-info effect curve harvest cells

This memo is the post-harvest bridge from the Lightning paired-axis plan to the TT5L side-info effect-curve builder. It does not launch provider work and does not claim score movement.

## Authority

- Score claim: `false`
- Promotion eligible: `false`
- Ready for exact eval dispatch: `false`
- Ready for provider dispatch: `false`
- Dispatch attempted: `false`

## Sources

- Lightning paired-axis plan: `.omx/research/lightning_paired_axis_plan.json`
- Lightning plan SHA-256: `890c5e7e53683833ce3b6507a7f672d87a2d6ce99bf74e21dc29614a7bd1890b`
- Variant manifest: `.omx/research/variant_manifest.json`
- Variant manifest SHA-256: `cc88026db44fd51c71f395058a6071ed0b1c9ff56c71fedf20d7887be1859bc8`

## Cell Status

| variant | axis | expected artifact | ready | blockers |
| --- | --- | --- | --- | --- |
| `zero` | `contest_cpu` | `experiments/results/lightning_batch/test_tt5l_paired_axes/zero/contest_cpu/contest_auth_eval.json` | `true` | - |
| `zero` | `contest_cuda` | `experiments/results/lightning_batch/test_tt5l_paired_axes/zero/contest_cuda/contest_auth_eval.json` | `false` | harvested_exact_eval_artifact_missing:zero:contest_cuda |
| `random_lsb` | `contest_cpu` | `experiments/results/lightning_batch/test_tt5l_paired_axes/random_lsb/contest_cpu/contest_auth_eval.json` | `false` | harvested_exact_eval_artifact_missing:random_lsb:contest_cpu |
| `random_lsb` | `contest_cuda` | `experiments/results/lightning_batch/test_tt5l_paired_axes/random_lsb/contest_cuda/contest_auth_eval.json` | `false` | harvested_exact_eval_artifact_missing:random_lsb:contest_cuda |
| `shuffled` | `contest_cpu` | `experiments/results/lightning_batch/test_tt5l_paired_axes/shuffled/contest_cpu/contest_auth_eval.json` | `false` | harvested_exact_eval_artifact_missing:shuffled:contest_cpu |
| `shuffled` | `contest_cuda` | `experiments/results/lightning_batch/test_tt5l_paired_axes/shuffled/contest_cuda/contest_auth_eval.json` | `false` | harvested_exact_eval_artifact_missing:shuffled:contest_cuda |
| `trained` | `contest_cpu` | `experiments/results/lightning_batch/test_tt5l_paired_axes/trained/contest_cpu/contest_auth_eval.json` | `false` | harvested_exact_eval_artifact_missing:trained:contest_cpu |
| `trained` | `contest_cuda` | `experiments/results/lightning_batch/test_tt5l_paired_axes/trained/contest_cuda/contest_auth_eval.json` | `false` | harvested_exact_eval_artifact_missing:trained:contest_cuda |
| `ablated` | `contest_cpu` | `experiments/results/lightning_batch/test_tt5l_paired_axes/ablated/contest_cpu/contest_auth_eval.json` | `false` | harvested_exact_eval_artifact_missing:ablated:contest_cpu |
| `ablated` | `contest_cuda` | `experiments/results/lightning_batch/test_tt5l_paired_axes/ablated/contest_cuda/contest_auth_eval.json` | `false` | harvested_exact_eval_artifact_missing:ablated:contest_cuda |

## Next Gate

- Ready for effect-curve build: `false`
- Ready cells: `1/10`
- Harvested exact-eval artifacts: `1`
- Missing exact-eval artifacts: `9`
- Effect-curve command: `.venv/bin/python tools/build_l5_v2_sideinfo_effect_curve.py --cell-json .omx/research/l5_v2_tt5l_sideinfo_effect_curve_harvest_cells_20260517_codex.json`

The downstream effect-curve artifact remains the only surface that can satisfy the side-info predicate. This bridge is custody plumbing, not a promotion artifact.
