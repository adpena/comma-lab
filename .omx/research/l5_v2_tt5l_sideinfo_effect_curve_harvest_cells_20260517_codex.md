# L5 v2 TT5L side-info effect curve harvest cells

This memo is the post-harvest bridge from the Lightning paired-axis plan to the TT5L side-info effect-curve builder. It does not launch provider work and does not claim score movement.

## Authority

- Score claim: `false`
- Promotion eligible: `false`
- Ready for exact eval dispatch: `false`
- Ready for provider dispatch: `false`
- Dispatch attempted: `false`

## Sources

- Lightning paired-axis plan: `.omx/research/l5_v2_tt5l_sideinfo_effect_curve_lightning_paired_axis_plan_20260517_codex.json`
- Lightning plan SHA-256: `8010d81abbcbca3df55b3681d6b7e455231283c08f16f2a6e669f24fab8ac9c1`
- Variant manifest: `.omx/research/l5_v2_tt5l_current_code_fullshape_sideinfo_variant_packets_20260517_codex.json`
- Variant manifest SHA-256: `80962a29c5abc8c8de2dbae742e228c396e5f3ce3423397cfe7d79bae1f06459`

## Cell Status

| variant | axis | expected artifact | ready | blockers |
| --- | --- | --- | --- | --- |
| `zero` | `contest_cpu` | `experiments/results/lightning_batch/l5_v2_tt5l_sideinfo_effect_curve_paired_axes/zero/contest_cpu/contest_auth_eval.json` | `false` | harvested_exact_eval_artifact_missing:zero:contest_cpu |
| `zero` | `contest_cuda` | `experiments/results/lightning_batch/l5_v2_tt5l_sideinfo_effect_curve_paired_axes/zero/contest_cuda/contest_auth_eval.json` | `false` | harvested_exact_eval_artifact_missing:zero:contest_cuda |
| `random_lsb` | `contest_cpu` | `experiments/results/lightning_batch/l5_v2_tt5l_sideinfo_effect_curve_paired_axes/random_lsb/contest_cpu/contest_auth_eval.json` | `false` | harvested_exact_eval_artifact_missing:random_lsb:contest_cpu |
| `random_lsb` | `contest_cuda` | `experiments/results/lightning_batch/l5_v2_tt5l_sideinfo_effect_curve_paired_axes/random_lsb/contest_cuda/contest_auth_eval.json` | `false` | harvested_exact_eval_artifact_missing:random_lsb:contest_cuda |
| `shuffled` | `contest_cpu` | `experiments/results/lightning_batch/l5_v2_tt5l_sideinfo_effect_curve_paired_axes/shuffled/contest_cpu/contest_auth_eval.json` | `false` | harvested_exact_eval_artifact_missing:shuffled:contest_cpu |
| `shuffled` | `contest_cuda` | `experiments/results/lightning_batch/l5_v2_tt5l_sideinfo_effect_curve_paired_axes/shuffled/contest_cuda/contest_auth_eval.json` | `false` | harvested_exact_eval_artifact_missing:shuffled:contest_cuda |
| `trained` | `contest_cpu` | `experiments/results/lightning_batch/l5_v2_tt5l_sideinfo_effect_curve_paired_axes/trained/contest_cpu/contest_auth_eval.json` | `false` | harvested_exact_eval_artifact_missing:trained:contest_cpu |
| `trained` | `contest_cuda` | `experiments/results/lightning_batch/l5_v2_tt5l_sideinfo_effect_curve_paired_axes/trained/contest_cuda/contest_auth_eval.json` | `false` | harvested_exact_eval_artifact_missing:trained:contest_cuda |
| `ablated` | `contest_cpu` | `experiments/results/lightning_batch/l5_v2_tt5l_sideinfo_effect_curve_paired_axes/ablated/contest_cpu/contest_auth_eval.json` | `false` | harvested_exact_eval_artifact_missing:ablated:contest_cpu |
| `ablated` | `contest_cuda` | `experiments/results/lightning_batch/l5_v2_tt5l_sideinfo_effect_curve_paired_axes/ablated/contest_cuda/contest_auth_eval.json` | `false` | harvested_exact_eval_artifact_missing:ablated:contest_cuda |

## Next Gate

- Ready for effect-curve build: `false`
- Ready cells: `0/10`
- Harvested exact-eval artifacts: `0`
- Missing exact-eval artifacts: `10`
- Effect-curve command: `.venv/bin/python tools/build_l5_v2_sideinfo_effect_curve.py --cell-json .omx/research/l5_v2_tt5l_sideinfo_effect_curve_harvest_cells_20260517_codex.json`

The downstream effect-curve artifact remains the only surface that can satisfy the side-info predicate. This bridge is custody plumbing, not a promotion artifact.
