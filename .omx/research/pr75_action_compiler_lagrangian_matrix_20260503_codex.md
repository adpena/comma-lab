# PR75 Action Compiler Lagrangian Matrix - 2026-05-03

## Scope

Extended the local PR75 action-compiler search without dispatching remote GPU
work. This is a deterministic archive-build and planning ledger only. All new
score estimates are `byte_and_trace_planning_only_until_exact_cuda`; no archive
below is promoted, ranked as score truth, or claimed as exact evidence.

## Inputs

- Baseline C067 trace/archive:
  `experiments/results/lightning_batch/exact_eval_c063_fixedslice_equiv_t4_20260502T0855Z`
- PR75 source archive:
  `experiments/results/top_submission_reverse_engineering_20260503_pr75/archive.zip`
- PR75 all-action diagnostic:
  `experiments/results/lightning_batch/exact_eval_c067_pr75_actions_only_p3_diag_20260503T0346Z`
- T4 frontier top40:
  `experiments/results/lightning_batch/exact_eval_c067_pr75_actions_top40_p3_t4_20260503T0440Z`
- RTX PRO diagnostics:
  `experiments/results/lightning_batch/exact_eval_c067_pr75_actions_top25_ampminus1_p3_rtxprodiag_20260503T0456Z`
  `experiments/results/lightning_batch/exact_eval_c067_pr75_actions_top25_ampminus1_p5_rtxprodiag_20260503T0456Z`
  `experiments/results/lightning_batch/exact_eval_c067_pr75_actions_top40_ampminus2_p3_rtxprodiag_20260503T0456Z`
- Prior Pauli builder outputs:
  `experiments/results/c067_pr75_tile_action_compiler_next_p3_20260503_codex`
  `experiments/results/c067_pr75_tile_action_compiler_next_p5_20260503_codex`

## Evidence Read

Exact/component evidence says the smaller top40 P3 frontier did not preserve
the full PR75 all-action gain, while top25 ampminus1 improved materially over
top40 and top40 ampminus2 but still did not beat the all-action diagnostic.
The P5 top25 ampminus1 run had identical components to P3 and paid extra rate,
so P5 is reserved here for custom-delta probes only.

| artifact | bytes | recomputed score | seg | pose | rate | sha256 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| actions_only_p3_diag | 276460 | 0.3152359074190711 | 0.0607680000 | 0.0703846574 | 0.1840832500 | 851a61ddd88c0ddb4a05762f11e900f132fb771fd9dba743ac2026637ddf15d9 |
| top40_p3_t4 | 276386 | 0.3155226919767294 | 0.0610380000 | 0.0704506920 | 0.1840340000 | 9feef7ffaa254f9e5408996a122682757a054144a3000539553786c5292b7d0a |
| top25_ampminus1_p3 | 276328 | 0.31530060435788376 | 0.0609510000 | 0.0703541044 | 0.1839955000 | bdc966be526bb8f5ddcd433eaff2e3708779fd291eb40deea5539df5a7bc2386 |
| top25_ampminus1_p5 | 276442 | 0.3153766043578837 | 0.0609510000 | 0.0703541044 | 0.1840715000 | 36c4c017eaccbc55e78b16a4e89181f9c81987260c58d30f5e0838834643f23e |
| top40_ampminus2_p3 | 276385 | 0.3153510002824738 | 0.0609570000 | 0.0703605003 | 0.1840335000 | b37163dec8afe5e22fbe04cd1929576dbd73bca6975d61c74b4f7fc5145c9de7 |

## Builder Change

`experiments/build_pr75_tile_action_subset_candidates.py` now supports:

- `--calibration-component-trace TRANSFORM:MAX_RANK:PATH`
- `lag_eval[_poseW]_topN` calibrated Lagrangian non-prefix selection
- `--candidate POLICY:WIRE_FORMAT` to emit a mixed P3/P5 matrix without a
  policy/wire cross-product

The calibrated selector chooses at most one evaluated transform per source
action using exact per-pair component deltas and an action-brotli rate penalty.

## Output Matrix

Matrix path:
`experiments/results/c067_pr75_tile_action_compiler_lagrangian_20260503_codex/candidate_matrix.json`

Matrix SHA-256:
`fb838bcdbd81cf8543d4f79e4705e51c681e3860f1560c85da747750181c7a52`

| priority | candidate | bytes | sha256 | rationale |
| ---: | --- | ---: | --- | --- |
| 1 | `c067_pr75_actions_top67_ampfit_p3/archive.zip` | 276461 | 380dfa26def2408bf1ff4197f9895965251f9d7c352c9039d9f437304a13b403 | Full-action body preserved because all-action exact evidence is strongest; per-record ampfit reacts to pose sign instead of uniform shrink. |
| 2 | `c067_pr75_actions_all_ampminus1_p3/archive.zip` | 276461 | 54dd2a3aac631e84658a7dcb97efe914850aa8bf1d19d045debc63ceb9b82c51 | Uniform ampminus1 follow-up to the top25 ampminus1 diagnostic, expanded to the full PR75 action body. |
| 3 | `c067_pr75_actions_poseharm_ampminus1_p3/archive.zip` | 276462 | 33598d9aa54a0587fbc3dc9f31d7b6219a744c262e2ae1036a8422490d5c1f95 | Selective ampminus1 only on proxy pose-harm records; isolates whether the top25 gain came from shrinkage or the smaller subset. |
| 4 | `c067_pr75_actions_lag_eval_top40_signedposemix1_p3/archive.zip` | 276411 | 34fc0dd56fb5f8782840aad545c95b619763985bc1ba29e5e1984b9bb70e9afb | Exact-trace calibrated non-prefix beam plus signed residuals; lower-rate signed combo probe. |
| 5 | `c067_pr75_actions_top67_signedposemix1_p3/archive.zip` | 276536 | 7dbfb45c7c2c1676eaab81efb257065b221716dc2539096b3d7223e33815405c | Aggressive full-action signed combo for nonlinear component-response escape. |
| 6 | `c067_pr75_actions_top67_custompose125_p5/archive.zip` | 276716 | cf4f973fb7b94155aded13f6f2c7bcb2503a5e4c935dbbf9bfeef2e965881464 | Charged custom dictionary probe; high rate, but tests whether custom pose-aware deltas unlock a component gain unavailable to fixed P3 actions. |

## Verification

- `python -m py_compile experiments/build_pr75_tile_action_subset_candidates.py src/tac/tests/test_build_pr75_tile_action_subset_candidates.py`
- `.venv/bin/python -m pytest src/tac/tests/test_build_pr75_tile_action_subset_candidates.py`

No remote dispatch was performed.
