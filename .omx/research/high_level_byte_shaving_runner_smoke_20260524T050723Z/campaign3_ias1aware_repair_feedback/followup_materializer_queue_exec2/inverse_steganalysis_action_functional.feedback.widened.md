# Inverse-Steganalysis Action Functional

- schema: `inverse_steganalysis_discrete_action_functional.v1`
- cells: `3`
- blocked_cells: `0`
- selected_cells: `3`
- selected_water_fill_cost_bytes: `41`
- selected_expected_score_gain: `0.0001606`

## Math Model
- representation: `discrete_riemann_sum_with_second_order_interactions`
- stationarity_rule: `select positive euler_lagrange_residual cells under byte budget and guard barriers`
- lambda_rate: `6.658589531221714e-07`

## Selected Water Buckets
| rank | atom | candidate | scope | component | bytes | expected gain | residual |
|---:|---|---|---|---|---:|---:|---:|
| 1 | `byte_shaving_unit_inverse_action_inverse_surface_high_level_smoke_rate_only_null_s_0000` | `high_level_smoke_candidate` | `full_video` | `segnet` | 1 | `2.6600000000000003e-05` | `2.593414104687783e-05` |
| 2 | `inverse_surface_high_level_smoke_receiver_sufficient_statistic_pose_pair_0012_00_0000` | `pair_region_pose_gain` | `pairs` | `posenet` | 24 | `9.6e-05` | `3.3341410468778285e-06` |
| 3 | `inverse_surface_high_level_smoke_rate_only_null_space_seg_pair_0020_0021_0001` | `rate_only_null_space` | `pairs` | `segnet` | 16 | `3.8e-05` | `1.7091410468778288e-06` |

## Authority Boundary
- score_claim: `false`
- score_claim_valid: `false`
- promotion_eligible: `false`
- rank_or_kill_eligible: `false`
- ready_for_exact_eval_dispatch: `false`
