# Inverse-Steganalysis Action Functional

- schema: `inverse_steganalysis_discrete_action_functional.v1`
- cells: `2`
- blocked_cells: `0`
- selected_cells: `2`
- selected_water_fill_cost_bytes: `40`
- selected_expected_score_gain: `0.000134`

## Math Model
- representation: `discrete_riemann_sum_with_second_order_interactions`
- stationarity_rule: `select positive euler_lagrange_residual cells under byte budget and guard barriers`
- lambda_rate: `5e-07`

## Selected Water Buckets
| rank | atom | candidate | scope | component | bytes | expected gain | residual |
|---:|---|---|---|---|---:|---:|---:|
| 1 | `inverse_surface_high_level_smoke_receiver_sufficient_statistic_pose_pair_0012_00_0000` | `high_level_smoke_candidate` | `pairs` | `posenet` | 24 | `9.6e-05` | `3.5e-06` |
| 2 | `inverse_surface_high_level_smoke_rate_only_null_space_seg_pair_0020_0021_0001` | `high_level_smoke_candidate` | `pairs` | `segnet` | 16 | `3.8e-05` | `1.8750000000000003e-06` |

## Authority Boundary
- score_claim: `false`
- score_claim_valid: `false`
- promotion_eligible: `false`
- rank_or_kill_eligible: `false`
- ready_for_exact_eval_dispatch: `false`
