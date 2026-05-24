# Byte-Shaving Campaign Plan: high_level_byte_shaving_runner_smoke_20260524T050723Z

- schema: `byte_shaving_campaign_plan.v1`
- generated_at_utc: `2026-05-24T05:12:10Z`
- lane_id: `inverse_steganalysis_action_byte_shaving`
- frontier_axis: `[planning-only inverse-steganalysis action]`
- planning_only: `True`

## Recommended Prefix
- `top_0002` units=`2` saved_bytes=`0` expected_delta=`-0.000134`
- units: `inverse_action_inverse_surface_high_level_smoke_receiver_sufficient_statistic_pose_pair_0012_00_0000,inverse_action_inverse_surface_high_level_smoke_rate_only_null_space_seg_pair_0020_0021_0001`

## Recommended Combination
- `combo_0001` units=`2` saved_bytes=`0` expected_delta=`-0.000134`
- units: `inverse_action_inverse_surface_high_level_smoke_receiver_sufficient_statistic_pose_pair_0012_00_0000,inverse_action_inverse_surface_high_level_smoke_rate_only_null_space_seg_pair_0020_0021_0001`
- operations: `materialize_inverse_scorer_cell_candidate`

## Top Combination Ladder
| rank | combo | units | saved bytes | expected delta | operations |
|---:|---|---:|---:|---:|---|
| 1 | `combo_0001` | 2 | 0 | `-0.000134` | `materialize_inverse_scorer_cell_candidate` |

## Bounded Permutation Priors
| rank | combo | top sequence | inversion penalty |
|---:|---|---|---:|
| 1 | `combo_0001` | `materialize_inverse_scorer_cell_candidate -> materialize_inverse_scorer_cell_candidate` | 0 |

## Search Policy
- combination_search: `bounded_beam_over_units_and_operation_alternatives`
- permutation_search: `bounded_operation_order_permutations_for_top_combos`
- non_bruteforce_principle: `rank by rate-distortion prior, component/scorer marginal costs, explicit interactions, conflicts, and confidence before any local or exact scorer spend`

## Authority Boundary
- score_claim: `false`
- promotion_eligible: `false`
- ready_for_exact_eval_dispatch: `false`
- next_gate: `materialize selected operation family, run locality/inflate controls, then exact auth eval before any score claim`
