# Byte-Shaving Campaign Plan: campaign3_ias1aware_feedback_candidate_actuation

- schema: `byte_shaving_campaign_plan.v1`
- generated_at_utc: `2026-05-25T07:38:45Z`
- lane_id: `inverse_steganalysis_action_byte_shaving`
- frontier_axis: `[planning-only inverse-steganalysis action]`
- planning_only: `True`

## Recommended Prefix
- `top_0003` units=`3` saved_bytes=`0` expected_delta=`-0.0001606`
- units: `inverse_action_inverse_surface_high_level_smoke_receiver_sufficient_statistic_pose_pair_0012_00_0000,inverse_action_inverse_surface_high_level_smoke_rate_only_null_space_seg_pair_0020_0021_0001,inverse_action_byte_shaving_unit_inverse_action_inverse_surface_high_level_smoke_rate_only_null_s_0000`

## Recommended Combination
- `combo_0002` units=`3` saved_bytes=`0` expected_delta=`-0.0001606`
- units: `inverse_action_inverse_surface_high_level_smoke_receiver_sufficient_statistic_pose_pair_0012_00_0000,inverse_action_inverse_surface_high_level_smoke_rate_only_null_space_seg_pair_0020_0021_0001,inverse_action_byte_shaving_unit_inverse_action_inverse_surface_high_level_smoke_rate_only_null_s_0000`
- operations: `compile_inverse_steganalysis_operation_set`

## Top Combination Ladder
| rank | combo | units | saved bytes | expected delta | operations |
|---:|---|---:|---:|---:|---|
| 1 | `combo_0002` | 3 | 0 | `-0.0001606` | `compile_inverse_steganalysis_operation_set` |
| 2 | `combo_0001` | 2 | 0 | `-0.000134` | `compile_inverse_steganalysis_operation_set` |
| 3 | `combo_0003` | 2 | 0 | `-0.0001226` | `compile_inverse_steganalysis_operation_set` |
| 4 | `combo_0004` | 2 | 0 | `-6.46e-05` | `compile_inverse_steganalysis_operation_set` |

## Bounded Permutation Priors
| rank | combo | top sequence | inversion penalty |
|---:|---|---|---:|
| 1 | `combo_0002` | `compile_inverse_steganalysis_operation_set -> compile_inverse_steganalysis_operation_set -> compile_inverse_steganalysis_operation_set` | 0 |
| 2 | `combo_0001` | `compile_inverse_steganalysis_operation_set -> compile_inverse_steganalysis_operation_set` | 0 |
| 3 | `combo_0003` | `compile_inverse_steganalysis_operation_set -> compile_inverse_steganalysis_operation_set` | 0 |
| 4 | `combo_0004` | `compile_inverse_steganalysis_operation_set -> compile_inverse_steganalysis_operation_set` | 0 |

## Search Policy
- combination_search: `bounded_beam_over_units_and_operation_alternatives`
- permutation_search: `bounded_operation_order_permutations_for_top_combos`
- non_bruteforce_principle: `rank by rate-distortion prior, component/scorer marginal costs, explicit interactions, conflicts, and confidence before any local or exact scorer spend`

## Authority Boundary
- score_claim: `false`
- promotion_eligible: `false`
- ready_for_exact_eval_dispatch: `false`
- next_gate: `materialize selected operation family, run locality/inflate controls, then exact auth eval before any score claim`
