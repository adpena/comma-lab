# Byte-Shaving Campaign Plan: campaign3_ias1aware_feedback_replan

- schema: `byte_shaving_campaign_plan.v1`
- generated_at_utc: `2026-05-25T06:01:18Z`
- lane_id: `inverse_steganalysis_action_byte_shaving`
- frontier_axis: `[planning-only inverse-steganalysis action]`
- planning_only: `True`

## Recommended Prefix
- `top_0001` units=`1` saved_bytes=`0` expected_delta=`-3.8e-05`
- units: `inverse_action_inverse_surface_high_level_smoke_rate_only_null_space_seg_pair_0020_0021_0001`

## Recommended Combination
- none

## Top Combination Ladder
| rank | combo | units | saved bytes | expected delta | operations |
|---:|---|---:|---:|---:|---|

## Bounded Permutation Priors
| rank | combo | top sequence | inversion penalty |
|---:|---|---|---:|

## Search Policy
- combination_search: `bounded_beam_over_units_and_operation_alternatives`
- permutation_search: `bounded_operation_order_permutations_for_top_combos`
- non_bruteforce_principle: `rank by rate-distortion prior, component/scorer marginal costs, explicit interactions, conflicts, and confidence before any local or exact scorer spend`

## Authority Boundary
- score_claim: `false`
- promotion_eligible: `false`
- ready_for_exact_eval_dispatch: `false`
- next_gate: `materialize selected operation family, run locality/inflate controls, then exact auth eval before any score claim`
