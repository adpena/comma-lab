# Byte-Shaving Campaign Plan: byte_shaving_campaign

- schema: `byte_shaving_campaign_plan.v1`
- generated_at_utc: `2026-05-23T14:47:18Z`
- lane_id: `master_gradient_byte_shaving_planning`
- frontier_axis: `[contest-CUDA]`
- planning_only: `True`

## Recommended Prefix
- `top_0004` units=`4` saved_bytes=`16246` expected_delta=`-0.010817544552422797`
- units: `mg_byte_span_0162171_0166267,mg_byte_span_0166267_0170363,mg_byte_span_0170363_0174459,mg_byte_span_0174459_0178417`

## Recommended Combination
- `combo_0055` units=`4` saved_bytes=`16246` expected_delta=`-0.010817544552422795`
- units: `mg_byte_span_0162171_0166267,mg_byte_span_0166267_0170363,mg_byte_span_0170363_0174459,mg_byte_span_0174459_0178417`
- operations: `entropy_recode`

## Top Combination Ladder
| rank | combo | units | saved bytes | expected delta | operations |
|---:|---|---:|---:|---:|---|
| 1 | `combo_0055` | 4 | 16246 | `-0.010817544552422795` | `entropy_recode` |
| 2 | `combo_0056` | 4 | 16246 | `-0.010817544552422795` | `entropy_recode,null_remove_or_seed` |
| 3 | `combo_0058` | 4 | 16246 | `-0.010817544552422795` | `entropy_recode,null_remove_or_seed` |
| 4 | `combo_0059` | 4 | 16246 | `-0.010817544552422795` | `entropy_recode,null_remove_or_seed` |
| 5 | `combo_0061` | 4 | 16246 | `-0.010817544552422795` | `entropy_recode,null_remove_or_seed` |
| 6 | `combo_0062` | 4 | 16246 | `-0.010817544552422795` | `entropy_recode,null_remove_or_seed` |
| 7 | `combo_0064` | 4 | 16246 | `-0.010817544552422795` | `entropy_recode,null_remove_or_seed` |
| 8 | `combo_0065` | 4 | 16246 | `-0.010817544552422795` | `entropy_recode,null_remove_or_seed` |
| 9 | `combo_0067` | 4 | 16246 | `-0.010817544552422795` | `null_remove_or_seed,entropy_recode` |
| 10 | `combo_0068` | 4 | 16246 | `-0.010817544552422795` | `null_remove_or_seed,entropy_recode` |
| 11 | `combo_0070` | 4 | 16246 | `-0.010817544552422795` | `null_remove_or_seed,entropy_recode` |
| 12 | `combo_0071` | 4 | 16246 | `-0.010817544552422795` | `null_remove_or_seed,entropy_recode` |

## Authority Boundary
- score_claim: `false`
- promotion_eligible: `false`
- ready_for_exact_eval_dispatch: `false`
- next_gate: `materialize selected operation family, run locality/inflate controls, then exact auth eval before any score claim`
