# CPU/CUDA exact-pair mechanism analysis

generated_at_utc: `2026-05-13T01:17:48Z`
evidence_grade: `paired_exact_auth_eval_mechanism_diagnostic`
score_claim: `false`
promotion_eligible: `false`

## Pair

- valid_individual_axis_scores: `True`
- valid_same_archive_axis_score_pair: `True`
- valid_for_mechanism_analysis: `False`
- valid_for_pair_score_analysis: `False`
- mechanism_class: `custody_incomplete`
- raw_output_pairing_status: `raw_output_manifest_missing`
- same_archive_sha256: `True`
- same_archive_bytes: `True`
- same_runtime_tree_sha256: `None`
- same_inflated_output_aggregate_sha256: `None`

| Axis | Score | Pose | Seg | Archive bytes |
| --- | ---: | ---: | ---: | ---: |
| CPU | 0.192847676138 | 3.286e-05 | 0.00056023 | 178262 |
| CUDA | 0.226352122180 | 0.00017103 | 0.00066299 | 178262 |

## CUDA Minus CPU

- score: `0.033504446041666025`
- pose_term: `0.02322844604166604`
- seg_term: `0.010276`
- rate_term: `0.0`

## Individual Axis Blockers

- none

## Blockers

- `cpu_runtime_tree_sha256_missing`

## Mechanism Blockers

- `raw_output_manifest_missing`

## Interpretation

- Negative score gap means CUDA scored lower than CPU for this exact pair; positive means CPU scored lower.
- Do not infer global CPU-better or CUDA-better behavior from one row.
- If raw outputs differ, runtime/inflate device behavior is part of the mechanism.
- If raw outputs match but scores differ, localize through GT loader and scorer-kernel xray probes.
