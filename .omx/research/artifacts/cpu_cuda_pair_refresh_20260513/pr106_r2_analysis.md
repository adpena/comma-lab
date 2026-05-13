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
- mechanism_class: `different_raw_outputs_runtime_or_inflate_drift`
- raw_output_pairing_status: `different_inflated_outputs`
- same_archive_sha256: `True`
- same_archive_bytes: `True`
- same_runtime_tree_sha256: `False`
- same_inflated_output_aggregate_sha256: `False`

| Axis | Score | Pose | Seg | Archive bytes |
| --- | ---: | ---: | ---: | ---: |
| CPU | 0.228092484052 | 0.00016402 | 0.00063196 | 186822 |
| CUDA | 0.206645986798 | 3.236e-05 | 0.0006426 | 186822 |

## CUDA Minus CPU

- score: `-0.021446497253930052`
- pose_term: `-0.022510497253930027`
- seg_term: `0.0010639999999999955`
- rate_term: `0.0`

## Individual Axis Blockers

- none

## Blockers

- `cpu_cuda_runtime_tree_sha256_mismatch`

## Mechanism Blockers

- none

## Interpretation

- Negative score gap means CUDA scored lower than CPU for this exact pair; positive means CPU scored lower.
- Do not infer global CPU-better or CUDA-better behavior from one row.
- If raw outputs differ, runtime/inflate device behavior is part of the mechanism.
- If raw outputs match but scores differ, localize through GT loader and scorer-kernel xray probes.
