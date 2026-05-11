# CPU/CUDA exact-pair mechanism analysis

generated_at_utc: `2026-05-11T03:18:52Z`
evidence_grade: `paired_exact_auth_eval_mechanism_diagnostic`
score_claim: `false`
promotion_eligible: `false`

JSON artifact:
`.omx/research/artifacts/cpu_cuda_drift_exact_pr103_pr106_20260511_codex/analysis.json`

## Pair

- valid_for_mechanism_analysis: `False`
- valid_for_pair_score_analysis: `True`
- mechanism_class: `same_archive_runtime_raw_outputs_unmeasured`
- raw_output_pairing_status: `raw_output_manifest_missing`
- same_archive_sha256: `True`
- same_archive_bytes: `True`
- same_runtime_tree_sha256: `True`
- same_inflated_output_aggregate_sha256: `None`

| Axis | Score | Pose | Seg | Archive bytes |
| --- | ---: | ---: | ---: | ---: |
| CPU | 0.229657686265 | 0.000164 | 0.00065592 | 185578 |
| CUDA | 0.208983075582 | 3.36e-05 | 0.00067084 | 185578 |

## CUDA Minus CPU

- score: `-0.02067461068280979`
- pose_term: `-0.022166610682809812`
- seg_term: `0.0014919999999999933`
- rate_term: `0.0`

## Blockers

- none

## Mechanism Blockers

- `raw_output_manifest_missing`

## Interpretation

- Negative score gap means CUDA scored lower than CPU for this exact pair; positive means CPU scored lower.
- Do not infer global CPU-better or CUDA-better behavior from one row.
- If raw outputs differ, runtime/inflate device behavior is part of the mechanism.
- If raw outputs match but scores differ, localize through GT loader and scorer-kernel xray probes.

## Codex adversarial verdict

This paired exact artifact directly falsifies the broad claim that HNeRV
leaderboard-family packets always score better on CPU than CUDA. For this
same-archive, same-runtime-content pair, CUDA is better by `0.02067461068280979`
score, almost entirely through PoseNet (`-0.022166610682809812` pose-term gap)
while SegNet is slightly worse on CUDA (`+0.0014919999999999933`).

The result is not a ranking substitution and not a new score claim. It is a
mechanism diagnostic. The unresolved blocker is raw inflated output custody:
the current pair lacks CPU/CUDA raw-output aggregate hashes, so the next
score-lowering-relevant action is to rerun or harvest paired inflate outputs
with manifests. If raw outputs match, the gap localizes to loader/scorer
device behavior; if raw outputs differ, the gap localizes to runtime/inflate
device behavior.
