# PR103 lc_ac Arithmetic Schema Manifest

- planning_only: `true`
- score_claim: `false`
- dispatch_attempted: `false`
- ready_for_exact_eval_dispatch: `false`
- ready_for_schema_review: `true`

## Source Archive

- path: `experiments/results/public_pr_intake_full/public_pr103_intake_20260505_auto/archive.zip`
- bytes: `178223`
- sha256: `31881b2d23d027e6619f2d8df2fe35d4d207d08882ec673d6c1b7ff119f18c30`
- member: `x` / `178123` bytes

## Merged Arithmetic Stream

- bytes: `153856`
- sha256: `08f0a219b395e6783c522bacc8239f01dcb3d27d9f7f8d2291a7478d04859de7`
- decoded_symbols: `237561`
- decoder_maybe_exhausted: `true`
- reencoded_byte_identical: `true`

## Next Targets

| rank | stream | role | symbols | model gap bytes | model floor bytes |
|---:|---|---|---:|---:|---:|
| 1 | `stem.weight` | `ac_weight_tensor` | 48384 | 46 | 31627 |
| 2 | `blocks.1.weight` | `ac_weight_tensor` | 46656 | 45 | 32478 |
| 3 | `blocks.0.weight` | `ac_weight_tensor` | 46656 | 33 | 32340 |
| 4 | `blocks.2.weight` | `ac_weight_tensor` | 34992 | 25 | 25462 |
| 5 | `blocks.3.weight` | `ac_weight_tensor` | 19440 | 9 | 14031 |

## Old/New Archive Pair

- old: `31881b2d23d027e6619f2d8df2fe35d4d207d08882ec673d6c1b7ff119f18c30`
- new: ``
- closed: `false`

## Blockers

- `replay_fidelity:public_leaderboard_score_mismatch`
- `candidate_archive_missing`
- `old_new_archive_sha256_pair_missing`
- `candidate_runtime_adapter_missing`
- `strict_pre_submission_compliance_json_missing`
- `lane_dispatch_claim_missing`
- `exact_cuda_auth_eval_missing`
