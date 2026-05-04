# PR85 -> HPM1 Residual Prefix Trajectory

- schema: `pr85_hpm1_residual_prefix_trajectory_v1`
- planning_only: `true`
- score_claim: `false`
- dispatch_unlocked: `false`
- exact_eval: `not_run`

| frames | raw token bytes | HPM1 segment bytes | HPM1 token bytes | marginal segment B/frame | elapsed s | segment sha256 |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 196608 | 28695 | 404 | n/a | 59.97 | `0539b0d3b2179b2a` |
| 2 | 393216 | 40203 | 11912 | 11508.0 | 122.205 | `1e1edc2cfff3efb7` |

## Custody

- raw token source: `experiments/results/public_pr85_intake_20260503_codex/qma9_token_source/pr85_qma9_tokens_u8_storage_order.bin`
- raw token sha256: `c1c47434fd1e6c876cb3e44910f5ab2e124285d9dba2f300bcf322d03fb8bb5a`
- normalized NHW sha256: `0344fcfc39e683f21a71db1085a8697a94c4606f91f883362e9acc02fc7b5b45`
- PR91 archive: `experiments/results/public_pr91_intake_20260504_codex/archive.zip`

## Safety

- This artifact is planning-only and local-only.
- It does not claim score, unlock dispatch, run exact eval, or launch remote work.
