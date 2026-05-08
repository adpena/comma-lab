# CPU/CUDA auth-eval drift analysis

generated_at_utc: `2026-05-08T12:46:35Z`
evidence_grade: `external_github_pr_comment_analysis`
score_claim: `false`
mechanism_claim_proven: `false`

| PR | CUDA score | CPU score | pose ratio | seg ratio | pose gap share |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 100 | 0.228269572711 | 0.195385423975 | 4.995 | 1.173 | 69.7% |
| 101 | 0.226354458744 | 0.192845012702 | 5.205 | 1.184 | 69.3% |
| 102 | 0.228390831180 | 0.195376176526 | 5.014 | 1.173 | 69.8% |
| 103 | 0.227764851625 | 0.194880702889 | 4.995 | 1.173 | 69.7% |
| 105 | 0.230437255695 | 0.197973979344 | 4.973 | 1.157 | 70.6% |

## Summary

- paired PR count: `5`
- median pose distortion CUDA/CPU ratio: `4.995`
- median seg distortion CUDA/CPU ratio: `1.173`
- median CPU pose distortion: `3.443e-05`
- public-comment pose tau hypothesis: `0.0058677082`

## Guardrails

- Paired public comments support a device-axis drift signal when CPU and CUDA rows exist for the same PR.
- The mechanism is not proven by comments alone; T4-specific TF32 claims are suspect because T4 is not an Ampere TF32 GPU.
- Use this as a CPU-leaderboard reproduction hypothesis until paired Linux CPU and CUDA exact eval JSONs exist.
- Do not use public-comment drift to promote, rank, kill, or retire internal CUDA lanes.

## Safe next actions

- Run paired dual-device exact eval plans on Linux x86_64 CPU and T4-equivalent CUDA for PR101, PR102, PR103, and PR105.
- Fit CPU-score predictor only from exact paired JSON artifacts, not macOS CPU or rounded comments.
- If exact pairs confirm a stable CPU pose floor, test a pose-floor/Huber-style loss as CPU-axis research while preserving CUDA promotion gates.
