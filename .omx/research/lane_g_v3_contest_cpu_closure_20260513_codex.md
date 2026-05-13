# lane_g_v3 contest-CPU closure (2026-05-13)

## Verdict

`lane_g_v3` has a harvested `[contest-CPU GHA Linux x86_64]` closure artifact.
This is public-leaderboard CPU reproduction evidence, not a new contest-CUDA
frontier and not a score-lowering result.

- Result path:
  `experiments/results/gha_cpu_eval/lane_g_v3_retry7_25772267506/contest_cpu_eval-lane_g_v3-25772267506/contest_auth_eval.json`
- Result JSON SHA-256:
  `e923525e11a4f332838b7899a09d96b1bc1a146c618b0f9649dc7ba5897a2ff4`
- GitHub Actions run:
  `25772267506`
- Workflow:
  `contest_cpu_eval.yml`
- Runner:
  GitHub Actions `ubuntu-latest`, Linux x86_64, CPU
- `score_axis`:
  `contest_cpu_gha`
- `score_claim_axis`:
  `contest_cpu`
- Evidence grade:
  `contest-CPU`
- Allowed uses:
  public-leaderboard reproduction, CPU/CUDA drift diagnosis, same archive/runtime
  medal-band context
- Forbidden uses:
  contest-CUDA promotion, rank-or-kill decisions, score-lowering claim, CPU to
  CUDA extrapolation

## Exact result

Components harvested from `report.txt` and recomputed from
`contest_auth_eval.json`:

```text
avg_segnet_dist        = 0.00400702
avg_posenet_dist       = 0.00305290
archive_size_bytes     = 694,074
n_samples              = 600
score_seg_contribution = 0.400702
score_pose_contribution= 0.17472549899771356
score_rate_contribution= 0.4621553870293179
canonical_score        = 1.0375828860270313 [contest-CPU GHA Linux x86_64]
display_score          = 1.04
```

Formula check:

```text
100 * 0.00400702
+ sqrt(10 * 0.00305290)
+ 25 * 694074 / 37545489
= 1.0375828860270315
```

The tiny final-digit difference versus the stored `canonical_score` is floating
point representation only.

## Custody

- Archive path in GHA:
  `/home/runner/work/comma-lab/comma-lab/experiments/results/lane_g_v3_landed/archive_lane_g_v3.zip`
- Archive SHA-256:
  `9b20bdfca246d8e32cc19da966c84cdae7e34f6b247161d107ec43cb9ef6870b`
- Archive bytes:
  `694074`
- Inflate script:
  `submissions/robust_current/inflate.sh`
- Inflate script SHA-256:
  `ed3cef1ab7e20dc06b7f6bad8e4f7b0d9b685d29e626873b5089fb76336bc40a`
- Runtime tree SHA-256:
  `ce34af1e2022fa06dc04cc9f7de8db8e5070a4230bafeaac9958e1e52d9450d4`
- Runtime content tree SHA-256:
  `3fd01dc820431f104ab690bd23a7b7c0ae3af08c1d217e44a8ce7e077f9ff410`
- Inflated raw aggregate SHA-256:
  `6f88aa4510cea64ffccc3cdd6a7e8a9f4ccba8e092783698f3582eb16f8ecc78`
- Inflated `0.raw` SHA-256:
  `c0ccf805d18c4cd98213a6820f9723f1a3c7c11920efc47ac76540faaee46baa`
- Upstream commit:
  `11ad728f563d8970929e8947a1cf6124ee6303e4`
- Pact commit evaluated by GHA:
  `7b8f7db0e9620b45ce9ab36f345f377a46570593`
- Torch:
  `2.5.1+cpu`
- CUDA available:
  `false`
- MPS available:
  `false`

## Classification

The artifact closes the CANON-1.I / lane_g_v3 `[contest-CPU]` reproduction item
as harvested. It does not improve the current score frontier and it does not
change any CUDA promotion state.

Important evidence labels:

- `score_claim=false`
- `score_claim_valid=false`
- `promotion_eligible=false`
- `rank_or_kill_eligible=false`
- `exact_cuda_eval_complete=false`
- `paired_cuda_score=missing-not-extrapolated`
- `cpu_leaderboard_reproduction_eligible=true`

The only accepted posterior update is the CPU-axis posterior row:

```json
{
  "schema": "contest_cpu_eval_posterior_update_v1",
  "lane_id": "lane_g_v3",
  "axis": "cpu",
  "evidence_tag": "[contest-CPU GHA Linux x86_64]",
  "archive_sha256": "9b20bdfca246d8e32cc19da966c84cdae7e34f6b247161d107ec43cb9ef6870b",
  "score_value": 1.0375828860270313,
  "accepted": true,
  "posterior_n_anchors_after": 1
}
```

## Review notes

The older local `experiments/results/lane_g_v3_gha_cpu_eval_20260512T204212Z`
dispatch metadata is pre-harvest and points at earlier failed/cancelled runs.
It is not the terminal evidence for this closure. The terminal evidence is the
retry7 harvested artifact under `gha_cpu_eval/lane_g_v3_retry7_25772267506/`.

No active dispatch claim exists at the time of this ledger:

```text
CLAIM_SUMMARY active=0
```

## Next score-lowering implication

Do not spend more time on `lane_g_v3` as a score-lowering target unless a new
byte-closed transform or exact CUDA pairing is proposed. The current highest-EV
near-term work remains PR106/R2 and A1/HNeRV-class byte-closed PacketIR work:

1. validate PR106/R2 `[contest-CPU]` and `[contest-CUDA]` as separate axes;
2. require consumed-byte PacketIR/runtime proof before sidecar byte claims;
3. only dispatch exact-eval packets after lane-claim registration;
4. preserve CPU/CUDA drift evidence without extrapolating one axis into the
   other.
