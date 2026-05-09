# A5 Trust-Region q6-low50 macOS CPU Advisory Result

Date: 2026-05-09

Scope: local macOS CPU advisory eval for a conservative A5 frame-conditional
q-bit schedule. This is not contest-CPU, not contest-CUDA, and not promotion
evidence.

## Candidate

Schedule:

- Artifact: `reports/a5_score_marginal_trust_region_qbits_20260509.json`
- SHA-256:
  `41891d628e7b743da56e6ed4196566e556a48ec9018d3fa13ec162b01124ec10`
- Policy: keep `q=8` for the highest 50% score-marginal pairs; set `q=6`
  for the lowest 50%.
- q-bit distribution: `6:300`, `8:300`
- q-bit mean: `7.0`
- q-bit SHA-256:
  `f90f23e707414ab23e8456a8f8fe5a4cb7c7d8cf989e64f0ef588ab93699feef`
- Source score-marginal manifest:
  `experiments/results/pr101_frame_conditional_runtime_packet_20260508_codex/per_pair_score_marginals.advisory.json`

Runtime packet:

- Archive bytes: `177,928`
- Archive SHA-256:
  `83e85a0918c5db6eba5d824b2c9ab0a9b7a7f47dd7d03fbbbbbb25bcbf99807e`
- Member `x` SHA-256:
  `b37bc5cd02e65f03695b120e1b8f89f443dc3592ca33b28de3fc54e199f27c4e`
- Runtime tree SHA-256:
  `9084d00f236755d00974aa025aee2bbdbac6bfdc5df84094c83eaf28e8afe9d5`
- q-bit side-info SHA-256:
  `4629fbdb1f9d442744a29a4ad780f310d7d2fa1c7ad0df94eee471d89391cc0f`
- Materialized wire-contract SHA-256:
  `056835b18ae97d4f7e032f364b5a44d608e0e3e8eea63b7ef7374269d8ce9d80`

Packet readiness:

- `ready_for_exact_eval_after_lane_claim=true`
- `readiness_blockers=[]`
- Remaining dispatch blockers:
  - `requires_level2_dispatch_claim_before_exact_eval`
  - `requires_exact_cuda_auth_eval_before_score_promotion`

## Advisory Eval

Command:

```bash
PYTHON=/Users/adpena/Projects/pact/.venv/bin/python \
.venv/bin/python experiments/contest_auth_eval.py \
  --archive experiments/results/pr101_frame_conditional_runtime_packet_trust_q6_low50_20260509_codex/packet/archive.zip \
  --inflate-sh experiments/results/pr101_frame_conditional_runtime_packet_trust_q6_low50_20260509_codex/packet/inflate.sh \
  --upstream-dir upstream \
  --device cpu \
  --work-dir experiments/results/pr101_frame_conditional_runtime_packet_trust_q6_low50_20260509_codex/macos_cpu_advisory_work \
  --json-out experiments/results/pr101_frame_conditional_runtime_packet_trust_q6_low50_20260509_codex/contest_auth_eval.macos_cpu_advisory.json \
  --inflate-timeout 1800 \
  --evaluate-timeout 5400 \
  --keep-work-dir
```

Result:

- Evidence grade: `[macOS-CPU advisory]`
- Hardware blocker: `contest_cpu_requires_linux_x86_64`
- Canonical score: `0.21336553243503031`
- Pose distortion: `0.00004139`
- Seg distortion: `0.00074546`
- Rate term: `0.118475`
- Archive bytes: `177,928`
- Samples: `600`
- Inflate elapsed: `35.3109s`
- Evaluate elapsed: `410.1826s`
- Total elapsed: `445.4973s`
- Auth eval JSON SHA-256:
  `39ddeac44549ab285d1a7c090303308ad2cf394437963cea30537da0dc160892`

Review packet:

- `.omx/research/artifacts/a5_trust_q6_low50_result_review_20260509_codex.json`
- SHA-256:
  `50f6819c86907081b5e6f63ef0472bda7ddf6f23e7542dbee6a26a4aa66b8dd4`

Evidence row:

- `reports/a5_trust_q6_low50_macos_advisory_evidence_row_20260509.json`
- SHA-256:
  `8870bb46d1868ddc66e81f54113c83dfae776d2a830807a266ae1e6f4802b567`
- Appended to `reports/cathedral_autopilot_evidence.jsonl` as non-CUDA,
  non-promotable review evidence.

## Classification

Measured-config advisory negative.

The trust-region schedule avoided the catastrophic A5 collapse (`1.937884`),
but it still loses to the PR101/PR107 CPU advisory band. The loss is dominated
by SegNet:

- Score term from seg: `0.074546`
- Score term from pose: `0.0203445`
- Score term from rate: `0.118475`

The byte win is only `216 B` versus the PR101 `178,144 B` baseline, while seg
drift adds roughly `0.014-0.016` score points versus the medal/advisory band.
This confirms that scalar per-pair score marginals are not enough for A5 at
this trust region; the next schedule must be SegNet-boundary-aware or much
tighter.

## Reactivation Criteria

1. Tighten the q-bit trust region, for example `q7 low50` or `q6 low25`, and
   require advisory SegNet not to dominate before exact eval spend.
2. Replace scalar score marginals with SegNet-boundary or per-pair component
   marginals.
3. Run exact contest-CUDA and contest-CPU only after macOS advisory is
   non-collapsed and a lane claim is active.

## Claim Closure

Lane: `a5_trust_q6_low50_macos_cpu_advisory`

Job: `local:a5-trust-q6-low50-macos-cpu-20260509T0042Z`

Terminal claim status: `completed_advisory_negative`.
