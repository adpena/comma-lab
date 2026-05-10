# PR101 A5 channel-qbits DP qsum200 exact CUDA result — negative classification (2026-05-10)

## Summary

The `pr101_a5_channel_qbits_dp_qsum200` packet was unblocked through the A5
readiness contract, passed strict no-auth compliance, and was dispatched through
the canonical Modal T4 auth-eval wrapper.

The exact CUDA result is negative relative to the current exact CUDA frontier.
It must not be promoted, submitted, or redispatched as a score-lowering
candidate.

## Custody

- Lane id: `pr101_a5_channel_qbits_dp_qsum200_exact_cuda`
- Modal job id:
  `pr101_a5_channel_qbits_dp_qsum200_exact_cuda_modal_20260510T194943Z`
- Local artifact dir:
  `experiments/results/modal_auth_eval/pr101_a5_channel_qbits_dp_qsum200_exact_cuda_modal_20260510T194943Z/`
- Candidate archive:
  `experiments/results/pr101_a5_channel_qbits_dp_qsum200_20260509_codex/packet/archive.zip`
- Archive SHA-256:
  `efc0466bc38edb9cb57193d5f43e4d4fbf2d993cd41069f2c3700aa4bedbfeae`
- Archive bytes: `178014`
- Readiness artifact:
  `experiments/results/pr101_a5_channel_qbits_dp_qsum200_20260509_codex/readiness.channel_qbits.json`
- Strict no-auth compliance artifact:
  `experiments/results/pr101_a5_channel_qbits_dp_qsum200_20260509_codex/strict_pre_submission_compliance.no_auth.json`
- GPU: Modal Tesla T4 (`gpu_t4_match=true`)
- Dispatch claim terminal status: `completed_contest_cuda_auth_eval_negative`

## Exact CUDA result

- `score_recomputed_from_components`: `0.2339625235094184`
- `final_score`: `0.23`
- `avg_posenet_dist`: `0.0001742`
- `avg_segnet_dist`: `0.00073693`
- `archive_size_bytes`: `178014`
- `n_samples`: `600`

## Classification

This is legitimate exact CUDA score movement, but in the wrong direction:
`0.2339625235094184` is worse than the active exact CUDA anchor
`0.20898105278`.

The measured configuration is falsified for score lowering. This does not kill
the broader A5/channel-allocation family. It specifically invalidates the
channel-qbits DP qsum200 packet and shows the latent-domain DP proxy does not
transfer to T4 CUDA score for this operating point.

## Contract fix made before dispatch

The readiness blocker exposed a real contract mismatch: the A5 anchor recorded
the inner member SHA as `input_archive_sha256`, while the channel schedule
correctly recorded the outer `archive.zip` SHA. The readiness contract now
takes source archive custody from the candidate packet manifest when present,
while keeping q-bit side-info and latent-payload SHAs separate.

## Implications

- Do not treat latent-domain channel DP or macOS/Kaggle proxy positives as
  score authority.
- Do not redispatch this archive/runtime identity.
- Future A5 work needs CUDA-calibrated objectives or a different mechanism,
  such as score-domain channel allocation, train-time q-bit noise, or a
  runtime-sidecar transform with score-gradient evidence.
- The PR101 proxy/A5 cluster now has repeated T4 negatives; near-term score
  lowering should prioritize T1/Ballé recovery, PR95 parity training, typed
  PacketIR transforms, and PR103/PR106 grammar-preserving entropy work.

## Reactivation criteria

Reopen only with at least one of:

- a different archive SHA-256 and explicit charged-byte diff;
- a different runtime-tree SHA-256 with `score_affecting_runtime_changed=true`
  and runtime-consumption proof;
- a CUDA-calibrated score-domain channel allocation objective;
- exact CUDA evidence from a new A5 variant that beats `0.20898105278`.
