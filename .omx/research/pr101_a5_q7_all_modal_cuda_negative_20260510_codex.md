# PR101 A5 q7-all exact CUDA result — negative classification (2026-05-10)

## Summary

The PR101 A5 q7-all packet was dispatched through the canonical Modal T4
auth-eval wrapper and completed the exact CUDA path:

`archive.zip -> inflate.sh -> upstream/evaluate.py --device cuda`

The result is a CUDA negative relative to the current exact CUDA frontier. It
must not be promoted, submitted, or redispatched as a score-lowering candidate.

## Custody

- Lane id: `pr101_a5_trust_q7_all_exact_cuda`
- Modal job id: `pr101_a5_trust_q7_all_exact_cuda_modal_20260510T193821Z`
- Local artifact dir:
  `experiments/results/modal_auth_eval/pr101_a5_trust_q7_all_exact_cuda_modal_20260510T193821Z/`
- Candidate archive:
  `experiments/results/pr101_frame_conditional_runtime_packet_trust_q7_all_20260509_codex/packet/archive.zip`
- Archive SHA-256:
  `39dbfd05d4861c6c5ea12e7bfc8fba17e8249dcc761e9c44943eeba8d56c6ade`
- Archive bytes: `177928`
- GPU: Modal Tesla T4 (`gpu_t4_match=true`)
- Dispatch claim terminal status: `completed_contest_cuda_auth_eval_negative`

## Exact CUDA result

- `score_recomputed_from_components`: `0.23488497704294406`
- `final_score`: `0.23`
- `avg_posenet_dist`: `0.00017505`
- `avg_segnet_dist`: `0.00074571`
- `archive_size_bytes`: `177928`
- `n_samples`: `600`

## Classification

This is legitimate exact CUDA score movement, but in the wrong direction:
`0.23488497704294406` is worse than the active exact CUDA anchor
`0.20898105278`.

The measured configuration is falsified for score lowering. This does not kill
the A5/score-marginal/channel-allocation family. It specifically invalidates
this PR101 A5 q7-all packet on Modal T4 CUDA.

## Implications

- Do not treat the macOS CPU advisory result as submission evidence.
- Do not redispatch this archive/runtime identity.
- Continue A5 only through CUDA-calibrated search, a fresh byte-closed packet,
  or a score-domain channel allocation variant with runtime-custody proof.
- The next A5 target remains the channel-qbits DP qsum200 readiness fix only if
  the readiness artifact gap is closed and the exact-ready audit passes.

## Reactivation criteria

Reopen this lane only with at least one of:

- a different archive SHA-256 and explicit charged-byte diff;
- a different runtime-tree SHA-256 with `score_affecting_runtime_changed=true`
  and runtime-consumption proof;
- a CUDA-calibrated drift model showing why the failed q7-all packet is not
  representative of the new candidate;
- exact CUDA evidence from a new A5 variant that beats `0.20898105278`.
