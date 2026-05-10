# PR103 histogram -8B exact CUDA classification (2026-05-10)

## Scope

Candidate:
`.omx/research/pr103_arithmetic_transform_plans_20260510_codex/combined_beam_candidate/packet/archive.zip`

- archive SHA-256:
  `2427cbb7f68e8e3bcf1e989eee0cf511bff5994a5e856b500bfca3c95ca181d8`
- archive bytes: `178215`
- source PR103 replay archive SHA-256:
  `31881b2d23d027e6619f2d8df2fe35d4d207d08882ec673d6c1b7ff119f18c30`
- source PR103 replay score [contest-CUDA T4]:
  `0.2277649714224471`

## Local gates before dispatch

- runtime adapter built:
  `.omx/research/pr103_arithmetic_transform_plans_20260510_codex/combined_beam_candidate/runtime_adapter_manifest.json`
- decoder-state parity: `passed=true`
- strict pre-submission compliance:
  `.omx/research/pr103_arithmetic_transform_plans_20260510_codex/combined_beam_candidate/packet/pre_submission_compliance.json`
  with `passed=true`
- Level-2 dispatch claim opened:
  `pr103_histogram_8b_packet_exact_cuda` /
  `pr103_histogram_8b_packet_exact_cuda_modal_20260510T221146Z`

## Exact CUDA result

Artifact directory:
`experiments/results/modal_auth_eval/pr103_histogram_8b_packet_exact_cuda_modal_20260510T221146Z`

Result:

| field | value |
|---|---:|
| evidence | `[contest-CUDA] Modal T4` |
| score_recomputed_from_components | `0.22777267708207616` |
| final_score rounded | `0.23` |
| avg_posenet_dist | `0.00017199` |
| avg_segnet_dist | `0.00067635` |
| archive_size_bytes | `178215` |
| gpu_model | `Tesla T4` |
| gpu_t4_match | `true` |
| path validation | `passed=true` |

Delta versus source PR103 T4 replay:

- score delta: `+0.00000770565962906`
- archive byte delta: `-8`
- classification: `measured exact-CUDA negative`

## Interpretation

The byte transform is real and runtime-consumed, but it is not score-lowering
under exact CUDA. The small rate gain is outweighed by a tiny component drift.
Do not promote this packet and do not keep widening local q8 coordinate beams
as a score lane.

Retain the infrastructure: the reusable runtime-adapter, packet builder,
decoder-state parity proof, and strict compliance path are useful for the next
algorithmic PR103 pass. Per the algorithmic review, the next PR103 work should
be a DP/Lagrangian histogram optimizer over exact range bytes plus Brotli
sideband cost, not more arbitrary coordinate sweeps.

## Claim closure

Terminal claim row recorded with status
`completed_contest_cuda_auth_eval_negative`.
