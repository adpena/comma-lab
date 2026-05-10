# PR103 histogram -8B exact CUDA packet result (2026-05-10)

## Supersession note - apples-to-apples classification

2026-05-10 follow-up review reclassifies this from a broad
`measured exact-CUDA negative` lane verdict to a narrower
`packet-specific exact-CUDA regression; method verdict indeterminate`.

The Modal T4 score below is a real exact CUDA result for the rebuilt candidate
packet, but the earlier interpretation overreached: decoded state/latent parity
does not prove full-frame inflate output parity, and this packet was not scored
side-by-side with a source PR103 packet through the same rebuilt runtime adapter.
Because public PR103 was accepted and the repo has known CPU/CUDA-axis drift,
the safe classification is:

- exact CUDA score for this packet: valid custody result;
- source-vs-candidate method conclusion: blocked until same-runtime source
  replay or full-frame inflate output parity exists;
- dispatch status: blocked for future variants by
  `full_frame_inflate_output_parity_missing` plus normal claim/CUDA gates.

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
- strict pre-submission compliance at dispatch time:
  `.omx/research/pr103_arithmetic_transform_plans_20260510_codex/combined_beam_candidate/packet/pre_submission_compliance.json`
  with `passed=true`
- post-review regenerated packet manifest:
  `.omx/research/pr103_arithmetic_transform_plans_20260510_codex/combined_beam_candidate/packet_manifest.json`
  now records `full_frame_inflate_output_parity_missing`
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
- packet classification: `exact-CUDA regression for this rebuilt packet`
- method classification: `indeterminate-harness-or-runtime-mismatch until same-runtime source replay or full-frame output parity`

## Interpretation

The byte transform is real and runtime-consumed by the rebuilt adapter packet,
and that packet did not beat the source PR103 CUDA replay number. This is not
enough to conclude that the arithmetic/histogram method is intrinsically
negative. The valid conclusion is narrower: do not promote this packet, and do
not use it as a CUDA-frontier or submission artifact.

Future PR103 variants must score source and candidate through the same runtime
contract, or prove full-frame inflate output parity before interpreting tiny
component deltas. This prevents decoded-tensor parity from being mistaken for
scored-frame parity.

Retain the infrastructure: the reusable runtime-adapter, packet builder,
decoder-state parity proof, and strict compliance path are useful for the next
algorithmic PR103 pass. Per the algorithmic review, the next PR103 work should
be a DP/Lagrangian histogram optimizer over exact range bytes plus Brotli
sideband cost, not more arbitrary coordinate sweeps.

## Custody Caveat

The harvested Modal `contest_auth_eval.json` recorded `pact_commit` as a remote
git error because the Modal image did not include `.git`. The Modal provenance
does record archive SHA, command, hardware, component fields, and the remote
runtime tree SHA
`e85c67ce83eeda7c2c8b7a13aa6895dce49478421edcd74ee24cb10d399d911d`.
This is sufficient to preserve the packet result, but not sufficient for a
paper/promotion artifact or a method-negative verdict. Future Modal runs inject
`PACT_SOURCE_COMMIT` from the local submitter.

## Claim closure

Terminal claim row recorded with status
`completed_contest_cuda_auth_eval_negative`.
