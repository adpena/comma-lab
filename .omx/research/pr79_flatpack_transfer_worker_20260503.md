# PR79 Flatpack Transfer Worker - 2026-05-03

## Scope

Goal: test whether PR73 `emir_flatpack` style single-stream flat packing has
positive EV when transferred to the current PR79/C102 qpose14/QZS3 fixed-slice
family. This worker did not dispatch remote GPU jobs and makes no exact score
claim.

Allowed runtime-closed container:

- `p = Brotli(RPK1 JSON member table + raw runtime members)`.

Skipped compressors:

- Zstd: not attempted because `robust_current` has no top-level `p` Zstd
  inflate closure.
- LZMA2: not attempted because `robust_current` has no top-level `p` LZMA2
  inflate closure.

## Implementation

Created:

- `experiments/build_pr79_flatpack_transfer_candidates.py`
- `src/tac/tests/test_build_pr79_flatpack_transfer_candidates.py`

Also hardened the existing PR75/minp micro-packer for the live PR79 fixed raw
slice:

- `experiments/build_pr75_minp_lossless_micro_candidates.py`
  - accepts PR79 fixed payload length `277288` with slices
    `219472/55756/1162/898`;
  - accepts public PR79 SHA
    `01dc02badf851d99108fd92c570271f36f74cc5424c6d2a8f1b499cb4d1c3446`;
  - fixes duplicate pair/tile reporting for PR79 action streams.

## Anatomy

Source comparison from
`experiments/results/pr79_flatpack_transfer_worker_20260503/candidate_matrix.json`:

- C102 anchor:
  `experiments/results/lightning_batch/exact_eval_c091_next_cem_pose_waterfill_top192_s0125_m06_t4_20260503T1238Z/archive.zip`
  bytes `276485`, SHA
  `79091f2c3f0c30ef3ca512808f3adc0306010e7f57fed3a09b3664c16fea4ea8`.
- PR79 public:
  `experiments/results/top_submission_reverse_engineering_20260503_pr79/archive.zip`
  bytes `277388`, SHA
  `01dc02badf851d99108fd92c570271f36f74cc5424c6d2a8f1b499cb4d1c3446`.
- C102 and PR79 decoded stream SHA equalities:
  masks equal, renderer equal, optimized poses differ, seg tile actions differ.

## Flatpack Byte Screen

Command:

```bash
.venv/bin/python experiments/build_pr79_flatpack_transfer_candidates.py \
  --output-dir experiments/results/pr79_flatpack_transfer_worker_20260503 \
  --force --fast-brotli-grid
```

Notes:

- A full Brotli parameter grid was started first but stopped before output
  because it was too slow for the current turn.
- The completed screen is a bounded fast-grid screen over all RPK1 member
  order permutations for the C102 anchor, PR79 public archive, and four
  highest-ranked existing PR79-on-C102 local candidates.
- Every emitted archive passed local runtime RPK1 parse and decoded-byte parity
  against its source archive.

Top flatpack rows:

| candidate | bytes | delta vs source | delta vs C102 | break-even component delta vs C102 | dispatch |
|---|---:|---:|---:|---:|---|
| `replace_pr79_pose_safe_top16_on_c102_p6_rpk1_single_brotli_flatpack` | `276660` | `+416` | `+175` | `0.00011652531679638` | no |
| `replace_pr79_exact_positive_top16_on_c102_p6_rpk1_single_brotli_flatpack` | `276664` | `+410` | `+179` | `0.00011918875260886868` | no |
| `replace_pr79_pair_opportunity_top32_on_c102_p6_rpk1_single_brotli_flatpack` | `276746` | `+414` | `+261` | `0.00017378918676488674` | no |
| `replace_pr79_pair_opportunity_top64_on_c102_p6_rpk1_single_brotli_flatpack` | `276843` | `+429` | `+358` | `0.00023837750521773737` | no |
| `c102_anchor_rpk1_single_brotli_flatpack` | `276987` | `+502` | `+502` | `0.00033426119446733` | no |
| `pr79_public_fixed_slices_rpk1_single_brotli_flatpack` | `278037` | `+649` | `+1552` | `0.00103341309524561` | no |

Conclusion: PR73-style RPK1 single-Brotli flatpack is byte-regressive for this
family under the runtime-closed schema. Do not dispatch these flatpack rows
without a separate non-byte component rationale.

## PR79 Fixed-Slice Micro-Packer Compatibility

User-reported failing command after hardening:

```bash
.venv/bin/python experiments/build_pr75_minp_lossless_micro_candidates.py \
  --public-archive experiments/results/top_submission_reverse_engineering_20260503_pr79/archive.zip \
  --output-dir experiments/results/pr79_minp_lossless_micro_packer_20260503_codex \
  --force
```

Result: completed successfully. Candidate matrix:

- `experiments/results/pr79_minp_lossless_micro_packer_20260503_codex/candidate_matrix.json`

Top rows:

| candidate | bytes | delta vs C089 | parse | dispatch gate |
|---|---:|---:|---|---|
| `public_renderer_c089_p6_lossless_stream_resweep` | `276124` | `-218` | passed | renderer transplant pose-safety preflight required |
| `c089_raw_no_header_fixedslice_probe` | `276321` | `-21` | parser rejected | non-dispatchable |
| `c089_p6_lossless_stream_resweep` | `276333` | `-9` | passed | exact CUDA auth eval required after lane claim |
| `c089_p6_action_resweep` | `276341` | `-1` | passed | exact CUDA auth eval required after lane claim |

This is not an exact score claim. The best byte row changes the decoded
renderer stream versus C089, so it must pass
`experiments/preflight_renderer_transplant_pose_safety.py` against exact source
and candidate SHA pairs before any exact eval dispatch is valid.

## Verification

Passed:

```bash
.venv/bin/python -m py_compile \
  experiments/build_pr79_flatpack_transfer_candidates.py \
  experiments/build_pr75_minp_lossless_micro_candidates.py

.venv/bin/python -m pytest \
  src/tac/tests/test_build_pr79_flatpack_transfer_candidates.py
```

Focused pytest result: `2 passed`.

## Evidence Grade

All results here are `empirical_byte_screen_only`. No CUDA auth eval was run,
no remote GPU job was dispatched, and no promotion/ranking claim should be made
from these byte-only matrices.

## Renderer-Transplant Pose-Safety Gate For PR79-Aware Micro-Packer

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -u \
  experiments/preflight_renderer_transplant_pose_safety.py \
  --source-archive experiments/results/lightning_batch/exact_eval_c067_pr75_qp1_top40_p6_t4_awsfix1_20260503T0630Z/archive.zip \
  --candidate-archive experiments/results/pr79_minp_lossless_micro_packer_20260503_codex/public_renderer_c089_p6_lossless_stream_resweep/archive.zip \
  --output-json experiments/results/pr79_minp_lossless_micro_packer_20260503_codex/public_renderer_c089_p6_lossless_stream_resweep/renderer_transplant_pose_safety_preflight.json \
  --max-pairs 32
```

Exact SHA pair:

- source C089/c067_pr75_qp1_top40_p6:
  `276342` bytes,
  `0ec53e5b871149ed6eea56c0b9bcca3baec998d5bfad4f371979e0c90e62cea8`
- candidate public_renderer_c089_p6_lossless_stream_resweep:
  `276124` bytes,
  `337b04040bee1316375bae8b2cfc2f08acddc235e4ee02d37ecfd62c0c831d95`

Artifact:

- `experiments/results/pr79_minp_lossless_micro_packer_20260503_codex/public_renderer_c089_p6_lossless_stream_resweep/renderer_transplant_pose_safety_preflight.json`

Result: fail closed. Archive contracts and transplant contract passed
(`masks.mkv`, `optimized_poses.qp1`, and `seg_tile_actions.bin` unchanged;
`renderer.bin` changed), but sampled local runtime output parity failed:

- `safe_for_exact_eval_dispatch=false`
- `fail_closed_reasons=["render_output_parity_unsafe"]`
- aggregate mean absolute delta `0.07326045632362366` versus threshold `0.05`
- aggregate RMS delta `0.1109318540372162` versus threshold `0.08`
- aggregate max absolute delta `2.1078109741210938` versus threshold `1.5`
- sampled pairs: `0,19,39,58,77,97,116,135,155,174,193,213,232,251,271,290,309,328,348,367,386,406,425,444,464,483,502,522,541,560,580,599`

Dispatch verdict: this exact renderer-transplant candidate may not be
T4-dispatched under the AGENTS.md renderer-transplant rule, even after a lane
claim, unless a later candidate/report against the exact source/candidate SHA
pair reports `safe_for_exact_eval_dispatch=true`. No remote/GPU job was
submitted.
