# A1 / PR106 CPU-CUDA Axis Validation

Date: 2026-05-13
Author: codex
Scope: apples-to-apples axis validation for A1 and PR106/R2 PR101-grammar

## Rule

CPU and CUDA are separate evidence spaces. This ledger records measured pairs
only; it does not convert CPU scores to CUDA scores or CUDA scores to CPU
scores for promotion, retirement, or submission readiness.

## A1 Dual Axis

Source: `submissions/a1/contest_auth_eval.cpu.json`,
`submissions/a1/contest_auth_eval.cuda.json`,
`submissions/a1/dual_eval_adjudicated.json`.

- archive SHA-256:
  `87ec7ca5f2f328a8acdfc65f5cce0ab08a3a558eae88f36d4140870f141492b5`
- archive bytes: `178262`
- candidate label: `track1_phase_a1_score_gradient_latentalign_importpathfix_lr2e6`

| Axis | Score | SegNet | PoseNet | Evidence |
| --- | ---: | ---: | ---: | --- |
| [contest-CPU GHA Linux x86_64] | `0.19284757743677347` | `0.00056023` | `0.00003286` | `contest-CPU-1to1` |
| [contest-CUDA Tesla T4] | `0.2263520234784395` | `0.00066299` | `0.00017103` | `A++`, `score_axis=contest_cuda` |

Observed same-archive gap:

- CUDA minus CPU score: `+0.03350444604166603`
- CUDA/CPU pose distortion ratio: about `5.2048`
- CUDA/CPU SegNet distortion ratio: about `1.1834`

Classification: A1 is CPU-dominant on its measured paired axes. This is a
property of the A1 archive/runtime/scorer path, not a global rule for all
submissions.

## PR106/R2 PR101-Grammar Source Packet

Source:
`experiments/results/modal_auth_eval/pr106_latent_sidecar_r2_pr101_grammar_20260511T180000Z/contest_auth_eval.json`
and
`experiments/results/modal_auth_eval_cpu/pr106_latent_sidecar_r2_pr101_grammar_20260511T200000Z/contest_auth_eval.json`.

- archive SHA-256:
  `c48631e11a9bb18d051da9100ca4d5773558a8a81ac38dc8f6f4e8b6119d0383`
- archive bytes: `186780`

| Axis | Score | SegNet | PoseNet | Evidence |
| --- | ---: | ---: | ---: | --- |
| [contest-CUDA T4] | `0.2066181354574151` | `0.0006426` | `0.00003236` | `contest-CUDA` |
| [contest-CPU Linux x86_64] | `0.22806463271134514` | `0.00063196` | `0.00016402` | `contest-CPU` |

Observed same-archive gap:

- CPU minus CUDA score: `+0.02144649725393004`
- CPU/CUDA pose distortion ratio: about `5.0686`
- CPU/CUDA SegNet distortion ratio: about `0.9834`

Classification: this PR106/R2 PR101-grammar packet is CUDA-dominant on total
score because CPU PoseNet distortion is much worse even though CPU SegNet is
slightly lower. This directly falsifies any universal "CPU is always better"
or "CUDA is always better" rule.

## PR106/R2 PR101-Grammar Low-Level 151B Repack

Candidate:
`experiments/results/pr106_r2_pr101_grammar_lowlevel_repack_20260513_codex/pr106_r2_pr101_grammar_hnerv_brotli_repack_candidate.zip`

- archive SHA-256:
  `287e6edc612803a9a9d5de3ce50b421c039704f38bae442a6dcc97a3e8d6ed4d`
- archive bytes: `186629`
- source delta: `-151` bytes versus the source PR106/R2 PR101-grammar packet.

Measured:

| Axis | Score | SegNet | PoseNet | Evidence |
| --- | ---: | ---: | ---: | --- |
| [contest-CUDA T4] | `0.2065174760196528` | `0.0006426` | `0.00003236` | `contest-CUDA` |
| [contest-CPU Linux x86_64] | `0.22796397327358284` | `0.00063196` | `0.00016402` | `contest-CPU` |

CUDA apples-to-apples movement versus the matching source packet:

- source CUDA: `0.2066181354574151`
- candidate CUDA: `0.2065174760196528`
- measured CUDA delta: `-0.00010065943776230331`

CPU apples-to-apples movement versus the matching source packet:

- source CPU: `0.22806463271134514`
- candidate CPU: `0.22796397327358284`
- measured CPU delta: `-0.00010065943776230331`

Observed same-archive candidate gap:

- CPU minus CUDA score: `+0.02144649725393004`
- CPU/CUDA pose distortion ratio: about `5.0686`
- CPU/CUDA SegNet distortion ratio: about `0.9834`

Classification: measured CUDA and CPU score lowering is rate-only and
byte-closed. The PR106/R2 PR101-grammar packet remains CUDA-dominant after the
151-byte repack.

## Implications

1. Device-axis behavior is submission-specific. Treat the axis gap as a
   measured property of the archive, runtime tree, inflate device, scorer
   device, and scorer components.
2. A1 and PR106/R2 differ qualitatively: A1 is CPU-dominant; PR106/R2
   PR101-grammar is CUDA-dominant.
3. Score lowering work must preserve paired-axis tables near every claim that
   says "medal-band", "submission-ready", "frontier", or "rounds to".
4. The next valid PR106/R2 action is a candidate-specific release surface and
   strict contest-final compliance only if this 151-byte rate win is worth
   submission packaging. The score-lowering frontier should otherwise move to
   transforms that change representation/scorer components, not more generic
   brotli polish.
