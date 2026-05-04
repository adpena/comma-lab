# Public PR100 Intake And Replay - 2026-05-04

## Source

- PR: https://github.com/commaai/comma_video_compression_challenge/pull/100
- title: `hnerv_lc_v2 submission (0.1954)`
- author: `BradyMeighan`
- PR state at intake: open
- source branch: `BradyMeighan:submission/hnerv_lc_v2`
- head SHA at intake: `0a8d343a5dc7f9e93d9d0e6cb8bc15f6c626c050`
- release archive URL:
  `https://github.com/BradyMeighan/comma_video_compression_challenge/releases/download/hnerv-lc-v2-archive/archive.zip`

## Public Claim

Reported rounded components in the PR body:

- PoseNet: `0.00003443`
- SegNet: `0.00057654`
- archive bytes: `178981`
- public exact CI score: `0.1954`
- recomputed from rounded components: `0.19538542397525555`

This public body claim remains external/CPU or CI evidence. The score-bearing
row for our work is the exact local CUDA/T4 replay below.

## Local Custody

- local archive: `experiments/results/public_pr100_intake_20260504_codex/archive.zip`
- archive SHA-256:
  `afd53348f50303bf0ec6a7ffecc1ac037df2f1c70745244b9c45c72e8eb80641`
- archive bytes: `178981`
- archive member: `0.bin`
- member bytes: `178873`
- source clone:
  `experiments/results/public_pr100_intake_20260504_codex/source`
- runtime adapter:
  `experiments/results/public_runtime_adapters_20260504_codex/pr100_runtime_adapter`

The adapter preserves the public decoder logic and only changes the wrapper to
fail closed on the canonical contest signature:

```text
inflate.sh <archive_dir> <output_dir> <video_names_file>
```

## Exact Replay Dispatch And Result

- lane: `public_pr100_hnerv_lc_v2_t4_adapter_replay`
- job: `exact_eval_public_pr100_hnerv_lc_v2_adapter_t4_20260504T1213Z`
- machine request: `g4dn.2xlarge` / T4
- source manifest:
  `.omx/state/public_pr100_adapter_replay_20260504T1211Z_manifest.json`
- manifest remote byte verification: `OK`, `1643` files, `29273195` bytes
- result artifact:
  `experiments/results/lightning_batch/exact_eval_public_pr100_hnerv_lc_v2_adapter_t4_20260504T1213Z/contest_auth_eval.adjudicated.json`
- evidence grade: `A++`
- hardware: Tesla T4, CUDA, full `600` samples
- exact score: `0.22826947142244708`
- archive bytes/SHA:
  `178981`,
  `afd53348f50303bf0ec6a7ffecc1ac037df2f1c70745244b9c45c72e8eb80641`
- components: SegNet `0.00067623`, PoseNet `0.00017198`
- runtime tree SHA-256:
  `ef6323533666c9cac1c204a9d3f7054157d44a185b16fc859fb3f0438ccd1832`
- strict packet:
  `experiments/results/submission_packet_pr100_adapter_20260504/apogee_pr100_hnerv_lc_v2_adapter`
- strict compliance gate:
  `experiments/results/submission_packet_pr100_adapter_20260504/pre_submission_compliance.pr100.json`

PR100 supersedes Apogee PR #107 in the local exact claim matrix by
`0.0010616432371564621` score points. The public PR100 body score remains
external; the ranked score is the adjudicated T4 replay above.

## Late Public Frontier Refresh

After PR100, public PRs #101-#106 appeared with lower self-reported/title
claims, including `kitchen_sink (0.19797)`, `hnerv_lc_ac submission (0.19)`,
and `belt_and_suspenders (0.20946)`. They are urgent deconstruction targets,
but they are not local exact evidence until each archive/runtime pair passes
the same intake, wrapper, exact CUDA replay, and strict packet gate.
