# Public HNeRV Adapter Replays - 2026-05-04

## Context

The public PR98 and PR99 HNeRV submissions were staged for exact CUDA replay
after PR95 stemperm established the then-current local A++ anchor:

- PR95 stemperm A++ score: `0.23089404465634825`
- PR95 stemperm archive bytes: `178277`
- PR95 stemperm archive SHA-256:
  `e40c3f2fb3587b12eccb8707e0a1b7831fde149318f3eb212500c674ccbfbf28`

Public claims to verify:

- PR98 `hnerv_muon_finetuned_from_pr95`: body recompute
  `0.19625777542725248`, archive SHA-256
  `7ecb0df1c4627d55d88e03eff3d890b7a7a5b047c62515acff20232cf29310eb`,
  bytes `178392`.
- PR99 `hnerv_muon_lc`: body recompute `0.19668072586615531`,
  archive SHA-256
  `278b1c7a1bd6b03a5bceddafcb3489b2624c558ad22825d9211b701333b6eefb`,
  bytes `178546`.

## Pre-Score Runtime Contract Failure

The first PR98 exact replay failed before score. Archive identity was correct,
but `contest_auth_eval.py` called the public `inflate.sh` with the canonical
contest signature:

```text
inflate.sh <archive_dir> <output_dir> <video_names_file>
```

The public PR98 wrapper forwarded those three arguments to `inflate.py`, which
expects:

```text
inflate.py <src.bin> <dst.raw>
```

The exact log printed the usage string and exited before
`contest_auth_eval.json` could be written. This is invalid score evidence and
does not rank or retire the method.

PR99 used the same public wrapper shape, so the original PR99 replay and both
duplicate public-wrapper jobs were stopped or allowed to fail, then closed in
the dispatch ledger as superseded by adapter replays.

PR95 residual-atom screen also failed before score because the public PR95
runtime path did not have `brotli` available inside inflate. That run is
invalid/no-score; it is a dependency-harness failure, not residual-atom method
evidence.

## Contest-Signature Adapters

Strict adapters were built under:

```text
experiments/results/public_runtime_adapters_20260504_codex/
```

They keep the exact archive bytes unchanged and only adapt the runtime call
contract:

```text
archive_dir/0.bin -> output_dir/0.raw
```

Both adapters fail closed unless `public_test_video_names.txt` contains exactly
one nonempty entry: `0.mkv`.

Adapter local preflight:

- PR98 runtime tree SHA-256:
  `0232154c17410621325ec1647e0f0723b3310d63b0d4bc4bf7bbb5e9aa2fccd0`
- PR99 runtime tree SHA-256:
  `0f201480ddc815c2ea761f5c52a02fcfd3fcd0e60542d3129dfbc0491769b697`
- Source manifest:
  `.omx/state/public_hnerv_adapter_replays_20260504T0956Z_manifest.json`
- Remote manifest byte verification passed with `1664` files and
  `29515753` bytes.
- Studio CPU preflight reported no CUDA, so exact-eval submit used an
  auditable remote-preflight skip; each Batch job still performs runner-side
  CUDA/DALI preflight on the T4 worker before scoring.

## Exact Eval Results

Both corrected adapter replays completed through the canonical exact CUDA path:

- `exact_eval_public_pr98_hnerv_adapter_t4_20260504T0958Z`
  - lane: `public_pr98_hnerv_muon_finetuned_t4_adapter_replay`
  - evidence grade: `A++`
  - score recomputed from components: `0.22933111465960354`
  - component contributions:
    - SegNet: `0.068841`
    - PoseNet: `0.041706114659603576`
    - rate: `0.11878399999999999`
  - component distances:
    - `avg_segnet_dist=0.00068841`
    - `avg_posenet_dist=0.00017394`
  - hardware: `Tesla T4`, CUDA, `600` samples
  - archive SHA-256:
    `7ecb0df1c4627d55d88e03eff3d890b7a7a5b047c62515acff20232cf29310eb`
  - bytes: `178392`
  - runtime tree SHA-256:
    `0232154c17410621325ec1647e0f0723b3310d63b0d4bc4bf7bbb5e9aa2fccd0`
  - canonical artifact:
    `experiments/results/lightning_batch/exact_eval_public_pr98_hnerv_adapter_t4_20260504T0958Z/contest_auth_eval.adjudicated.json`
- `exact_eval_public_pr99_hnerv_adapter_t4_20260504T0958Z`
  - lane: `public_pr99_hnerv_muon_lc_t4_adapter_replay`
  - evidence grade: `A++`
  - score recomputed from components: `0.2297226895103603`
  - component distances:
    - `avg_segnet_dist=0.00069279`
    - `avg_posenet_dist=0.0001727`
  - hardware: `Tesla T4`, CUDA, `600` samples
  - archive SHA-256:
    `278b1c7a1bd6b03a5bceddafcb3489b2624c558ad22825d9211b701333b6eefb`
  - bytes: `178546`
  - runtime tree SHA-256:
    `0f201480ddc815c2ea761f5c52a02fcfd3fcd0e60542d3129dfbc0491769b697`
  - canonical artifact:
    `experiments/results/lightning_batch/exact_eval_public_pr99_hnerv_adapter_t4_20260504T0958Z/contest_auth_eval.adjudicated.json`

PR98 is the current local exact-T4 champion. PR99 is independently valid but
slightly worse. Both public-wrapper failures are now classified as pre-score
adapter contract failures, not method failures.

## Submission Packet Gate

The PR98 champion packet was built under:

```text
experiments/results/submission_packet_pr98_adapter_20260504/
```

Primary packet files:

- submission directory:
  `experiments/results/submission_packet_pr98_adapter_20260504/apogee_pr98_hnerv_adapter`
- packet manifest:
  `experiments/results/submission_packet_pr98_adapter_20260504/submission_packet_manifest.json`
- packet checklist:
  `experiments/results/submission_packet_pr98_adapter_20260504/submission_packet_checklist.md`
- strict pre-submission compliance JSON:
  `experiments/results/submission_packet_pr98_adapter_20260504/pre_submission_compliance.json`

The strict pre-submission gate passed with exact archive SHA/bytes, T4 auth
eval, runtime-tree match against the submission directory, report linkage,
archive manifest consistency, public hygiene, dispatch-claim closure, and a
single `0.bin` archive member.

## Ready Follow-Ups

PR98 validated as the exact T4 champion. The PR98 fixed-channel postprocess
ablation set is already prepared under:

```text
experiments/results/pr98_channel_ablation_candidates_20260504_codex/
```

Those candidates change runtime tree only, not archive bytes, so runtime-tree
SHA is score-affecting custody.

PR99 also validated as A++ but is slightly worse than PR98. The sidecar
deconstruction worker is examining the one-dimensional per-pair latent
correction stream for lossless repack and ablation opportunities.
