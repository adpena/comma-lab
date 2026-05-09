# Phase A1 SegNet-Guarded Modal Dispatch — 2026-05-09

## Status

Dispatched, active. This is not a score claim.

- Lane: `track1_phase_a1_score_gradient`
- Instance/job: `track1_phase_a1_score_gradient_segguard_kl0p5_l1p02_40e_20260509T052414Z_modal`
- Modal call id: `fc-01KR5K3V8VT735P73ZV881ZQA2`
- Dispatch URL: local Modal app run `ap-mxMMZkm7iYAqQX58jvQK0T`
- Claim status: `active_dispatching`
- Predicted ETA: `2026-05-09T07:54:39Z`
- Estimated cost: `$1.47`
- Recover command:
  `.venv/bin/python experiments/modal_phase_a1_score_gradient_pr101.py recover --label track1_phase_a1_score_gradient_segguard_kl0p5_l1p02_40e_20260509T052414Z_modal`

## Inputs

- PR101 archive:
  `experiments/results/public_pr101_hnerv_ft_microcodec_intake_20260504_codex/archive.zip`
  - bytes: `178258`
  - SHA-256: `b83bf3488625dbd73adeddff91712994197ab53098e578e91327a0c6e49efb3e`
- PR101 runtime source snapshot:
  `experiments/results/public_pr101_hnerv_ft_microcodec_intake_20260504_codex/source/submissions/hnerv_ft_microcodec/src`
  - zip-stored bytes: `19137`
  - SHA-256: `cf7853a09a08654daa5a6363eba0e36f2b5d2ac9060999f7b799d3d99f8a6a17`
- Video:
  `upstream/videos/0.mkv`
  - bytes: `37545489`
  - SHA-256: `2611f5f3e186f3529777749f97bd4cce3a208d6b3559e137bd45d256980d2fa9`

## Configuration

This is a guarded refire after the long `lr=1e-6` run regressed on macOS CPU
advisory. It keeps the winning short schedule shape from the current A1
anchor, but strengthens reconstruction/latent regularization to protect
SegNet:

- epochs: `40`
- steps per epoch: `8`
- batch size: `4`
- learning rate: `2e-6`
- max frames: `1200`
- aux KL weight: `0.5`
- aux pixel L1 weight: `0.02`
- `continue_after_nvdec_failure=true`

The dispatcher now writes remote eval work under
`/workspace/pact/experiments/results/modal_phase_a1_remote/...`, not `/tmp`,
so `contest_auth_eval.py` should no longer reject the evidence path as temp
scratch.

## Classification Rules

No score promotion is allowed until the run is harvested and reviewed.

- If Modal T4 DALI/NVDEC exact eval succeeds, classify with `[contest-CUDA]`
  custody and then require paired `[contest-CPU]`.
- If DALI/NVDEC fails but training/build completes, classify as
  `cuda-training-build-only`, screen locally on macOS CPU advisory, and do not
  promote.
- If the candidate is worse than the current A1 anchor, retire only this
  measured configuration.

## Rationale

The current A1 anchor is CPU-positive but CUDA-not-frontier:
`178262 B`, `[contest-CPU] 0.19284757743677347`, `[contest-CUDA] 0.2263520234784395`.
The blind long `lr=1e-6` follow-up built cleanly but regressed to
`0.19359165212458496` on macOS CPU advisory. The next score-lowering move is
therefore not longer training; it is a short guarded schedule that tries to
retain the A1 CPU gain while reducing the SegNet drift seen in longer runs.

## Reactivation / Follow-up

- Harvest within Modal result-cache TTL.
- Record archive bytes, archive SHA-256, runtime-tree SHA, components, logs,
  and claim terminal row.
- If build-only, run macOS CPU advisory as a screen, then exact CUDA/CPU only
  if competitive with the current A1 anchor.
- If positive, freeze this schedule as the new A1 branch and launch a paired
  schedule-neighborhood sweep around KL/L1 weights.
