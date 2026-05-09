# Phase A1 Latent-Aligned Import-Path-Fixed Modal Refire - 2026-05-09

## Status

Dispatch is active. Immediate recovery probe returned `NOT READY`, which means
the call is still queued or running rather than failing immediately with the
previous `ModuleNotFoundError: No module named 'tac'` worker import-path bug.

This is not a score claim and is not promotion-eligible.

## Dispatch

- Lane: `track1_phase_a1_score_gradient`
- Platform: Modal T4
- Instance/job id: `track1_phase_a1_score_gradient_latentalign_importpathfix_lr2e6_20260509T012628Z_modal`
- Modal call id: `fc-01KR55GH98QW3J3QDGQB1EG4CR`
- Modal app run: `https://modal.com/apps/adpena/main/ap-niKPFHkxiRjp2EDOe27dx6`
- Dispatched at: `2026-05-09T01:26:57Z`
- Predicted ETA: `2026-05-09T03:56:54Z`
- Estimated cost: `$1.47`
- Evidence grade: `[advisory only - dispatch in flight]`
- Score claim: `false`
- Promotion eligible: `false`

## Inputs

- PR101 archive: `178258` bytes,
  SHA-256 `b83bf3488625dbd73adeddff91712994197ab53098e578e91327a0c6e49efb3e`
- PR101 source zip snapshot: `19137` bytes,
  SHA-256 `cf7853a09a08654daa5a6363eba0e36f2b5d2ac9060999f7b799d3d99f8a6a17`
- Video: `37545489` bytes,
  SHA-256 `2611f5f3e186f3529777749f97bd4cce3a208d6b3559e137bd45d256980d2fa9`

## Configuration

- Epochs: `40`
- Steps per epoch: `8`
- Batch size: `4`
- Learning rate: `2e-6`
- Max frames: `1200`
- Auxiliary KL weight: `0.2`
- Auxiliary pixel L1 weight: `0.01`
- `continue_after_nvdec_failure`: `true`

This keeps the conservative latent-aligned A1 configuration from the failed
`20260509T011929Z` refire, but uses the committed Modal worker import-path fix.

## Recovery

```bash
.venv/bin/python experiments/modal_phase_a1_score_gradient_pr101.py recover \
  --label track1_phase_a1_score_gradient_latentalign_importpathfix_lr2e6_20260509T012628Z_modal
```

Immediate probe:

```text
NOT READY: call_id=fc-01KR55GH98QW3J3QDGQB1EG4CR still queued or running.
```

## Next Review

On recovery, classify by exact terminal evidence:

- If exact CUDA succeeds, record archive bytes, archive SHA-256, runtime-tree
  SHA, components, recomputed score, logs, sample count, and terminal claim row.
- If DALI/NVDEC fails but training/build succeeds, preserve the archive and run
  macOS CPU advisory only as collapse screening.
- If training/build fails, close the claim with a terminal failure row and keep
  reactivation criteria scoped to that failure class.
