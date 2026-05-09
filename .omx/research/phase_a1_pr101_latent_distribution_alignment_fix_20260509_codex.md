# Phase A1 PR101 Latent Distribution Alignment Fix - 2026-05-09

Evidence grade: `[engineering bugfix; no score claim]`

## Finding

The first A1 score-gradient fine-tune was not merely too aggressive. The
non-smoke training loop sampled random latent vectors while the downstream
packet builder preserved PR101's original `latent_blob + sidecar_blob`
bit-for-bit. That trains a decoder on a latent distribution the scored archive
will never feed it.

This explains the measured-config collapse:

- archive: `205,879 B`
- archive SHA-256:
  `cb9de2b71133929b0c2df00b0e511b9c306939d62438ffb348e947aef719e185`
- macOS CPU advisory score: `3.7216542390470915`
- pose: `0.17846388`
- seg: `0.02248664`
- exact CUDA: not run, Modal DALI/NVDEC preflight failed

The result remains a retired measured configuration only. It is not a family
kill and should not be used as evidence against A1 after this fix.

## Fix

The archive-derived latent contract is now reusable and scorer-free:

- `src/tac/pr101_split_brotli_codec.py` ports PR101's compact latent decoder
  and latent sidecar grammar.
- `src/tac/pr101_archive_state_loader.py` exposes
  `load_pr101_archive_latents()`, returning the same `(600, 28)` float32 rows
  that PR101 `inflate.py` feeds to `HNeRVDecoder`.
- `experiments/train_score_gradient_pr101_finetune.py` non-smoke
  `RealPairBatchSource` now samples those archive latent rows, aligned with the
  non-overlapping frame-pair stream `(0,1), (2,3), ...`.

## Validation

Focused checks run before promotion:

```bash
.venv/bin/python -m py_compile \
  src/tac/pr101_split_brotli_codec.py \
  src/tac/pr101_archive_state_loader.py \
  experiments/train_score_gradient_pr101_finetune.py

.venv/bin/python -m pytest \
  tests/test_pr101_archive_state_loader.py \
  tests/test_train_score_gradient_pr101_real_data.py \
  -q

.venv/bin/python experiments/train_score_gradient_pr101_finetune.py \
  --smoke --device cpu \
  --output experiments/results/track1_a1_archive_latent_contract_smoke_20260509_codex
```

Results: focused loader/training tests `14 passed`; A1 smoke run completed with
`SMOKE PASS`.

The PR101 fixture test compares `load_pr101_archive_latents()` against the
public PR101 runtime `codec.py::parse_archive()` on the same archive member
`x`; tensors are `torch.equal`.

## Dispatch consequence

A1 may be refired only with a constrained fine-tune that uses archive-derived
latents and preserves the reconstruction basin before exact CUDA or
contest-CPU spend. Suggested reactivation profile:

- small learning rate (`1e-6` to `2e-6` range)
- short run first (`40-60` epochs, low steps per epoch)
- byte-closed archive rebuild
- local macOS CPU advisory as a collapse screen only
- exact `[contest-CUDA]` and `[contest-CPU]` only after packet custody passes

No score is promoted by this fix.
