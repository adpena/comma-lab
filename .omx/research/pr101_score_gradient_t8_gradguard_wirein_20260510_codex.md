# PR101 score-gradient T8 + gradguard wire-in (2026-05-10)

## Summary

`experiments/train_score_gradient_pr101_finetune.py` now uses the tensor-valued
score-domain oracle `tac.losses.scorer_loss_terms_btchw` instead of the older
`scorer_loss` wrapper that detached `pose_dist` / `seg_dist` into logging floats.

This makes the PR101/HNeRV fine-tune path consistent with the Phase 1 T8 verdict:
the primary SegNet surrogate defaults to `sinkhorn`, and the first training step
must prove that the primary scorer-domain loss itself reaches decoder
parameters before auxiliary KL/pixel losses are allowed to contribute.

When `--pr101-source-dir` is supplied in a non-smoke run, the trainer now
attempts archive closure in the same run by building byte-closed candidates
from `checkpoint_best_proxy.pt` and `checkpoint_ema.pt` through
`tools/build_pr101_finetuned_archive.py`, then writes
`archive_builds_manifest.json`. The remote A1 script selects
`checkpoint_best_proxy.pt` when present and falls back to `checkpoint_ema.pt`
otherwise, so the archive builder consumes the scorer-domain-selected
checkpoint rather than silently defaulting to the last EMA file.

Codex integration follow-up: the remote A1 script now also passes
`--pr101-source-dir "$WORKSPACE/$PR101_SOURCE_DIR"` into the trainer itself.
That keeps the export-first guard and the remote train path aligned; otherwise
the next non-smoke CUDA dispatch would fail before training starts.

## Evidence

- Code path: `experiments/train_score_gradient_pr101_finetune.py`.
- Primary scorer oracle: `tac.losses.scorer_loss_terms_btchw`.
- Default primary SegNet surrogate: `sinkhorn`.
- First-step proof field: `train_log.jsonl[*].step_metrics[*].decoder_grad_l2`.
- Primary scorer-only proof field:
  `train_log.jsonl[*].step_metrics[*].primary_scorer_decoder_grad_l2`.
- Archive closure manifest when `--pr101-source-dir` is provided:
  `archive_builds_manifest.json`.
- Remote builder checkpoint selection:
  `scripts/remote_track1_phase_a1_score_gradient_pr101.sh`.
- Remote trainer source-dir closure:
  `scripts/remote_track1_phase_a1_score_gradient_pr101.sh`.
- Remote CUDA manifest parser now uses `tac.auth_eval_schema` so canonical
  `contest_auth_eval.py` fields (`canonical_score`, `avg_posenet_dist`,
  `avg_segnet_dist`, `rate_unscaled`, `archive_size_bytes`, `n_samples`) are
  required before `score_claim=true`. A successful eval with missing canonical
  metrics now exits fail-closed instead of writing null score fields.
- Remote A1 script now requires a real active dispatch claim ledger before any
  CUDA train/eval work and captures train/build/eval return codes without
  losing logs under `set -euo pipefail`.
- Focused test:

```bash
.venv/bin/python -m pytest tests/test_train_score_gradient_pr101_real_data.py -q
```

Result after hardening: covered by the focused suite in this turn.

Additional focused verification after remote integration:

```bash
bash -n scripts/remote_track1_phase_a1_score_gradient_pr101.sh
.venv/bin/python -m pytest \
  src/tac/tests/test_dispatch_phase_a1_score_gradient_pr101.py \
  tests/test_train_score_gradient_pr101_real_data.py \
  src/tac/tests/test_build_pr101_finetuned_archive_codec_dir.py -q
# 26 passed
```

## Claim discipline

No score claim is made here. This is training-path correctness and score-lowering
readiness only. A promoted result still requires:

- PR101 archive custody and loaded latent/source metadata;
- real contest video source;
- CUDA training run with eval-roundtrip and differentiable YUV6 enabled;
- byte-closed archive build from the selected checkpoint;
- exact CUDA auth eval on the emitted archive;
- formula recomputation from component fields.
- non-null normalized manifest fields and `n_samples=600`; otherwise the result
  stays non-claimable even when the eval process exits 0.

MPS remains sweep/proxy-only and must not be used for auth eval.
