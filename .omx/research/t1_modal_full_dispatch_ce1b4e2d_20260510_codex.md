# T1 Modal full dispatch from ce1b4e2d — Codex ledger

Date: 2026-05-10

## Dispatch

Launched the first full 600-pair T1 Ballé end-to-end Modal T4 run after the
subset guard was hardened to avoid false auth-eval attempts.

- Lane: `t1_balle_128k_endtoend`
- Instance/job id: `t1_balle_modal_full_ce1b4e2d_20260510T1333Z`
- Modal call id: `fc-01KR91CR2V5AM6Y2DVBFTXCDVM`
- Modal app: `comma-t1-balle-endtoend`
- Local mounted commit: `ce1b4e2d8804c07abd2ee8d7beae9c95fc98e754`
- Mounted-code dirty state: `false`
- GPU: Modal T4
- Estimated cost: `$14.16`
- Predicted ETA: `2026-05-11T13:31:56Z`

Command:

```bash
PYTHONPATH=src:upstream:$PWD .venv/bin/modal run --detach \
  experiments/modal_t1_balle_endtoend.py --execute \
  --label t1_balle_modal_full_ce1b4e2d_20260510T1333Z \
  --epochs 3000 \
  --batch-size 16 \
  --timeout-hours 24.0 \
  --cost-cap-usd 80.0 \
  --train-timeout-hours 22.5 \
  --sinkhorn-max-positions-per-chunk 2048
```

Recover:

```bash
.venv/bin/python experiments/modal_t1_balle_endtoend.py recover \
  --label t1_balle_modal_full_ce1b4e2d_20260510T1333Z
```

## Custody and score-claim status

The run is **in flight**. There is no score claim, no promotion claim, and no
rank/kill decision until recovery verifies exact contest-CUDA evidence:

- `auth_eval_schema` blockers equal zero;
- `n_samples == 600`;
- device is CUDA on contest-equivalent hardware;
- archive SHA and size match the scored packet;
- score is recomputed from components;
- dispatch claim is closed terminally.

Plan metadata recorded `contest_cuda_auth_eval_requested=true` because
`max_target_pairs=None` means full 600-pair export. This is intentionally
different from bounded guard runs, which now remain training/export smoke only.

## Next action

Do not launch a duplicate T1 run while the active claim exists. Harvest with the
recover command above, then classify the result as one of:

- `completed_t1_contest_cuda_recovered` with a valid exact score packet;
- `completed_t1_training_only_recovered_no_score_claim` if training succeeds
  without valid score evidence;
- `failed_t1_modal_recovered_no_score_claim` with the exact blocker if the
  remote script fails.

## 2026-05-10T13:38Z harvest result: full-run batch-16 T4 OOM

Recover closed the dispatch terminally as
`failed_t1_modal_recovered_no_score_claim`.

What worked:

- Modal scorer import probe passed;
- NVDEC probe passed on `Tesla T4`;
- mounted-code custody was clean at commit `ce1b4e2d`;
- full-run auth eval was correctly requested (`max_target_pairs=None`).

What failed:

```text
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 58.00 MiB.
GPU 0 has a total capacity of 14.56 GiB of which 57.81 MiB is free.
Process 1 has 14.50 GiB memory in use.
```

The OOM happened inside SegNet during score-domain training before checkpoint,
packet compile, or auth eval. Classification: `t1_full_batch16_t4_scorer_oom`.
This is not a model-family negative and not score evidence.

Hardening follow-up:

- `experiments/modal_t1_balle_endtoend.py` now rejects full 600-pair Modal T4
  score-domain plans with `batch_size > 1` until gradient accumulation or
  activation checkpointing exists;
- the default Modal T1 batch size is now `1`;
- duplicate same-lane active dispatches still fail closed via the claims
  summary check.

Next valid T1 score-path probe is a full 600-pair, batch-1 full-path smoke
(`epochs=1`, no `max_target_pairs`) to validate memory, export, packet compile,
and exact auth-eval plumbing before any long training run.

## 2026-05-10T13:47Z batch-1 full-path smoke launched

Launched the follow-up full 600-pair, batch-1 Modal T4 smoke from clean commit
`b82164a7` after the OOM guard, tracked PR95 parity profile, and Modal mount
fix landed.

- Lane: `t1_balle_128k_endtoend`
- Instance/job id: `t1_balle_modal_fullpath_smoke_b82164a7_20260510T1348Z`
- Modal call id: `fc-01KR928YET5BZGFV803ANT8CTS`
- Modal app: `comma-t1-balle-endtoend`
- Mounted PR95 parity profile:
  `.omx/research/pr95_hnerv_muon_trainer_parity_profile_20260510.json`
- Epochs: `1`
- Batch size: `1`
- `max_target_pairs`: unset, so full 600-pair export/auth eval is requested
- Train timeout: `3h`
- Estimated cost cap: `$14.16` planned Modal T4 upper bound

Recover:

```bash
.venv/bin/python experiments/modal_t1_balle_endtoend.py recover \
  --label t1_balle_modal_fullpath_smoke_b82164a7_20260510T1348Z
```

This is a full-path plumbing and memory probe. It may produce an exact score,
but no score claim is valid until recover closes the active dispatch claim and
auth-eval adjudication reports zero blockers.

## 2026-05-10T14:03Z harvest result: batch-1 EMA proxy eval OOM

Recover closed the batch-1 smoke terminally as
`failed_t1_modal_recovered_no_score_claim`.

What worked:

- Modal wrapper returned a ready result and recovery harvested artifacts;
- scorer import probe passed;
- NVDEC probe passed on `Tesla T4`;
- mounted-code custody was clean at commit `b82164a7`;
- training entered the full 600-pair score-domain path with PR95 parity flags:
  `eval_roundtrip=True`, differentiable YUV6, `yuv6_mode=monkey_patch_global`,
  T8 Sinkhorn, T13, and T19.

What failed:

```text
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 12.74 GiB
```

The failure occurred in `_eval_ema_proxy()` after epoch 1. The trainer had
correctly trained with `--batch-size 1`, but the EMA proxy evaluation decoded
the entire 600-pair latent table at once:

```text
balle_out = balle(latents)
decoded = decoder(balle_out["y_hat"])
```

Classification: `t1_batch1_fullpath_ema_proxy_eval_oom`. This is a trainer
memory bug and not evidence against T1/Ballé/HNeRV parity.

Hardening follow-up:

- `_eval_ema_proxy()` now evaluates EMA in bounded chunks and accumulates
  pixel-L1 and rate over chunks;
- `--eval-batch-size` was added, defaulting to `--batch-size`;
- `scripts/remote_lane_t1_balle_endtoend.sh` passes `--eval-batch-size` through
  explicitly so Modal, Vast, Kaggle, AWS, Azure, and GCP provider wrappers use
  the same bounded trainer contract;
- tests cover the exact bug class: proxy eval must not call the decoder with
  the full latent table.

Next valid T1 score-path probe is the same full 600-pair, batch-1,
eval-batch-1 Modal T4 smoke from the patched commit. It should progress past
epoch-1 EMA proxy eval to archive export, packet compile, and exact auth-eval
or expose the next concrete blocker.
