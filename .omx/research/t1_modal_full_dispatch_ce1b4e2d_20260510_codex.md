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

## 2026-05-10T14:23Z active e7845e4c smoke classification after runtime-custody hardening

The follow-up Modal call remains pending:

- Label: `t1_balle_modal_fullpath_smoke_e7845e4c_20260510T1410Z`
- Modal call id: `fc-01KR93KESQB87VSD2WKE4GYZEC`
- Lane: `t1_balle_128k_endtoend`
- Status at this addendum: `pending`

Important promotion boundary: this run launched from clean git head `e7845e4c`
before the extracted-archive runtime hardening landed. It is still useful and
must be harvested for logs, training behavior, memory behavior, and packet
compiler/auth-eval failure classification. It must not be promoted as
contest-compliant score evidence unless the harvested artifacts independently
prove the exact runtime consumed evaluator-provided extracted archive members
and auth-eval adjudication has zero blockers.

Current working-tree hardening closes that bug class for future T1 dispatches:

- emitted Phase 1 `inflate.py` reads `x`, `decoder.bin`, and `balle.bin` from
  the passed `archive_dir`;
- `tac.phase1_packet_compiler` no-op smoke materializes extracted archive
  members only, matching `contest_auth_eval.py`;
- Catalog #146 preflight rejects runtime-local `HERE/archive.zip` fallback.
- Modal T1 recovery now adds
  `t1_mounted_code_missing_extracted_archive_runtime_hardening` for any
  harvested result whose mounted code does not contain commit `0be54cbf`.

Next valid score-lowering dispatch after this active claim closes is a patched
T1 Modal smoke from the post-hardening commit. Do not duplicate the active
claim while it is pending.

## 2026-05-10T14:33Z e7845e4c smoke harvest classified

Recovery harvested `t1_balle_modal_fullpath_smoke_e7845e4c_20260510T1410Z`
terminally and closed the active dispatch claim as
`completed_t1_training_only_recovered_no_score_claim`.

Custody:

- Modal call id: `fc-01KR93KESQB87VSD2WKE4GYZEC`
- Harvest summary:
  `experiments/results/t1_balle_modal_fullpath_smoke_e7845e4c_20260510T1410Z/harvest_summary.json`
- Harvest summary SHA-256:
  `adc672d0b13a565882c0ea23c7a0cd76ba28385c90214da3855eec9a954c1bab`
- Contest CUDA auth-eval JSON SHA-256:
  `651e0bdf9e792c6322f03e8cac0421f46e964c66686a2d6e92ca9c2582bd2124`
- Packet compiler manifest SHA-256:
  `3f0fee9175fbcd9f542de7453c0cb965a6c34db8b5a0284aed5faccf1d6f2290`

Auth-eval did run on `Tesla T4`, `n_samples=600`, and recomputed the formula
from components, but this is **not score evidence for promotion**:

- `score_claim=false`
- blocker:
  `t1_mounted_code_missing_extracted_archive_runtime_hardening`
- mounted code was pre-`0be54cbf` runtime-custody hardening
- `promotion_eligible=false`
- `rank_or_kill_eligible=false`

Measured diagnostic result:

```text
score_recomputed_from_components = 56.06364706567909
seg_avg = 0.50482631
pose_avg = 2.75759292
archive_size_bytes = 495206
archive_sha256 = 8cd20cc0caaa2327e4ca5a1b642d9e666bdea926e75fe53ae72e38cca65435e2
runtime_tree_sha256 = bf459a0f84619c01bf12ff45e2797ecb1850da4bf2be87265e9914e420e8708e
```

Classification: this is a T1 full-path training/runtime diagnostic negative,
not a T1/Ballé family kill and not a score-lowering candidate. It proves the
post-EMA-fix remote path can reach archive export, packet compile, and exact
CUDA auth eval, but the one-epoch output is catastrophically untrained and the
run cannot be promoted because the runtime-custody hardening was absent at
dispatch time.

Next valid T1 score-lowering work is a bounded guard or full dispatch from a
post-`0be54cbf` commit that keeps the extracted-archive runtime contract,
small Sinkhorn memory settings, mounted-code custody, and schema blockers at
zero before any score claim.

## 2026-05-10T14:38Z current-head full Phase 1 dispatch launched

Commit `ab2d0f6e` launched the next T1 run after the e7845 pre-hardening smoke
closed. This is no longer a one-epoch smoke: it is a full 3000-epoch,
600-pair T1 Phase 1 run, still bounded by Modal function timeout and the
operator cost cap.

- Instance job id: `t1_balle_modal_phase1_ab2d0f6_20260510T1437Z`
- Modal call id: `fc-01KR955JSYQAVTTYZA48VAV7WJ`
- Modal URL:
  `https://modal.com/apps/adpena/main/ap-1fCuVHqShCT1puDuPs7SHY`
- Mounted code git head:
  `ab2d0f6ec1cf7aed05b8424a0b5f5d79b42698bf`
- Mounted code dirty: `false`
- Worktree/index patch SHA-256: empty patch
  `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`
- Metadata SHA-256:
  `e3cfc8dc088c42822edb3cf1b035612057b4d16da17ffbc6b4e1bc28104cce09`
- Plan SHA-256:
  `7003baabb61c0545ff9177a5c3759d050f5a9e5d502004d4994b5d3eafac1d35`

Parameters:

```text
epochs=3000
batch_size=1
max_target_pairs=null
sinkhorn_max_positions_per_chunk=2048
train_timeout_hours=22.5
timeout_hours=24
cost_cap_usd=80
contest_cuda_auth_eval_requested=true
```

Immediate Modal status:

```text
result_state=pending
function_call_id=fc-01KR955JSYQAVTTYZA48VAV7WJ
```

The Level-2 claim is active. Do not duplicate this lane. Recovery command:

```bash
.venv/bin/python experiments/modal_t1_balle_endtoend.py recover \
  --label t1_balle_modal_phase1_ab2d0f6_20260510T1437Z
```

Authority boundary: `score_claim=false` until recovery verifies exact
contest-CUDA auth-eval schema blockers at zero. If this run fails, classify the
stage precisely and preserve it as implementation/runtime/training evidence,
not a family kill.
