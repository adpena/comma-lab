# T1 Modal guard relaunch and Kaggle proxy materialization — Codex ledger

Date: 2026-05-10

## T1 bounded Modal guard

- Lane: `t1_balle_128k_endtoend`
- Label / instance job id: `t1_balle_modal_guard_c4100de4_20260510T0915Z`
- Modal app: `comma-t1-balle-endtoend`
- Modal run URL printed by local launcher:
  `https://modal.com/apps/adpena/main/ap-4EJPBdeQqxzaG7rE2KEriI`
- Function call id: `fc-01KR8JMK531A9PECP0CV513KQM`
- Local committed code mounted into Modal:
  `c4100de4496b6231fb77a3f3f1b0929500029f45`
- Mounted code snapshot: `dirty=false`, worktree patch bytes `0`, index patch
  bytes `0`
- Parameters: `epochs=50`, `batch_size=8`, `max_target_pairs=64`,
  `train_timeout_hours=2`, `timeout_hours=24`, CUDA, T13/T19 enabled,
  segmentation surrogate `sinkhorn`
- Claim state at launch: `active_dispatching` in
  `.omx/state/active_lane_dispatch_claims.md`
- Recover command:
  `.venv/bin/python experiments/modal_t1_balle_endtoend.py recover --label t1_balle_modal_guard_c4100de4_20260510T0915Z`

This dispatch is a bounded runtime/path guard, not a score claim. It remains
`score_claim=false`, `promotion_eligible=false`, and `rank_or_kill_eligible=false`
unless recovery returns exact contest-CUDA auth-eval JSON with schema blockers
at zero and closes the active claim terminally.

## Canonical A1 payload mounted

- Archive:
  `experiments/results/A1_canonical/harvested_artifacts/finetuned_archive/archive.zip`
  - bytes: `178262`
  - SHA-256:
    `87ec7ca5f2f328a8acdfc65f5cce0ab08a3a558eae88f36d4140870f141492b5`
- Checkpoint:
  `experiments/results/A1_canonical/harvested_artifacts/train/checkpoint_best_proxy.pt`
  - bytes: `925250`
  - SHA-256:
    `5846043656d8261855d58de6d9c3568c7b2c4ecabdc3b4b1729aef913f7cb272`
- Extracted latents:
  `experiments/results/A1_canonical/harvested_artifacts/extracted_frozen_latents.pt`
  - bytes: `69088`
  - SHA-256:
    `5ba13604837e27b834867d2ace06d7c21228e8b97d241da6744020bee9f79090`
- Designation memo: `.omx/state/canonical_a1_designation.md`
  - bytes: `3617`
  - SHA-256:
    `064842006b08e5d7d0527bfa3dc06c112ed3de97d423a1fd43f25a6169c2ea45`

## Current harvest state

Recovery attempts returned:

```text
NOT READY: call_id=fc-01KR8JMK531A9PECP0CV513KQM still queued or running. Re-run later.
```

Modal logs currently show the worker began repository initialization:

```text
Initialized empty Git repository in /workspace/pact/.git/
```

No terminal result has been harvested yet. Do not launch a duplicate T1 dispatch
while the active claim exists.

## Terminal harvest update

Recovery later closed the claim as
`failed_t1_modal_recovered_no_score_claim`. This is not score evidence:

- `score_claim=false`
- `promotion_eligible=false`
- `rank_or_kill_eligible=false`
- stage: `remote_script_failed`
- exact score fields: missing

The important positive custody signal is that the Modal scorer import probe
passed and the T4 worker exposed NVDEC:

```text
t1 modal scorer import probe OK
[probe_nvdec] OK (NVDEC exposed, DALI video pipeline buildable)
cuda: True
device: Tesla T4
```

The actual failure was CUDA OOM in the T8 Sinkhorn surrogate during Stage 5
score-domain training:

```text
torch.OutOfMemoryError: CUDA out of memory
...
src/tac/losses.py, sinkhorn_w2_mask_distortion_per_pixel
```

Classification: guard sizing / surrogate memory bug, not dependency closure,
not a model negative, and not a score result.

Immediate fix landed in this tranche: the Sinkhorn-W2 surrogate now chunks over
flattened spatial rows by default instead of building one full graph-wide
`(N, C, C)` tensor, and Modal runtimes set
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to reduce allocator
fragmentation. The next T1 guard rerun should keep the same bounded training
intent but uses the chunked Sinkhorn implementation.

## c0ea27df rerun

After commit `c0ea27df` landed the chunked Sinkhorn and Modal allocator fix, a
fresh bounded T1 guard was dispatched:

- Label / instance job id: `t1_balle_modal_guard_c0ea27df_20260510T0927Z`
- Function call id: `fc-01KR8KCXEGTVFGSZ75HAK9S2QX`
- Modal run URL:
  `https://modal.com/apps/adpena/main/ap-g6JJhRr82ENgaEaOcfduIu`
- Mounted code snapshot:
  `c0ea27dfaf06db1d143f30d950ac18abba334d7b`, `dirty=false`, worktree patch
  bytes `0`, index patch bytes `0`
- Parameters remain bounded: `epochs=50`, `batch_size=8`,
  `max_target_pairs=64`, `train_timeout_hours=2`
- Claim state at dispatch: `active_dispatching`

Immediate recovery returned:

```text
NOT READY: call_id=fc-01KR8KCXEGTVFGSZ75HAK9S2QX still queued or running. Re-run later.
```

Do not duplicate T1 while this claim is active. Harvest with:

```bash
.venv/bin/python experiments/modal_t1_balle_endtoend.py recover --label t1_balle_modal_guard_c0ea27df_20260510T0927Z
```

## Kaggle proxy materialization

The completed Kaggle/Optuna/CMA-ES proxy candidate has been converted only into
local handoff artifacts:

- Handoff:
  `experiments/results/kaggle_pr101_proxy_sweep_20260510_codex/pr101_proxy_sweep/local_materialization/archive_builder_handoff.json`
  - SHA-256:
    `5e3ee3974ece1011790e3604a402811649865563f10b87e2ae87716c18f39251`
- Manifest:
  `experiments/results/kaggle_pr101_proxy_sweep_20260510_codex/pr101_proxy_sweep/local_materialization/materialization_manifest.json`
  - SHA-256:
    `dc709594374c915ffa9c825b33cccff97a9d1e98cc9bc4a991eea2268f71b804`

The materialization boundary is explicit:

- `score_claim=false`
- `ready_for_exact_eval_dispatch=false`
- `archive_zip_emitted=false`
- `inflate_runtime_emitted=false`
- `contest_cuda_auth_eval=false`
- `dispatch_attempted=false`

The handoff exists only for a future archive builder that emits a byte-closed
archive/runtime packet and proves local inflate/runtime consumption before any
fresh exact-CUDA lane claim.

## Independent audit consequence

The parallel PR95/PR101 parity audit agreed that the current score-lowering
sequence should prove the bounded T1 guard path first, not jump to a full T1
run. If that guard clears, the next PR101-family packet-producing action is the
certified A2 sensitivity-weighted packet ladder. Existing A2 byte savings are
not enough for sub-0.17 by themselves and are blocked from exact-dispatch as
score evidence until stub/proxy sensitivity is replaced with certified
sensitivity.

## 2026-05-10T10:20Z c0 guard harvest + hardening follow-up

T1 guard `t1_balle_modal_guard_c0ea27df_20260510T0927Z` is now terminal:
`failed_t1_modal_recovered_no_score_claim`, with local dispatch summary
`active=0`. The result remains a guard/runtime finding only.

Evidence classification:

- import probe passed;
- NVDEC/CUDA surfaced on Tesla T4;
- Stage 5 score-domain training still OOMed inside the chunked
  `sinkhorn_w2_mask_distortion_per_pixel` path;
- no archive score, no promotion, no T1 model negative.

Adversarial diagnosis: the first chunked implementation reduced a single large
allocation but retained the autograd graph for all chunks, and the Modal guard
still used full-run-ish settings (`batch_size=8`, `max_target_pairs=64`) with no
trainer CLI to shrink Sinkhorn chunks. That is a guard sizing and actuator
configuration bug.

Hardening landed in this tranche:

- scorer-domain Sinkhorn chunk size is now plumbed from CLI to
  `scorer_loss_terms_btchw`;
- Modal guard labels fail closed unless `batch_size <= 1`,
  `max_target_pairs <= 8`, `train_timeout_hours <= 3`, and
  `sinkhorn_max_positions_per_chunk <= 2048`;
- Modal passes explicit mounted-code git custody into the remote script instead
  of synthesizing a fake `.git` directory;
- recover now separates exact-CUDA evidence validity from promotion
  eligibility, so `promotion_eligible=false` does not discard otherwise valid
  contest-CUDA evidence;
- provider dry-run templates set `T1_MOUNTED_CODE_GIT_*` and
  `SINKHORN_MAX_POSITIONS_PER_CHUNK=2048` for non-Modal launch paths.

Focused verification for this hardening:

```bash
.venv/bin/python -m pytest \
  src/tac/tests/test_sinkhorn_w2_segnet_surrogate.py \
  src/tac/tests/test_modal_t1_balle_endtoend.py \
  tests/test_dispatch_t1_balle_endtoend.py \
  tests/paradigm_delta_epsilon_zeta/test_eval_roundtrip_in_training_loop.py \
  src/tac/tests/test_build_component_sensitivity_manifest.py \
  tests/test_build_a2_sensitivity_weighted_pr101_packet.py \
  tests/test_probe_a2_packet_runtime_closure.py \
  src/tac/tests/test_audit_a2_packet_ladder_closure.py \
  src/tac/tests/test_component_sensitivity_artifact.py \
  tests/test_build_pr101_kaggle_proxy_runtime_packet.py -q
# 184 passed
```

Next T1 dispatch should use the stricter guard only:

```bash
PYTHONPATH=src:upstream:$PWD .venv/bin/modal run --detach \
  experiments/modal_t1_balle_endtoend.py --execute \
  --label t1_balle_modal_guard_<commit>_<stamp> \
  --epochs 50 \
  --batch-size 1 \
  --max-target-pairs 8 \
  --sinkhorn-max-positions-per-chunk 2048 \
  --timeout-hours 24 \
  --train-timeout-hours 2 \
  --cost-cap-usd 80
```

## 2026-05-10T12:40Z strict guard launched from 80b139c9

Strict T1 guard dispatch is now active:

- commit: `80b139c9`
- label / instance job id:
  `t1_balle_modal_guard_80b139c9_20260510T1240Z`
- Modal call id: `fc-01KR8YCW8F12KACG5TKWSNZ7A7`
- Modal run URL:
  `https://modal.com/apps/adpena/main/ap-QR661mpdFN68qaOLXXv2JC`
- bounded params: `epochs=50`, `batch_size=1`, `max_target_pairs=8`,
  `sinkhorn_max_positions_per_chunk=2048`, `train_timeout_hours=2`
- claim status after dispatch: `active_dispatching`
- immediate recovery poll: `NOT READY`, queued or running

Do not launch another T1 job while this claim is active. Harvest with:

```bash
.venv/bin/python experiments/modal_t1_balle_endtoend.py recover \
  --label t1_balle_modal_guard_80b139c9_20260510T1240Z
```
