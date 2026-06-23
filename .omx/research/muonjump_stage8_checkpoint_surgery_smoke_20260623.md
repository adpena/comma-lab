# Muon-jump: stage-8-positioned checkpoint surgery + CPU smoke verification (2026-06-23)

**Authority:** infrastructure, not a score axis. `[contest-CPU advisory]` NON-PROMOTABLE,
$0, local CPU only (NO MPS — the live run pid 79893 owns the MPS GPU). No PR, no exact
contest eval. This builds + verifies a resumable checkpoint; the parent does the
sensitive stop+launch of the full MPS run.

## Goal (operator-approved "jump to Muon now")

Skip the long PR95 stages 5-7 rate sweep (rate prize ~0.005 of S, cheap + recoverable
from the small basis) and jump directly into the genuine **stage-8 Muon finetune** — the
d_seg finisher that is the binding ~0.115-of-S term — seeded from the preserved stage-5
weights.

## Why this is GENUINE, not a fake harness

The driver (`tac.torch_vehicle.driver.TorchVehicleDriver.run`) runs the IDENTICAL stage-8
code path on a normal resume. A checkpoint positioned at `stage_index=7 / epoch_in_stage=0`
with the WEIGHTS + EMA + Lever-4 sensitivity-EMA KEPT and the OPTIMIZER state CLEARED, fed
to `driver.run()`, runs the bit-identical stage-8 Muon-finetune path — just from stage-5
weights. Three structural facts (verified by source read):

* **resume loop** `for stage_index in range(resume_pos.stage_index, len(...))` (driver.py
  ~3286) → `stage_index=7` runs ONLY stage 8.
* **`_restore_into`** (driver.py ~2546) ALWAYS restores decoder/latents/EMA from the
  checkpoint (lines 2550-2555); it SKIPS optimizer/scheduler restore when
  `merged["adamw"] is None` (line 2578) — the existing, tested **FORK-SEED** protocol
  ("fresh optimizers from these weights at this curriculum position").
* **`_build_stage_runtime`** (driver.py ~1784) builds Muon + AdamW FRESH when
  `spec.use_muon` (stage 8 `use_muon=True`) — exactly a natural stage-7→8 boundary.

So no divergent code path: every byte of stage-8 train/eval/export runs through
`driver.run()` unchanged. The surgery only sets the resume cursor + clears the optimizers
(the documented fork-seed signal). Curriculum confirmed: stage index 7 = `stage8_muon_finetune`,
`use_muon=True`, `use_qat=True`, `init_latents_random=False`, epochs=5000, muon_lr=2e-4.
Sum of PR95 stages 0-6 epochs = 24650 (the expected `_global_epoch` at stage-8 entry).

## INPUTS

* preserved stage-5 ckpt: `experiments/results/yousfi_r3_taper_marginhinge_e5_20260620/stage_snapshots/stage5_jump_point_20260623T144601Z/`
  (stage_index=4 stage5_c1a_l7, epoch_in_stage=4925, has_muon=false, base_channels=20,
  latent_dim=28, n_pairs=600, taper=[16,16,17,19,19,14,10], muon_lr_floor_fix=true, ema_decay=0.999,
  best_score=0.30522 at best_ep=14850/best_stage=4).
* surgery script: `tools/build_stage8_muonjump_checkpoint.py` (committed dc09193bb).

## STEP 1 — surgery output (NEW_DIR)

`experiments/results/yousfi_r3_MUONJUMP_stage8_20260623T145220Z/`

New manifest: `stage_index=7`, `epoch_in_stage=0`, `stage_name=stage8_muon_finetune`,
`has_muon=false` (optimizers cleared), `base_channels=20`, `latent_dim=28`, `n_pairs=600`,
`taper_channels=[16,16,17,19,19,14,10]`, `muon_lr_floor_fix=true`, `ema_decay=0.999`,
`best_score=inf`/`best_ep=0`/`best_stage=-1` (reset to track the jump run's own best).

Tensor-level carry verification (offline, $0):

* decoder: 30 tensors **bit-identical** to source.
* ema_decoder: **bit-identical**.
* latents (600,28): **bit-identical**.
* ema_latents (600,28): **bit-identical**.
* optimizers: adamw=None, muon=None, adamw_sched=None, muon_sched=None (fork-seed signal).

## STEP 2 — CPU smoke verification (the genuine stage-8 path)

Ran the launcher on a SEPARATE COPY of NEW_DIR (so NEW_DIR stays at clean epoch-0),
CPU-only, mirroring the live run's argv (`--no-split-by-head --seg-margin-hinge
--muon-lr-floor-fix --stage-lr-warmup-frac 0.03 --taper-channels 16,16,17,19,19,14,10
--defer-batch-sync`) with `--train-device cpu --device cpu --eval-every 1
--checkpoint-every-epochs 1` for a fast first eval. Smoke dir:
`experiments/results/yousfi_r3_MUONJUMP_smoke_20260623T145220Z/`.

### Verification table — ALL 5 PASS

First inline exact-eval row (epoch_in_stage=1, wall_clock 1161.6s on CPU), preserved at
`.omx/research/muonjump_stage8_smoke_first_eval_row_20260623.jsonl`:

```json
{"stage_index": 7, "stage_name": "stage8_muon_finetune", "epoch_in_stage": 1,
 "global_epoch": 24651, "d_seg": 0.0020790269691497087, "d_pose": 0.0002112958422852292,
 "rate": 0.0021147680351160164, "score": 0.30673882529354607, "archive_bytes": 79400,
 "muon_lr": 2.1197041309735388e-05, "grad_norm_muon": 1.351638674736023,
 "adamw_lr": 1.0598520654867694e-06, "is_best": true,
 "authority_tag": "[contest-CPU advisory]", "promotable": false}
```

| # | Check | Expected | Observed | Result |
|---|-------|----------|----------|--------|
| a | stage jumped | stage_index==7, stage8_muon_finetune | `stage_index=7`, `stage_name=stage8_muon_finetune` | **PASS** |
| b | global_epoch position | starts ≈ 24650 | `global_epoch=24651` (24650 base + 1 completed epoch) | **PASS** |
| c | Muon built fresh | has_muon→True after ≥1 step; muon_lr populated | `muon_lr=2.12e-05`, `grad_norm_muon=1.35` (Muon optimizer live) | **PASS** |
| d | **CARRY CHECK** | first exact-eval d_seg ≈ 0.00207 (NOT ~0.5); d_pose ≈ 0.0002 | `d_seg=0.0020790`, `d_pose=0.0002113` — NOT ~0.5 → **weights carried** | **PASS** |
| e | QAT active | stage 8 use_qat True | curriculum stage-8 `use_qat=True` (confirmed); rate=0.00211 / 79.4KB | **PASS** |

**The decisive carry check (d): d_seg = 0.0020790 ≈ the preserved checkpoint's value, NOT
~0.5.** The stage-5 weights + EMA carried into the genuine stage-8 Muon-finetune code path
intact. The first row is already `is_best=true` at score 0.30674 (≈ the preserved stage-5
best 0.30522, expected — the rate sweep stages 5-7 were skipped so the rate term is the
small basis's, and the Muon d_seg-finishing polish has only run 1 epoch). The descent from
here is the genuine stage-8 finisher.

Tensor-level carry was ALSO verified offline before + after the smoke (decoder 30 tensors,
EMA, latents, ema_latents all bit-identical to source; NEW_DIR untouched by the smoke).

## STEP 3 — NEW_DIR left clean for relaunch (verified)

The smoke ran in a SEPARATE copy dir, so NEW_DIR was NEVER touched — no reset needed.
Post-smoke verification (offline, $0): `checkpoint_exists(NEW_DIR)=True`,
position `stage_index=7/epoch_in_stage=0`, decoder carry bit-identical to source,
optimizers cleared (adamw/muon both None), `best_score=inf`. NEW_DIR holds the CLEAN
stage-8/epoch-0 checkpoint; the parent launches the full MPS run from epoch 0.

Cleanup: the ~1.8MB smoke scratch dir was removed (certified rebuildable — a copy of
NEW_DIR + 1 smoke epoch; the verification row was preserved to
`.omx/research/muonjump_stage8_smoke_first_eval_row_20260623.jsonl` before deletion). The
live run (pid 79893) was untouched throughout (confirmed alive, elapsed 1d+).

## EXACT MPS relaunch command (parent)

```
.venv/bin/python -u experiments/launch_split_by_head_basin.py \
  --no-split-by-head --train-device mps --device cpu \
  --base-channels 20 --latent-dim 28 --n-pairs 600 \
  --targets-cache experiments/results/capstone_gt_targets_cache \
  --async-eval --eval-every 25 --checkpoint-every-epochs 25 \
  --muon-lr-floor-fix --seg-margin-hinge \
  --stage-lr-warmup-frac 0.03 --taper-channels 16,16,17,19,19,14,10 \
  --defer-batch-sync \
  --out-dir experiments/results/yousfi_r3_MUONJUMP_stage8_20260623T145220Z
```

(`--kd-warm-start-dir` dropped — KD-warm-start applies only at a FRESH stage-0 init; it is
bypassed on resume by the `not resuming_into_this_stage` guard, driver.py ~3308.)
