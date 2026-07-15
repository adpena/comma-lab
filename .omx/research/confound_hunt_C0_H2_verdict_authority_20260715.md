# Confound Hunt — C0 baseline, Hunter H2: MEASUREMENT AUTHORITY / VERDICT VALIDITY

**Run:** `experiments/results/levelset_n600_witness_20260715T095030Z/` (pid 72377, C0).
**Git HEAD:** `cacff6c1a2`. **Date:** 2026-07-15. **$0, report-only, no training/dispatch/score-claim.**
**Verdict argv (launch.sh):** `--verdict-pairs 0 --verdict-batch 32 --async-verdict --verdict-device cpu --eval-every 25 --verdict-anchor-every 0 --ema-decay 0.997 --num-pairs 600` (`--self-orient` ABSENT → OFF; `--verdict-live-gap-every` ABSENT → 0/OFF).
**Runtime state at hunt time:** alive, ETIME 13:49, RSS ~26.6 GiB; last log `mem_probe before_v0_verdict n_pairs=600 verdict_batch=32 rss_gib=23.68`. No verdict rows emitted yet (first at ep25; v0 pending).

Sibling scopes NOT touched: H1 (liveness/guards/schedule-firing), H3 (lever-efficacy/control-validity), H4 (config-drift/resume/loss-scale).

---

## L3 VERDICT-CLEARANCE PRECONDITION (the apparatus-validity contract C0's d_seg/d_pose MUST satisfy to be an AUTHORITY)

A C0 verdict row is an admissible AUTHORITY (advisory, non-promotable) iff ALL hold:
1. **n600** — verdict over all 600 pairs, not a subset. **✓ SATISFIED** (code + runtime cite below).
2. **through-R** — rendered fp32-numpy → `_torch_R_to_camera_uint8` → frozen scorer (realized-through-R, not a proxy). **✓ SATISFIED**.
3. **EMA-shadow** — scored on `ema.shadow`, not live weights. **✓ SATISFIED** (`parameter_key="ema_np"`).
4. **CPU-torch, NOT MPS/MLX** — `DistortionNet().eval()` on `torch.device("cpu")`, fp32, `inference_mode`. **✓ SATISFIED**.
5. **epoch-stamp matches the scored state (not a stale checkpoint)** — snapshot captured point-in-time on the MAIN thread; row stamped with that epoch; best-ckpt preserved from the SAME snapshot. **✓ SATISFIED**.

**The ONE precondition NOT auto-satisfied for EARLY rows: EMA warmup (see confound C-H2-1).** Every precondition above holds; the residual risk is INTERPRETATION of early rows, not the measurement mechanism.

**Positive-control sentinel status: WEAK.** The precise sentinel (EMA-vs-live `live_gap` / gpu-vs-cpu `verdict_anchor`) is DEFAULT-OFF in C0. Only the coarse v0-init anchor remains (init must score badly).

---

## RANKED FINDINGS

### C-H2-1 — CONFOUND (mild, INTERPRETATION-scope): early-run EMA-shadow-lag risk with the in-run positive-control DISABLED
- **Signature:** DEFAULT-OFF × SILENT × verdict-interpretation-corrupting. The verdict scores `ema.shadow` (decay 0.997); the shadow lags live weights by ~1/(1−0.997) ≈ 333 updates. An EARLY d_pose (or d_seg) row can RISE while training loss falls — the KNOWN EMA-shadow-lag artifact (DAG anchor "early-run verdict-d_pose RISE while training-pose falls = EMA-shadow lag; CONFIRMED run-1"). A reader who reads an early RISE as pose-regression is WRONG.
- **Cited fact:** `--verdict-live-gap-every` is ABSENT from launch.sh → argparse `default=0` (`train_levelset...py:12940`) → OFF. The built-in EMA-vs-live sentinel (`_verdict_live_gap_every`, lines 3650/8731–8732/12090) that would emit `d_seg_live`/`d_pose_live` + `d_pose_ema_minus_live` and DIRECTLY clear the lag is therefore never computed. The gpu-vs-cpu `verdict_anchor` is also OFF (`--verdict-anchor-every 0`, cpu-device run). So C0 has NO active precise positive control for verdict validity — only the coarse v0-init anchor.
- **Poison-scope:** early-epoch d_pose/d_seg TRAJECTORY READING only (not the converged best-ckpt d_seg, which the L68 pose-banked physics and eventual shadow≈live make sound). Does NOT poison the final verdict; DOES risk a spurious "pose getting worse early" mis-narration and, worst case, a premature intervention keyed off an early rise.
- **L1 fix:** default-ON `--verdict-live-gap-every` (cheap: one extra inference every Kth verdict) at least during EMA warmup, so `d_pose_ema_minus_live` is emitted on early rows — turning the silent lag LOUD. Per the "off is a tracked queue, observability defaults ON when score-neutral" discipline: this live-gap row is score-neutral telemetry, so default-off is the orphan bug, not caution.
- **L2 fix:** a preflight/emit-side check that stamps early verdict rows (where elapsed EMA updates < ~2/(1−decay)) with an `ema_warmup=true` caveat field so no reader/controller treats an early RISE as a converged signal.
- **L3 fix (binding):** verdict-clearance precondition — no "pose regressing" / "lever hurts pose" verdict is admissible off an early row unless the `live_gap` sentinel is ON or EMA warmup has elapsed. (This is the ANCESTOR of the run-1 lesson; encode it as the apparatus-validity gate.)

### C-H2-2 — MINOR NOTE (borders H1 liveness; stated once, not duplicated): async verdict cadence self-throttles → trajectory may have GAPS
- **Cited fact:** `_schedule_async_verdict` (line 8713) SKIPS a scheduled verdict when a prior one is still in flight (`verdict_skip` row, `_verdict_skipped`), so under load the realized-verdict cadence is SPARSER than `--eval-every 25`. Each emitted row is honestly epoch-stamped; the risk is a reader interpreting a MISSING row (skip) as a plateau. Flagged for cross-check with H1; no verdict-authority poison (the rows that ARE emitted are valid).

---

## CLEAN SURFACES (proof-cited)

- **CLEAN-1 — `--verdict-pairs 0` is truly ALL n600, NOT a silent subset.** `train_levelset...py:7357-7358`: with `verdict_pairs=0`, line 7357 leaves `vpairs=[0]`, then line 7358 `... if args.verdict_pairs else list(range(P))` → `0` is falsy → `vpairs = list(range(600))` = all 600. Argparse help (`:12976`) documents this is the C12 confound fix (2026-07-05): a prior 24-pair default "violated the n600 non-negotiable at the number that DEFINES the goal." **Runtime confirmation:** `mem_probe before_v0_verdict n_pairs=600 verdict_batch=32` in run.log — the live process is configured over all 600 pairs. The allergic-to-non-n600 discipline is satisfied.

- **CLEAN-2 — `--verdict-batch 32` chunking is bit-identical to full-batch.** `_verdict_dseg_dpose_chunked` (`train_levelset...py:246`) means over the concatenation of per-chunk per-pair lists, contiguous 0..n, same order as the single-batch list. The per-pair scalars are batch-size-independent because (verified `train_witness...py:753-810`): scorers are `DistortionNet().eval()` → BatchNorm uses RUNNING stats (no cross-frame interaction), `argmax(dim=1)` is per-pixel, MSE is per-pair, all under `torch.inference_mode()`. Same values + same order → `float(np.mean(...))` bit-identical. The docstring claim was independently re-derived from the code, not trusted.

- **CLEAN-3 — CPU-torch is the numpy-fp32-equivalent authority, NOT MPS/MLX.** Scorers loaded `DistortionNet().load_state_dicts(posenet_sd, segnet_sd, torch.device("cpu"))` (`train_witness...py:630/695`); the trainer binds `gt, seg_cpu, posenet_cpu = load_gt_from_cache(...)` (`train_levelset...py:3686`), and the verdict runs `cpu_verdict_d_seg_batch(seg_cpu, ...)` / `cpu_verdict_d_pose_batch(posenet_cpu, ...)` in fp32 (`.float()`). `--verdict-device cpu` selected. The `gpu` verdict device is FAIL-CLOSED against `--async-verdict` (`train_levelset...py:3667-3679`, `raise SystemExit`), so the async+MLX GPU-stream race is structurally impossible; the run stays on CPU authority. No MPS anywhere in the verdict path.

- **CLEAN-4 — self-orient / cf-feats cache staleness is MOOT for C0.** run.log emits `"self_orient": false` / `"basis": "legacy_fourier_ab_control"` → `use_self_orient=False` → `dir_feats_per_pair=None` (`train_levelset...py:4121`). The async snapshot's `"dir"` is `None` (`:8364`) and `_feats_for_snapshot` returns the STATIC `curv_feats_np` (`:4155`). No reorient cache exists to go stale; the verdict feats are the fixed curvelet chart. (For a future self-orient run: the reorient cache IS consistent-by-design — the deployed byte-closed artifact freezes the same dir_feats at the last reorient, so the verdict faithfully measures the deploy contract; not a confound, but re-check when self-orient turns on.)

- **CLEAN-5 — async does NOT introduce a stale-checkpoint misread.** `_schedule_async_verdict` (`train_levelset...py:8713`) captures the snapshot on the MAIN thread point-in-time (`snap["ema_np"]` = materialized numpy copy of the shadow at call time), stamps the emitted row with that epoch, and `_maybe_preserve_best` preserves the shadow from the SAME snapshot the verdict scored — so best-ckpt selection-metric and preserved weights cannot desync. The pending-verdict record persists the exact snapshot inputs → a `--resume-from` recomputes the row bit-identically (`:8746-8758`). The worker reads ONLY its own copies + constants (never `ema.shadow`/`model`/`cf_mx_cache` while the main loop mutates them) → race-free, and the verdict is PURELY OBSERVATIONAL (training never reads it) → BIT-IDENTICAL weights/checkpoints.

---

## VERDICT (H2)
C0's verdict apparatus is a SOUND authority on all five mechanism preconditions (n600 · through-R · EMA-shadow · CPU-not-MPS · epoch-faithful). The single live risk is INTERPRETATION of EARLY rows under EMA-shadow lag (C-H2-1), made worse because the in-run positive-control (`live_gap`) that would clear it is DEFAULT-OFF and no other precise sentinel is active. Recommend defaulting `--verdict-live-gap-every` ON during EMA warmup (L1) + the early-row EMA-warmup clearance precondition (L3). No verdict-authority-POISONING confound found; the final best-ckpt d_seg remains a valid advisory number.
