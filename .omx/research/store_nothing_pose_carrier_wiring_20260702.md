# STORE-NOTHING-but-xi pose carrier — INTEGRATED into the launchable vehicle + triality (#205 A/B pose arm)

**Date:** 2026-07-02 · **Axis:** `[macOS-CPU advisory] NON-PROMOTABLE`. NOT a contest score. Canonical
frontier pointer **0.19110 UNMOVED**. `score_claim=false, promotable=false`. Every d_pose is the FROZEN
CPU-torch PoseNet authority (NEVER MPS); every byte is a real codec on real `gt_f0`/`gt_poses`.
**Means/ends:** these are MEANS (an A/B-able pose carrier). The END is a byte-closed `upstream/evaluate.py`
n600 exact row on a converged #205 witness where store-nothing's residual-closed d_pose + ~0 rate beats
the table carrier's S. **Held per CONTAINMENT: build + verify + commit ONLY; NO GPU, NO #205 launch.**

Integrates Track B's measured "store-nothing-but-ξ" pose finding
(`keyframe_rate_minimization_builds_20260702.md`, commit `18927a1ae`) into the four launch surfaces so
the #205 run can A/B it against the current `table` (warp-real-luma) carrier and MEASURE its real d_pose
through the byte-closed decode. Builds ON the byte-close agent's warp-real-luma PCAR1 decode
(`n205_pose_aware_byte_close_confirmed_20260702T221839Z.md`, `7a43b8844`).

---

## 0. TL;DR — the store-nothing carrier is wired end-to-end, byte-close BIT-EXACT, A/B-able

Store-nothing = store ONLY the ego twist ξ (+H) and warp the witness's OWN frame0 INR render by ξ (the
render is FREE, rule-118) — NO stored real keyframe. The keyframe payload (video-derived, COUNTED)
collapses to ~0 marginal bytes. **MEASURED (byte-close BIT-EXACT, n6/t1, this session):** on the SAME
checkpoint, store_nothing pose-carrier section **1049 B** (H+xi, 0 keyframe) vs warp_real_luma ds4
**697941 B**; archive.zip rate_term **0.0491** vs **0.5133**. frame0 DECODE BIT-EXACT
(`frame0_max_abs_uint8_diff = 0`). d_pose is #205-gated (Track B classmean proxy 4.97 pre-residual; the
witness render is richer → ≤4.97; the trained rank-6 dxi residual closes the offset — UNMEASURED, NO
borrowed number). `table` is INTACT (A/B), NOT removed.

---

## 1. What was wired on each surface (5 commits)

1. **BYTE-CLOSE** (`tools/levelset_byte_close_and_eval.py`, commit `a0c13355c`): new
   `--pose-carrier-mode {warp_real_luma,store_nothing}` (default warp_real_luma → byte-identical). In
   store_nothing the PCAR1 block stores H+xi with **n_keyframes=0** (NO stored keyframe luma); the
   shipped inflate + the numpy-fp32 oracle both GENERATE `frame0 = warp(the witness's OWN plain frame0
   render, per-pair H)`. New helpers: `pose_carrier_mode`, `pose_carrier_frame0_from_source`,
   `_dequant_blob`. `_cap_pose_carrier` guards empty keyframes; `pose_carrier_confirm` is mode-aware
   (store_nothing regenerates the witness-render frame0 authority from the blob via
   `numpy_oracle_reference_frames`; ceiling = warp(real gt_f0, H)). +4 tests (9 total).
2. **CONFIG** (`src/tac/witness_autoconfig.py` + `tools/launch_witness_run.py`, commit `3fabbb609`):
   `derive_store_nothing_205_config(...)` = `sealed_205` + `--pose-carrier-source generated`
   (`WitnessConfig.pose_carrier_source` field; emitted ONLY when != real_keyframe → `sealed_205` stays
   BYTE-IDENTICAL). `--config store_nothing_205` in the launcher. +2 tests.
3. **DSL gauge** (`src/tac/witness_dsl/gauge.py`, commit `0357fb306`): `PoseGauge.STORE_NOTHING_XI`
   chart + a byte-close-MEASURED cost cell (7200 B byte-optimal xi-only; compliant+deterministic =
   byte-close-PROVEN; a CARRIER that MOVES d_pose, NOT a dead sidecar). +2 gauge tests.
4. **EQUATIONS** (`src/tac/canonical_equations/store_nothing_pose_carrier_rate_dpose_20260702.py`, commit
   `0357fb306`): `store_nothing_pose_carrier_rate_collapse_vs_dpose_v1` (2 EmpiricalAnchors — the
   byte-close rate collapse + the Track B pre-residual d_pose; consumers = gauge + autoconfig; producers
   = byte-close + ladder tool). +5 tests.
5. **TRAINER** (`experiments/train_levelset_witness_realized_through_R_mlx.py`, commit `dd3aa824f`):
   `--pose-carrier-source {real_keyframe,generated}` (default real_keyframe → byte-identical). In
   `generated` mode the render dispatch warps the witness's OWN plain frame0 render (up to camera-native
   via `apply_contest_faithful_roundtrip_nhwc(output_hw=CAMERA_HW)` — the R "up" step == the byte-close
   `_R`) by the carrier twist; the `dxi` residual co-grads THROUGH the witness render. `_pc_verdict_f0_uint8`
   is mode-aware (store_nothing warps `_fwd_numpy(f0 code) → _torch_R_to_camera_uint8`, matching the
   byte-close decode). +1 MLX render-path test (shape contract + warp active).

**Triality:** DAG **FEED-snx** (`.omx/research/sub015_DAG_*`) ↔ **DSL** `PoseGauge.STORE_NOTHING_XI` ↔
**equations** `store_nothing_pose_carrier_rate_collapse_vs_dpose_v1` — all three AGREE (the byte-close
section 1049 B / rate 0.0491 appears in the DAG narrative, the gauge cost provenance, AND the equation
anchor).

---

## 2. MEASURED — the byte-close store-nothing rate + d_pose-parity confirmation (n6/t1, BIT-EXACT)

Same t1 checkpoint (`experiments/results/levelset_pose_smoke_20260627T070546Z/t1`, n_pairs=6), gt_n6.

| carrier | pose-carrier section | keyframe bytes | archive.zip | rate_term | frame0 decode bit-exact | d_pose carrier |
|---|---:|---:|---:|---:|---|---:|
| **store_nothing** | **1049 B** | **0** | 73758 B | **0.0491** | **True (max_abs=0)** | 189.65 |
| warp_real_luma ds4 | 697941 B | 696931 B | 770889 B | 0.5133 | True (max_abs=0) | 172.66 |

- **RATE COLLAPSE (the store-nothing win):** the pose-carrier section drops **697941 B → 1049 B** (665×);
  the archive rate_term **0.5133 → 0.0491**. store_nothing stores ONLY H+xi (byte-optimal: xi-only 12
  B/pair, H derived FREE via exp_se3 → 7200 B @ n600); the whole real-keyframe payload is GONE.
- **DECODE BIT-EXACT / TRAINING↔DECODE PARITY:** `frame0_max_abs_uint8_diff = 0` — the shipped inflate's
  store-nothing warp of the witness render == the numpy-fp32 oracle bit-for-bit (both the general
  bit-exact gate over 6 frames AND the pose-carrier CONFIRM). raw frame0 == oracle frame0 AND raw frame1
  == witness render ⇒ the training-side warp d_pose EQUALS the real-decode d_pose — NO surrogate gap.
- **d_pose is #205-gated (honest, NO over-claim):** on the t1 SMOKE witness (w_pose=0, untrained render)
  the store-nothing d_pose ≈189.6 (≈ null — warping a garbage render adds no pose info); warp_real_luma
  warps the REAL keyframe → 172.66. BOTH are POSE-BLIND on the smoke. The real store-nothing d_pose→low
  needs a CONVERGED #205 witness (good render) + the trained dxi residual (Track B classmean proxy 4.97
  pre-residual; witness render richer → ≤4.97). This is INFRASTRUCTURE + rate confirmation, NOT a
  promotable d_pose row. Reports: `.omx/tmp/store_nothing_smoke/report_store_nothing.json`.

---

## 3. The exact A/B command the operator runs (once the OOM fix lands + a converged #205 checkpoint exists)

**Byte-close A/B** on the SAME converged witness (compares the exact-eval S — rate + realized d_pose — of
the two carriers):

```bash
# store-nothing arm (stores ONLY xi/H)
.venv/bin/python tools/levelset_byte_close_and_eval.py \
  --ckpt-dir <converged_205_run_dir> \
  --gt-cache experiments/results/mlx_fleet_gt_cache/gt_n600.npz \
  --pose-carrier --pose-carrier-mode store_nothing \
  --verify-bit-exact --run-exact-eval

# table (warp-real-luma) arm — the sealed_205 carrier
.venv/bin/python tools/levelset_byte_close_and_eval.py \
  --ckpt-dir <converged_205_run_dir> \
  --gt-cache experiments/results/mlx_fleet_gt_cache/gt_n600.npz \
  --pose-carrier --pose-carrier-mode warp_real_luma --pc-keyframe-downscale 4 \
  --verify-bit-exact --run-exact-eval
```

**Train the store-nothing arm** (so the dxi residual co-adapts to the witness-render warp; the faithful
A/B) — NOTE: the dxi residual trained for gt_f0 (table) is NOT optimal for the witness render, so a
faithful A/B trains BOTH arms:

```bash
# store-nothing training arm
TAC_MLX_CUSTOM_GROUPED_BACKWARD=1 .venv/bin/python tools/launch_witness_run.py \
  --config store_nothing_205 \
  --gt-cache experiments/results/mlx_fleet_gt_cache/gt_n600.npz --num-pairs 600 --epochs 1000

# table (sealed) arm
TAC_MLX_CUSTOM_GROUPED_BACKWARD=1 .venv/bin/python tools/launch_witness_run.py \
  --config sealed_205 \
  --gt-cache experiments/results/mlx_fleet_gt_cache/gt_n600.npz --num-pairs 600 --epochs 1000
```

(store_nothing_205 == sealed_205 + `--pose-carrier-source generated`; same seg/curriculum/optimizer, ONLY
the pose STORE gauge differs → a clean d_pose + rate attribution.)

---

## 4. Provenance / reproducibility

- **Byte-close data:** `experiments/results/levelset_pose_smoke_20260627T070546Z/t1` (n=6),
  `experiments/results/mlx_fleet_gt_cache/gt_n6.npz`. d_pose authority = frozen CPU-torch PoseNet
  (`cpu_verdict_d_pose_batch`, NEVER MPS). Warp = `tac.boundary_math.warp_real_luma_frame0`.
- **Store-nothing warp SOURCE (both training + decode):** the witness's OWN plain frame0 render
  up-sampled to camera-native (874×1164) — training-side via MLX
  `apply_contest_faithful_roundtrip_nhwc(output_hw=CAMERA_HW)`; decode-side via the numpy
  `levelset_rgb_forward_numpy` + `_R` — then the SE(3) ground-homography warp by the twist.
- **NO borrowed number:** the ancestor-RGB 3.4e-5 is NOT reproduced; the post-residual store-nothing
  d_pose is OPEN (#205-gated). All rows advisory/research-signal; pointer **0.19110 UNMOVED**.
- **Tests:** byte-close 9, config 8, gauge 34, equation (canonical-eq suite) 258, trainer render-path 2 —
  all GREEN. ruff F821 clean on every touched file.
