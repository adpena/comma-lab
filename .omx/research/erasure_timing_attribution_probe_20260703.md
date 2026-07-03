# Erasure-Timing Per-Stage Attribution Probe (#253) — landed 2026-07-03

**Tool:** `tools/erasure_timing_attribution.py` · **Tests:** `src/tac/tests/test_erasure_timing_attribution.py` (19 pass)
**Status:** BUILT + validated at small scale (n24). `advisory_non_promotable` — a per-stage
ATTRIBUTION probe, NOT a frontier/score claim. The exact pointer (0.19110) moves only through
`upstream/evaluate.py`. The full n600 erasure curve runs LATER on the SEALED per-stage checkpoints
(operator-coordinated — do NOT stack a 2nd heavy n600 job while a train run is live, #246).

## What it measures

A `$0` post-hoc tool: given a set of per-stage witness checkpoints from a #205 level-set run, it
turns them into an **erasure-timing CURVE** so the surgical-lever A/B becomes a per-stage attribution
DAG. For EACH checkpoint it measures — **REALIZED THROUGH THE REAL R OPERATOR + the FROZEN CPU-torch
SegNet argmax on the exact rendered bytes** (numpy fp32 ONE CODEPATH; NO proxy; NO direct-argmax
mirage; NEVER MPS) — three long-tail metrics at n600 (small for validation):

1. **Per-class d_seg** — decomposes the d_seg flip mass by the 5 canonical comma10k classes
   (which classes flip). Reports `flip_count`, `frac_of_total_pixels`, `frac_of_dseg` per class.
2. **Lane / Movable RECALL** — for the rare classes Lane(1) + Movable(3):
   `recall = correct GT-class px / GT-class px`. Low recall = ERASED (the long-tail metric).
   Also reported for all 5 classes.
3. **Finest-scale island BIRTH / SURVIVAL** — two complementary views:
   - **connected-component islands** (`scipy.ndimage.label`, 8-conn) of the GT rare-class mask,
     sized; `n_islands`, `n_fine_islands` (size ≤ `--fine-size-thresh`), `fine_island_survival_rate`
     (fraction of fine islands with ≥50% pixels correct), `mean_fine_island_recall`.
   - **#218 birth-death persistence** (`h0_superlevel_persistence`) of the GT margin field within
     the rare-class mask → survival binned by persistence tercile (low = finest = dashes). The
     `error ∝ 1/persistence` law: low-persistence survival is the erasure headline.

### The 3-stage BIRTH / SURVIVE-TRAINING / SURVIVE-R pipeline (task #253)

- **BIRTH** — is the fine feature ever argmax-correct at a stage (early stage / ep0 seed)?
- **SURVIVE-TRAINING** — the erasure-timing curve: survival vs stage/epoch. Alive early then dead
  = erased DURING training (the per-stage attribution).
- **SURVIVE-R** — `--measure-r-survival`: also SegNet the RAW render (pre-R) → `d_seg_pre_R` +
  `r_roundtrip_erasure_gap = d_seg_realized − d_seg_pre_R`. Isolates the R (bicubic-camera→uint8→
  bilinear) roundtrip erasure from training/representation erasure.

Also emits `d_seg_witness_sdf_internal` — the witness's OWN out_sdf argmax vs GT — as a **diagnostic**
(NOT a pre-R SegNet pass; for a store_nothing witness the realized d_seg rides the RGB→SegNet path,
so this internal argmax need not match GT and its gap is EXPECTED to be large, ~0.32 measured).

## Canonical reuse (do-not-reinvent)

- `tac.boundary_math.lever_b_levelset_generator.levelset_rgb_forward_numpy` — the numpy ONE CODEPATH
  (bit-mirror of the MLX witness + the byte-close/inflate decode) → `(rgb, phi)`.
- `…lever_b_levelset_generator.{build_coords, curvelet_directional_B, curvelet_feats}`
- `tac.local_acceleration.torch_levelset_inflate.dir_feats` — the self-orient fixed point.
- `tools.levelset_byte_close_and_eval.{_load_levelset_ckpt, detect_self_orient, _bank_cfg, CAMERA_H, CAMERA_W}`
- `tac.boundary_math.seg_core.load_real_segnet("cpu")` + the batched CPU-torch SegNet argmax verdict
  (mirror of `train_witness_realized_through_R_mlx.cpu_verdict_d_seg_batch`; chunked per the #205 OOM law).
- `tools.birth_death_persistence_dseg.h0_superlevel_persistence` — #218 persistence.
- GT cache `experiments/results/mlx_fleet_gt_cache/gt_n{96,600}.npz` — keys `lstars` (GT SegNet argmax
  384×512), `margins` (top1−top2 field, #141-style), read once, `[:P]` sliced.

## CANONICAL CLASS ORDER — comma10k MEASURED (a 3×-recurring bug)

`0=Road 1=Lane 2=Undrivable(incl sky) 3=Movable(cars) 4=MyCar(hood)`.
It is **FORBIDDEN** to re-derive by luma-sorting `class_values=[41,76,90,124,161]`. Validated
self-consistently below: the per-class support ordering (Undrivable > MyCar > Road ≫ Movable > Lane)
exactly matches the CLAUDE.md MEASURED per-class areas — NOT the luma-sort.

## Small-scale validation result (CORRECTNESS proof — advisory, non-promotable)

Ran on the R1 EMA checkpoint
`experiments/results/levelset_n600_R1_storenothing_descent_ev1_20260703T004906Z/levelset_witness_ema_BEST.npz`
(ckpt sha16 `bfb520fea869b365`, git `9a10cf062`, gt_cache `gt_n96.npz`).

**n24, `--islands both`** (164 s CPU): `d_seg_realized = 0.00369` (matches R1's descent ~0.0045).

| metric | value |
|---|---|
| per-class flip mass (frac of d_seg) | Road **0.405** · Lane **0.333** · Undrivable 0.116 · Movable 0.089 · MyCar 0.058 |
| recall | Lane **0.806** · Movable 0.979 · Road 0.994 · Undrivable 0.999 · MyCar 0.999 |
| support px (class-order sanity) | Undrivable 2,330,251 > MyCar 1,208,975 > Road 1,077,102 ≫ Movable 72,381 > Lane 29,883 |
| island_cc (Lane) | 561 islands, 418 fine, **fine_survival 0.237** (finest lane dashes largely ERASED) |
| island_cc (Movable) | 77 islands, 12 fine, fine_survival 0.417 |
| persistence bins (survival) | low_pers_finest **0.818** < mid 0.901 < high 0.934 (monotone = error ∝ 1/persistence) |

**n4 + `--measure-r-survival`:** `d_seg_realized 0.00380 = d_seg_pre_R 0.00380`,
`r_roundtrip_erasure_gap ≈ −0.00000` → **R is near all-pass** for this witness (corroborates the
measured "R all-pass" finding). The residual erasure is training/representation, NOT the R roundtrip.

Interpretation (advisory): the finest lane islands (dashes) are the erasure long-tail — 80.6% Lane
pixel recall but only **23.7%** of the FINEST lane islands survive, and low-persistence features
survive least (0.818) — exactly the dash-erasure crux, per-stage-attributable when run on the SEALED
stage checkpoints.

## Exact commands

### Full n600 erasure curve on a #205 run's per-stage checkpoints (RUN LATER, coordinated)
```bash
.venv/bin/python tools/erasure_timing_attribution.py \
  --run-dir experiments/results/<205_RUN_DIR> \
  --stage-glob 'levelset_ckpt_*.npz' --fallback-ema \
  --gt-cache experiments/results/mlx_fleet_gt_cache/gt_n600.npz \
  --num-pairs 600 --verdict-batch 32 --islands both --measure-r-survival \
  --out experiments/results/<205_RUN_DIR>/erasure_curve_n600.json
```
(Per-stage checkpoints save as `levelset_ckpt_<stageTag>_ep<N>.npz`: stageCE / stageTau / stageL7 /
stageHinge / stageMuonStart. The tool sorts the curve by epoch. `--verdict-batch 32` chunks the SegNet
forward per the #205 OOM law. NOTE: n600 × camera-res CPU-SegNet ≈ tens of minutes per checkpoint —
do NOT launch while a heavy n600 TRAIN job is live; #246 hard-blocks a 2nd heavy job.)

### `--compare` per-lever attribution (which STAGE a lever moves d_seg)
```bash
.venv/bin/python tools/erasure_timing_attribution.py \
  --compare experiments/results/<BASELINE_SEALED_RUN> experiments/results/<LEVER_WARMSTART_RUN> \
  --stage-glob 'levelset_ckpt_*.npz' \
  --gt-cache experiments/results/mlx_fleet_gt_cache/gt_n600.npz \
  --num-pairs 600 --verdict-batch 32 --islands both \
  --out experiments/results/<LEVER_WARMSTART_RUN>/erasure_compare_vs_baseline.json
```
Emits `delta = B − A` per matched `stage_tag` for d_seg / lane-recall / movable-recall / fine-lane
island-survival / low-persistence survival → BIRTH (ep0 seed) vs SURVIVE-TRAINING (persistence) vs
SURVIVE-R (R gap) per lever. Build-the-hook now; the deep per-lever use lands on the SEALED checkpoints.

### Small-scale validation reproduction (this landing)
```bash
.venv/bin/python tools/erasure_timing_attribution.py \
  --checkpoints experiments/results/levelset_n600_R1_storenothing_descent_ev1_20260703T004906Z/levelset_witness_ema_BEST.npz \
  --gt-cache experiments/results/mlx_fleet_gt_cache/gt_n96.npz \
  --num-pairs 24 --verdict-batch 24 --islands both \
  --out experiments/results/erasure_timing_probe_validation_n24_<utc>/curve.json
```

## NO-FAKE / authority tags

- Every d_seg is REALIZED through the real R (`_R_to_camera` = op-for-op mirror of the shipped inflate
  `_R`) + the frozen CPU-torch SegNet argmax on the exact rendered bytes — NO proxy, NO MPS
  (`_assert_segnet_cpu` refuses non-CPU params; unit-tested). Deterministic (numpy fp64 render, torch
  CPU bicubic/SegNet, scipy) — CPU-locked authority (MLX-GPU is NOT bit-identical cross-process).
- The free curvelet bank must reproduce the trained `in_proj` width or the tool REFUSES (NO-FAKE).
- Every row carries provenance: git hash, checkpoint sha256, gt-cache sha256, seed, so-overrides,
  render_hw, class-order dict, and the `advisory_non_promotable` authority string.
- The validation is a CORRECTNESS proof, NOT a finding. The real per-stage erasure curve is the later
  n600 run on the SEALED #205 per-stage checkpoints (operator-coordinated).
