# Yousfi levers PREPARED (additive, default-off) for the level-set witness — BUILD + $0-validate

**UTC:** 2026-06-27T05:54:42Z · **Author:** levers-build subagent (FEED-dc) · **Axis:** `[macOS-CPU advisory] / [macOS-MLX training-gradient]` NON-PROMOTABLE · **Pointer:** UNMOVED 0.19110.

**Scope.** Prepare 3 "Yousfi levers" as **ADDITIVE, default-OFF** config on the level-set witness so the
NEXT GPU run (n600 / capacity-sweep / rescue) can fire with them on, WITHOUT disturbing the running
`levelset_n96_mod32` process (pid 72600, GPU — untouched; the old code stays in its memory). All edits are
additive: new optional flags default to current behavior. Files touched:
`experiments/train_levelset_witness_realized_through_R_mlx.py`,
`src/tac/boundary_math/lever_b_levelset_generator.py`, and a new test
`src/tac/boundary_math/tests/test_levelset_yousfi_levers.py`.

**Re-ranking (per FEED-db/FEED-dd order-exploit measurement + coordinator):** RATE IS NOT BINDING at the
witness operating point (d_seg contribution ~0.256 dominates rate ~0.059 ~4:1; witness payload is a small
fraction of the frontier 177 KB). So the SCORE-impact ranking is **LANE-EDGE (top, d_seg gate) > CHROMA
(#2, d_seg) > STEM-NYQUIST (#3, free-bytes polish — biggest rate lever but rate isn't the gate).**

---

## LEVER 1 (TOP→ now #2): CHROMA — d_seg actuator (VERIFIED, was already wired)

**Flag:** `--chroma` (BooleanOptionalAction, **default True** — already ON). The trainer already had this; this
task VERIFIED it works end-to-end + byte-closes.

**$0 smoke (n6, 3 ep, CPU, render 96×128, chroma ON default):** runs end-to-end through R → frozen CPU-torch
SegNet verdict, **finite d_seg** (0.55→0.67 @ 3 ep — 3-ep noise, NOT a verdict). Self-orient reorient fired
(mean|dir| 0.64). No errors.

**Byte-close compatibility (numpy ONE-CODEPATH `levelset_rgb_forward_numpy`, the forward a level-set
inflate/byte-close uses):**
- phi (SDF/argmax) **identical** with/without chroma → chroma is a pure color realization, does not perturb
  the partition structure.
- RGB **differs materially** (max|Δ| = **209.7**/255) → chroma carries genuine argmax-relevant signal (the
  comma10k boundary colors are chroma-saturated → real d_seg actuator, confirms operator "Chroma too").
- achromatic mode → R==G==B (correct BT.601 luma collapse); chroma mode keeps R≠G.
- **blob bytes are chroma-INDEPENDENT** (`out_tex` is 3-ch either way) → chroma is a **FREE-rate** d_seg lever.
- int8-dequant round-trip render is finite + chroma-aware → byte-close safe.
- `--chroma` persists to the npz cfg (`__cfg_chroma`); the level-set numpy forward + blob quantizer are
  chroma-aware. **GAP (not built here):** a level-set-specific inflate.py packet builder analogous to
  `tools/witness_byte_close_and_eval.py` does not yet exist (that tool is RGB-witness-only, keyed on
  `out.weight`); the level-set witness byte-closes via its OWN primitives (`levelset_rgb_forward_numpy` +
  `quantize_levelset_blob` + `int8_dequant_params`), which ARE the ONE CODEPATH the verdict already uses.

**Does chroma move d_seg on the smoke?** Not measurable at 3 epochs (noise); the render-level proof above
shows chroma is a real actuator (RGB Δ 209.7, partition-invariant). Efficacy needs the real GPU run.

**Expected score impact:** d_seg lever (#2). Free bytes. Re-measure any prior witness d_seg verdict with chroma
active (CLAUDE.md: chroma-off verdicts are provisional).

---

## LEVER 2 (was top → now #3, DEMOTED): STEM-NYQUIST — the rate lever (rate isn't binding)

**New code (additive):**
- `stem_nyquist_max_freq_cycles_per_unit(scorer_w=512, stem_stride=2)` → `f_max = scorer_w/(4·stem_stride)`.
- `curvelet_directional_B(cfg, max_freq=None)` — drops atoms with |f| > max_freq (default None = no cap =
  current behavior; never empties the bank). Threaded into `LevelSetConfig(max_freq=None)`.
- Trainer flag `--max-bank-freq` (default None).

**Nyquist math (rigorous).** EfficientNet-B2 stride-2 stem → finest feature map ~256×192; its Nyquist is 128
cycles across the 512-wide SegNet input. Coords ∈ [-1,1] (span 2): a feature sin(2πf·x) makes 2f cycles
across the width, so `2·f_max = 128 → f_max = 64 cycles/unit`.

**Where the over-Nyquist waste is (MEASURED):**
- The **default curvelet bank** (f0=2, base=2, n_scales=4) maxes at **16 cyc/unit = 4× BELOW Nyquist** →
  capping the curvelet bank is a **no-op** (smoke C confirmed: `--max-bank-freq 64` → cols 40→40). The bank
  is already conservative.
- The **self-orient directional feats** are the waste: `--freq-across 32 --n-dir-freqs 6` (the running run's
  config) gives across-edge freqs **{32, 64, 128, 256, 512, 1024}** — **4 of 6 octaves (128…1024) are
  OVER the 64 Nyquist**: detail SegNet structurally cannot see + actively ALIASING under R (high-freq →
  uint8 @ camera → off-boundary argmax flips = the d_seg killer).

**Param / byte deltas (deterministic, n600; int8-raw == n_params, brotli q11 estimate):**

| config | in_feat | decoder params | total | blob (brotli) | Δblob |
|---|---|---|---|---|---|
| CURRENT (H96, nd6, freq32) | 104 | 73 463 | 111 863 | 97 063 B | — |
| LEAN-DIR (H96, nd2 @ freq32, Nyq) | 88 | 71 927 | 110 327 | 96 625 B | −438 B |
| LEAN-DIR (H96, nd4 @ freq8, Nyq) | 96 | 72 695 | 111 095 | 96 329 B | −734 B |
| LEAN-H64 (nd4) *capacity arm* | 96 | 40 279 | 78 679 | 68 895 B | **−28 KB** |
| LEAN-H48 (nd4) *capacity arm* | 96 | 27 143 | 65 543 | 56 822 B | **−40 KB** |

**$0 smoke C (n6, lean-dir freq8/nd4 + `--max-bank-freq 64`):** runs end-to-end, finite d_seg, `stem_nyquist`
stage print fired; blob **73 696 B (nd6) → 72 929 B (nd4) = −767 B** at smoke scale (matches the −734 B n600
analysis direction).

**Honest framing (NO-FAKE).** The Nyquist-justified part (cap the over-Nyquist directional octaves) saves only
**~0.4–0.7 KB rate** — its real value is **anti-aliasing (a d_seg help)**, not rate. The BIG rate lever is the
decoder `hidden_dim` shrink (H96→H48 = −40 KB blob, −41% total), but that is an **EMPIRICAL capacity cut (the
bc20/bc36 trilemma), NOT a Nyquist-free lunch** — whether d_seg survives is the capacity question, so it is a
**separate capacity-sweep arm**, not folded into the main launch. Since rate is not binding, this lever is
free-bytes polish; land it (free is free) but don't over-invest.

**Does d_seg stay comparable while bytes drop?** Bytes drop (deterministic, measured). d_seg comparability is
NOT measurable at 3 epochs — needs the real run; the lean-dir cap is expected to HELP d_seg (anti-alias).

---

## LEVER 3 (NEW TOP, #1): LANE-EDGE fragility weighting — the d_seg gate

**New flags (additive, default-OFF):** `--lane-edge-weight 0.0` (0 = OFF = current behavior),
`--lane-edge-class 1` (Lane in the contest order [Road0, Lane1, MyCar2, Undrivable3, Movable4]),
`--lane-margin-target 0.5`.

**Mechanism (REALIZED, through R).** When `lane_edge_weight > 0`, `total_loss_fn` adds an ADDITIVE term that
renders f1 → R → frozen SegNet logits, takes the **live decision margin** `signed = gt_logit − max_competitor`
ONLY where GT == lane (class 1), and penalizes `relu(margin_target − signed)·lane_mask` (mean over lane px).
The hinge fires exactly on **small-margin (fragile = boundary) lane pixels** → adds gradient pressure to widen
the lane margin at the lane double-edges — the EXACT d_seg gate FEED-db pinpointed (lane class-1 consecutive-
frame IoU **0.263**; 83% of all partition change in the margin<2 boundary band = 4.7% of px). Lane (19% of
flips) is under-fit because the Yousfi/CE baseline trains CrossEntropy with NO class weighting.

**Mechanism soundness (standalone numpy mirror of the mx term):** counts exactly the lane pixels (mask selects
only class-1); hinge fires on the fragile lane-boundary pixel (margin 0.1 < target 0.5 → hinge 0.4), does NOT
fire on the lane-interior pixel (margin 5.0 → 0); road pixels contribute 0 (not lane); 0 + finite when no lane
pixels; 0 when all lane margins large. **MECHANISM SOUND = True.**

**$0 smoke A (n6, 3 ep, CPU, `--lane-edge-weight 50`):** the `lane_edge` stage print fired (`"active": true`),
ran end-to-end through R + the 2nd realized seg forward, finite verdict, no errors.

**Cost.** When ON, the lane term costs a **2nd realized seg forward** (render+R+SegNet) per pair (acceptable per
operator "score > training time"; the optimal-form fusion into the base seg loss needs a parent-trainer edit
— `make_loss_fn` in `train_witness_realized_through_R_mlx.py` — which is out of scope for this additive prep).

**Expected score impact:** **#1.** Directly targets the d_seg gate (lane class-1 IoU 0.263). Tune
`--lane-edge-weight` to its own optimum (per optimal-form discipline) before any verdict; suggested starting
range 25–50.

---

## POSE (reference only — not built here)

Pose is SOLVED by the stored-target Quantizr sidecar (store 6 PoseNet scalars/pair + supervise the render via
`--w-pose>0`, d_pose 3.4e-5). FEED-db measured a **per-column fixed-point bit-allocation** pose sidecar
(bits [11,5,5,4,4,5] + Δ + brotli) → **~2.3 KB, d_pose-neutral** (added MSE 1.8e-6 < fp16's 6.18e-6) vs the
current 6.8 KB fp16 sidecar = −0.0030 S free. Reference that for the byte-close; not this subagent's build.

---

## READY-TO-FIRE "Yousfi-levers-ON" launch command (next n600 / sweep / rescue)

Base config = the running good config (n96→n600). Levers ON: LANE-EDGE (#1) + CHROMA (#2, already default) +
conservative STEM-NYQUIST directional cap (#3, `--freq-across 32 --n-dir-freqs 2` = keep the 2 sub-Nyquist
octaves {32,64}, drop the 4 over-Nyquist aliasing octaves; `--max-bank-freq 64` defensive no-op):

```bash
env TAC_MLX_CUSTOM_GROUPED_BACKWARD=1 .venv/bin/python -u \
  experiments/train_levelset_witness_realized_through_R_mlx.py \
  --out-dir experiments/results/levelset_n600_yousfi_levers_$(date -u +%Y%m%dT%H%M%SZ) \
  --gt-cache experiments/results/mlx_fleet_gt_cache/gt_n600.npz \
  --num-pairs 600 --epochs 1500 --render-h 384 --render-w 512 \
  --hidden-dim 96 --mod-dim 32 --activation hosc --siren-init \
  --softmax-temp-start 1.0 --softmax-temp-end 0.05 \
  --curriculum --tau-softplus-start-epoch 300 --l7-start-epoch 900 \
  --palette-anchor --self-orient --reorient-every 50 \
  --freq-across 32 --n-dir-freqs 2 --freq-along 4 --max-bank-freq 64 \
  --chroma \
  --lane-edge-weight 30 --lane-edge-class 1 --lane-margin-target 0.5 \
  --w-seg 100 --w-pose 0 --eikonal-weight 0.01 --length-weight 0.001 \
  --ema-decay 0.997 --accum-pairs 8 --grad-clip 1.0 --verdict-pairs 96 \
  --mlx-device gpu --eval-every 25
```

Wrap with `tools/safe_run.py --rss-mb <cap> --timeout <s> --label levelset_n600_yousfi --` per the
scale-safeguard discipline (do NOT launch while pid 72600 holds the GPU).

**Sweep arms (separate dispatches, NOT the main launch):**
- **Capacity / rate arm (lever 2 big cut):** add `--hidden-dim 64` (→ −28 KB blob) or `--hidden-dim 48`
  (→ −40 KB blob) — EMPIRICAL capacity test (does d_seg survive the band-limited shrink?).
- **Lean-dir arm:** `--freq-across 8 --n-dir-freqs 4` (4 finer sub-Nyquist octaves vs the conservative 2).
- **Lane-weight optimum:** sweep `--lane-edge-weight {15,30,50}` to its own optimum before any verdict.

---

## Validation summary

- All edits ADDITIVE, default-OFF → the running pid-72600 process + any in-flight config are unaffected
  (defaults reproduce current behavior; confirmed by callers grep — only new threading uses the new kwargs).
- `py_compile` OK; 7/7 new tests pass; 19/19 existing `test_lever_b_generator.py` pass; preflight introduces
  no new failure on the 3 touched files (a pre-existing `CodebaseDriftError` about unrelated `launch_*.py`
  files is not mine).
- `$0` CPU smokes only (`--mlx-device cpu`, n6/3ep, gt_n6 cache); GPU never touched; numpy-fp32 verdict;
  smoke scratch cleaned (rebuildable). NO score/frontier/promotion claim; pointer UNMOVED 0.19110.
