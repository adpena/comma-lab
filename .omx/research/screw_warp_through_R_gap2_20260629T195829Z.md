# SCREW/TWIST warp d_seg THROUGH R (GAP-2) — the PRE-R lower bound → the REAL through-R number

**UTC** 2026-06-29T195829Z · **authority** `[macOS advisory / CPU-torch research-signal]` · **pointer UNMOVED 0.19110**
**score_claim** false · **promotable** false · **ready_for_exact_eval_dispatch** false
**Tool** `tools/measure_screw_warp_through_R.py` (new; reuses a513372a's screw-homography fit machinery)
**JSON** `experiments/results/screw_warp_through_R_n96/results.json` + `…_n200/results.json`
**Authority** FROZEN CPU-torch SegNet (`load_real_segnet("cpu")` + `measure_segnet_argmax`), NEVER MPS.
**Settles** the decisive next step a513372a named: take the screw warp THROUGH the contest R operator.

> **What a513372a left open.** a513372a (`screw_analysis`) proved the single-ego-twist per-class stratified
> warp is a ~0-byte parameterization WIN PRE-R, in LABEL space, pure numpy. But it had **no R operator**, so
> the binding Lane-survival residual (the ~0.58 lower bound) and the real bulk-warp number through the
> contest scorer were **not measurable**. This converts that PRE-R lower bound into the REAL through-R number.

## Method (the through-R conversion)

For each pair p: warp **gt_f0** (the true previous frame, native 874×1164 RGB) by the SAME screw-derived
per-class homographies → R (uint8@874 → scorer `preprocess_input` bilinear↓384) → **frozen CPU-torch SegNet**
→ argmax → d_seg vs **gt_f1 `lstars`** (`= SegNet(gt_f1)` = L*).

- **Geometry is CLEANER than a513372a's across-pair label probe.** We warp WITHIN a pair (gt_f0 frame 2p →
  gt_f1 frame 2p+1); `gt_poses[p]` is the raw PoseNet 6-vector estimated FROM that exact pair, so it IS the
  f0→f1 relative ego-motion — **no adjacent-pose proxy** (a513372a had to use `poses[p+1]`).
- **Per-class regime** (a513372a's `SCREW_REGIME`): Road/Lane/Movable → ground plane-induced homography
  `H=K(R−t nᵀ/d)K⁻¹`; Undriv(sky, Z→∞) → rotation-only `H=K R K⁻¹`; MyCar(hood) → identity (= naive copy).
  Each distinct regime = ONE frozen-SegNet forward on the warped RGB through R; per-target-class-c d_seg is
  read under regime(c). Faithful through-R analog of a513372a's `dseg_for_target_class`.
- **Calibration** = 3 global scalars (s_t,s_r,pitch) fit in LABEL space (warp lstar0→lstars on Road+Lane,
  full-coverage persist-fallback; lstar0=`SegNet(gt_f0)`), then APPLIED to the RGB warp. Within-pair fit:
  n96 `s_t=−0.00148, s_r=0, pitch=−0.07`; n200 `s_t=−0.00143, s_r=0, pitch=−0.05`.
- **R for a camera-res warp witness:** gt_f0 is already at 874, so the contest's bicubic↑874 is **identity**
  here; the acting R = warp bilinear interp + uint8@874 + scorer bilinear↓384. (Flagged: excludes the
  sub-874 bicubic↑ aliasing a low-capacity sub-camera INR would add.)
- **NO-FAKE self-check (PASSED both samplings):** `SegNet(gt_f1) == lstars` EXACTLY (0 px disagreement) →
  the SegNet pipeline is byte-faithful to the cache authority. The tool ABORTS rather than report a number
  if this fails.

## Results — two independent samplings AGREE

**Through-R per-class d_seg (n96; n200 in the same direction):**

| class | regime | area | naive-copy thru R | **screw thru R** | contrib (×area) | warp vs naive |
|---|---|---|---|---|---|---|
| Road | ground | 0.230 | 0.0155 | **0.0142** | 0.00326 | **−8%** (helps) |
| Lane | ground | 0.006 | 0.3934 | **0.3896** | 0.00229 | −1% (barely) |
| Undriv(sky) | rotonly | 0.493 | 0.0016 | 0.0016 | 0.00078 | ~0 |
| Movable | ground | 0.016 | 0.0347 | 0.0348 | 0.00054 | ~0 (independent motion) |
| MyCar(hood) | identity | 0.256 | 0.0028 | 0.0028 | 0.00072 | = (identity) |
| **TOTAL** | | | **0.00792** | **0.00760** | | warp helps |

| sampling | pre-R label screw TOTAL | **through-R screw TOTAL** | through-R naive TOTAL | bulk-warp term | lane term | movable term |
|---|---|---|---|---|---|---|
| n96 (within-pair) | 0.00744 | **0.00760** | 0.00792 | **0.00477** | 0.00229 (flip 0.39) | 0.00054 |
| n200 strided | 0.00788 | **0.00797** | 0.00831 | **0.00512** | 0.00228 (flip 0.39) | 0.00057 |

## The three KEY outputs the task asked for

### (a) The BULK-warp-through-R term (Road+sky/Undriv+hood/MyCar) = **0.0048 (n96) / 0.0051 (n200)**
This is the budget's bulk-warp term — the classes the screw handles deterministically. **Through R+SegNet,
the warp HELPS Road ~8% over naive-persist (0.0155→0.0142)**, sky/hood are unchanged (correctly routed to
rotonly/identity). The bulk through-R ≈ the bulk pre-R: **R + SegNet are essentially NEUTRAL on the bulk**
(neither destroy nor recover it materially), confirming F1's "R is benign at scorer res" for the bulk.

### (b) The LANE-through-R d_seg (the binding term) = **flip ≈ 0.39** (n96 0.3896 / n200 0.3901)
This converts a513372a's 0.58 lower bound into the real number — **with an honest refinement.** The 0.58 was
a513372a's looser ACROSS-pair geometry (2-frame-apart + adjacent-pose proxy). The cleaner WITHIN-pair pre-R
label lane is already **0.395**, and through R+SegNet it is **0.390** — i.e. **R/SegNet barely move the lane
vs the within-pair label warp (0.395→0.390); the warp barely helps (naive 0.393→screw 0.390).** So the
0.58→0.39 drop is dominated by the cleaner geometry, NOT by SegNet recovering lanes from texture. **The lane
is ~39% flip through R+SegNet and is WARP-UNEXPLAINABLE** — it MUST be the trained-witness / stored residual.
This is GAP-2, the binding wall, now a real number.

### (c) Movables (stored, F3) = contrib **0.00054 / 0.00057** — fit comfortably; the warp can't predict
independent motion (warp ≈ naive), consistent with "store small movables" (F3).

## Budget implication (does the bulk fit under 1.23e-3, leaving room for the lane?)

**NO — not via warp-of-the-previous-frame.** The bulk-warp-through-R term (0.0048–0.0051) is **~4× the
1.23e-3 d_seg total budget**, and the lane (0.0023) on top blows it further (full screw total 0.0076–0.0080
≈ **6× budget**). The decomposition: bulk 0.0048 (≈63% of total, Road-dominated) + lane 0.0023 (≈30%) +
movable 0.0005 (≈7%).

**Why (the deep cause, honest + NOT a kill):** the **naive-persist** baseline (store the previous frame's
SegNet output) already totals 0.0079–0.0083 — the **inter-frame SegNet boundary-jitter floor**. The screw
warp barely beats it (0.0076–0.0080) because at 384 res the f0→f1 ego shift is small and the residual is
dominated by texture-dependent boundary flicker the ego-warp cannot remove. **The frontier (PR95) achieves
d_seg ~6e-4 by STORING/RENDERING each frame's partition (per-pair latent), NOT by warping a neighbor** —
existence proof that sub-budget d_seg IS reachable, just not by previous-frame warp.

**What this means for the v2 vehicle (refines, does not kill):**
1. **The LANE is confirmed as the trained-witness/stored residual** (warp-unexplainable, 0.39 flip, ~2×
   budget alone). Consistent with the v2 memo's "trained INR shrinks to ONLY the Lane-survival residual."
2. **The bulk is NOT free via previous-frame warp.** The v2 claim "bulk needs NO INR" is an
   **upper-bound-falsifying** result for the previous-frame-warp: the bulk warp residual (Road inter-frame
   jitter 0.0048) is real and above budget. The **decisive untested follow-up** is the **clean
   stored-canonical-warp** (warp a temporally-aggregated jitter-free canonical, not the noisy previous
   frame) — a temporally-aggregated canonical could undercut the 0.0155 naive Road jitter floor. My number
   is the previous-frame-warp UPPER BOUND; canonical-warp is the next $0 measurement.
3. **Movables → store (F3).** Settled.

## Honest caveats / NO-FAKE (this is a MEANS, not the end)
- **Warps GT RGB, not a shipped witness RGB.** Bounds the deterministic bulk-warp part (the true previous
  frame is a RICHER source than a single stored canonical → bulk is upper-bound-ish) and gives a realistic
  Lane-through-R signal. The authority is realized-through-R inside the witness INR + exact CPU/CUDA eval on
  byte-closed bytes — NOT this advisory probe.
- **Per-target-class-stratified DIAGNOSTIC**, not a single blind-routed composite render. Each target-class-c
  pixel is scored under regime(c) (the right "does regime(c) predict class c?" question). A real witness
  routes static classes via a stored hood-mask/sky-region descriptor (the screw byte accounting) — a
  separate step.
- **Camera-res R model** excludes sub-874 bicubic↑ aliasing of a low-capacity sub-camera INR.
- **Calibration = 3 global scalars** fit in label space, applied to RGB; the raw PoseNet 6-vector column
  interpretation [fwd,lat,vert,r0,r1,r2] is the flagged INFERRED assumption (col0 dominant forward).
- `[macOS advisory / CPU-torch research-signal]`; pointer **0.19110 UNMOVED**. This redirects the v2 witness
  build; it is not a byte-closed exact row. **The exact pointer did not move.**

## Single most decisive next step
**The clean stored-canonical warp.** This probe warps the NOISY previous frame; its bulk residual (0.0048)
is dominated by inter-frame SegNet jitter that NO previous-frame warp removes. Build a temporally-aggregated
**clean canonical partition** (mode/median over the clip, or a stored reference L*) → warp IT by the per-pair
screw twist → R → SegNet → measure the bulk d_seg. If the canonical-warp bulk drops under ~1.23e-3, the v2
"bulk needs no INR" holds and the trained witness is lane-only; if not, the bulk needs per-pair correction
too. (Secondary: lane carrier through R — F1 measured SDF lane@render-192 ≈ 0.032; compose that with this
warp's lane residual.)

## rule-118 tags
- **FREE (generic, expandable in inflate.py, uncounted):** the plane-induced-homography (LHP) + expmap +
  per-class regime selection + warp + R chain — a deterministic geometric algorithm.
- **COUNTED-but-EXISTING:** the per-pair 6-DOF pose (already stored for d_pose; the screw adds NO per-pair
  payload).
- **COUNTED-but-TINY:** the static scene descriptor (n, d, hood-mask, calibration s_t/s_r/pitch) — once/clip.
- **NOT FORBIDDEN:** honest geometry, not a smuggled per-frame argmax/warp table.

## Wire-in hooks (Catalog #125)
1. **sensitivity-map** ACTIVE — per-class warp-vs-naive through-R deltas are new sensitivity rows (Road
   −8% warp-explainable; Lane −1% warp-unexplainable = the binding axis). 2. **Pareto** ACTIVE — the
   `WarpGauge.SCREW_TWIST` cell (`src/tac/witness_dsl/gauge.py`) now records the measured through-R d_seg
   in provenance (bulk 0.0048 / total 0.0076 / lane 0.39); the rate↔distortion arm screw(~0 byte) ↔ d_seg
   is updated. 3. **bit-allocator** ACTIVE — spend bytes on the LANE residual (binding) + small Movables,
   NOT on the bulk warp params (free) NOR fighting R on the bulk (neutral). 4. **cathedral autopilot** N/A
   (advisory probe). 5. **continual-learning** this memo + the gauge-cell provenance + DAG FEED-jk.
   6. **probe-disambiguator** — `tools/measure_screw_warp_through_R.py` IS the disambiguator (pre-R label
   vs through-R; bulk vs lane vs movable; warp vs naive); the clean-canonical-warp is the next disambiguator.

## Primary citations
- a513372a screw probe (`.omx/research/screw_twist_warp_dseg_probe_20260629T192609Z.md`); F1 R-survival
  (`.omx/research/r_survival_physics_20260629T182659Z.md`).
- Longuet-Higgins & Prazdny 1980 (ego-flow split); Hartley & Zisserman plane-induced homography
  `H=K(R−t nᵀ/d)K⁻¹`; openpilot/comma2k19 EON intrinsics (fx=fy=910, pp=(582,437) @ 1164×874).
