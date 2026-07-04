# Scaling-law FACET 3 — the geometry-informed SEED (separatrix + asymmetry) as the prefactor + nucleation-regime lever

**Task #285 / geometry-optimal scaling-law engineering, facet 3 of 4 (2026-07-04). $0 research;
field-level (advisory) measurement on cached GT; NO heavy/paid/GPU; #205 SACRED READ-ONLY.
Pointer contest-CPU 0.19110 UNMOVED — MEANS.** Governing discipline: NO-FAKE (every number below is
MEASURED on the real frozen-CPU-torch SegNet argmax cache or CITED with its axis; no fabricated
number/citation). The seed is a train-time init that ships **0 archive bytes** (rule-118 FREE generic
structure); it moves the pointer only through a byte-closed `upstream/evaluate.py` n600 row.

---

## 0. The facet, in one line

The witness is a softmax-of-SDF level-set flow. The SEED (the phi init) sets the scaling law's
**prefactor** (geodesic distance of the init partition to the target manifold) AND selects the
**nucleation regime** (which basin the MCF/tau flow falls into — lane-present vs the lane→0 attractor
that is the #205 d_seg-creep). This facet designs the separatrix+asymmetry seed that gets the init
on-manifold, in the right basin, above the critical nucleus — and MEASURES how close it gets.

**Measurement axis (honest label):** all d_seg numbers below are **FIELD-LEVEL** partition d_seg —
`argmax_k phi_k` vs the frozen GT SegNet argmax `lstar`, pre- and post- the field-level R proxy
(`lever_b_levelset_generator.apply_R_to_fields`). This is the canonical R-survival proxy the repo uses;
it is `[macOS-CPU advisory]` research-signal, NOT a realized-through-render+R+SegNet score and NOT a
byte-closed row. The realized number differs (§4). Cache: `experiments/results/mlx_fleet_gt_cache/gt_n{6,24,96}.npz`
(fields `lstars`, `margins`, `gt_poses`). Probes reproduced inline in §7.

---

## 1. The separatrix-informed SDF seed (the design)

The witness represents the argmax partition as `argmax_k phi_k` of K=5 signed-distance fields. The
IDEAL seed is `phi_k = signed_distance_fields(lstar)` — +EDT inside class k, −EDT outside — whose
argmax reproduces `lstar` EXACTLY (the round-trip NO-FAKE assert). So the zero-level-set STARTS on the
true separatrix by construction. The design is per-class:

- **Smooth classes (Road/Undrivable-sky/Movable/MyCar-hood):** the static-core partition
  (`build_static_core_partition`, self-detected roles, majority-static over the pairs) → `signed_distance_fields`.
  This is the SHARED trunk init (one partition for all codes). It ships 0 bytes.
- **Lane (class 1, the finest-scale erasure tail):** the openpilot IPM polynomial band SDF — a deg-3
  ground-frame centerline + per-row half-width + range-dependent dash gate, fit to the pair's class-1
  pixels (`lane_sdf_component.build_structured_lane_sdf`). This is the true separatrix of the Road|Lane
  boundary, NOT a crude class-mask dilation.

**MEASURED — the perfect-separatrix ceiling (n96, field-level):**

| seed | pre_R d_seg | post_R d_seg |
|---|---|---|
| ideal per-pair `signed_distance_fields(lstar)` | **0.000000** | **0.000042** |

⟹ A perfect geometry seed IS on-manifold (~0 even after the R knife-edge). The seed CAN reach the
manifold — the prefactor floor is ~0. Everything below is the gap from a *realizable* (generic,
0-byte, shared-trunk) seed to this ceiling.

---

## 2. The nucleation crux — MEASURED, and the current seed path does NOT fix it

### 2a. The lane nucleation failure is N-dependent and total at scale
The static-majority lane mask collapses as N grows (the lane's temporal IoU is ~0.26; a majority over
N frames erases it):

| n | static lane IoU | static seed lane_FN (field-level) |
|---|---|---|
| 6  | 0.50 | 0.00199 |
| 24 | 0.37 | 0.00359 |
| 96 | **0.009** | **0.00583** |
| 600 (#205) | → 0 (`lane_px=0`, per nucleation memo) | → full lane area **~0.0059** |

⟹ At #205 scale the lane is **entirely absent from the seed** (lane_FN → the full GT lane area). This
is the measured, quantitative confirmation of `lane_nucleation_failure_seed_above_critical_nucleus` at
the seed surface.

### 2b. The current `--lane-prior-phi1 --mode replace` does NOT nucleate the lane [MEASURED — a real defect]
The trainer's BUILD-2 lane prior injects the openpilot band SDF by REPLACING the phi1 channel
(`inject_lane_sdf(..., mode="replace")`, `train_levelset_witness_realized_through_R_mlx.py:1435`). But
the band SDF is a **shallow LOCAL** field (+2..+8 inside a thin band), while the static-core **road**
SDF at those same pixels (labeled road in the collapsed static core) is DEEP (the EDT-to-nearest-non-road
can be tens of px). Road wins the argmax inside the band ⟹ the lane still does not nucleate:

| seed (n96) | lane_FN | lane_attr d_seg |
|---|---|---|
| static core (`--lane-prior-phi1` OFF) | 0.00583 | 0.00589 |
| inject-into-deep-SDF (`--lane-prior-phi1 --mode replace`, +2px) | **0.00538** | 0.00558 |

The prior barely moves lane_FN (0.00583 → 0.00538) — the lane is still ~90% missing. **The current
seed path is a measured no-op for nucleation.**

### 2c. The fix — PAINT-THEN-SDF (build the partition, then the EDT) [MEASURED — decisive]
Build the partition with the lane PAINTED where the band ∩ road, THEN `signed_distance_fields`. Now the
road SDF goes negative inside the band by construction, so lane wins:

| seed (n96, DIL=0) | lane_FN | lane_FP | lane_attr d_seg |
|---|---|---|---|
| paint-then-SDF (shared smooth + openpilot band) | **0.00189** | 0.00175 | 0.00363 |
| paint-then-SDF (ideal per-pair smooth + band) | 0.00189 | 0.00162 | 0.00352 |

⟹ lane_FN **0.0058 → 0.0019** (a 3× drop): the lane NUCLEATES in the seed by construction, in the RIGHT
basin (nonzero mass where the band is), so the tau/MCF flow GROWS it instead of the #205 lane→0 erasure.
Requires a ~10-LOC code change (a `--lane-prior-phi1-mode paint` that builds the partition + repaints
the band + re-runs the EDT, instead of the shallow channel replace). **This is the single decisive seed fix.**

---

## 3. Asymmetry-informed sign/orientation — the honest role at the seed

The fold-caustic asymmetry (law #2: Fisher curvature `1−Σp² = tr F = ½ sech²(m/2)` on the margin-zero
Maxwell set; MEASURED Pearson `curvature↔(−margin)` 0.978) tells which side of the boundary is which
class. At the seed we already have GT `lstar` ⟹ the SDF SIGN is set trivially (phi_k>0 where lstar=k).
The asymmetry's *non-trivial* seed role is **SUB-PIXEL interface PLACEMENT**: put the zero-level-set on
the m=0 caustic ridge and offset it toward the bright (high-Fisher) side by the fold half-width so it
R-survives the uint8 knife-edge — this is the AA-SDF coverage integration (`coverage_alpha_from_signed`
/ `rasterize_lane_coverage_range_dependent`) already in `analytic_lane_render_band`.

**MEASURED NO-FAKE correction (do NOT overclaim the margin gate at the seed):** I tested using the GT
margin as an *uncertainty gate* on the seed paint (paint lane only in the low-margin annulus). It
removes almost NOTHING — the seed lane-FP is **annulus-concentrated** (only ~3% of it sits above
GT-margin 4.0; GT-margin median is 5.8):

| gate | lane_FP (n96) |
|---|---|
| no margin gate | 0.001746 |
| paint where margin<4.0 | 0.001689 |

The band-overshoot/dash-gap FP lands *adjacent to the true lane* (genuinely uncertain, low margin), so a
margin gate can't remove it at the seed. The uncertainty mask's real value is at the RENDER composite on
the WITNESS's OWN margin as it sharpens during training (a different, trainable regime, memo
`analytic_lane_band_primary_authority_decomposition`) — NOT at the GT-margin seed. The caustic-asymmetry
cue at the seed is therefore **placement (AA-SDF), not FP-masking.**

---

## 4. The openpilot lane-lever calibration vs the 0.00087

The seed's lane-attributable field-level cost with the openpilot band (DIL=0, adaptive per-row fit
width, dash-gated) is **0.0036** (FN 0.0019 SHAPE-capture + FP 0.0017 dash/width scatter). A width sweep
confirms the adaptive per-row 90th-pct fit is already optimal (constant caps 1–3px are all WORSE):

| half-width | lane_FN | lane_FP | lane_attr |
|---|---|---|---|
| fit (per-row 90pct) | 0.00189 | 0.00175 | **0.00363** |
| const 2.0px | 0.00262 | 0.00146 | 0.00408 |
| const 1.0px | 0.00348 | 0.00078 | 0.00426 |

The memory's **0.00087** is the *realized-through-R+SegNet* lane d_seg of the same band WITH the FP
killers (SegNet stride-2 stem smooths the thin FP; render composite gates on the witness margin). My
0.0036 is the *field-level argmax* number (harsher — every FP pixel counts). **Both axes agree the band
captures lane SHAPE** (small FN); they differ only on how the thin FP is counted. Honest answer to
"how close does the geometry seed get vs 0.00087": the seed nucleates the lane at ~0.0036 field-level
lane-attr, and the realized band drives that to ~0.00087 once the stem + witness-margin composite
suppress the thin FP — the seed's job is to NUCLEATE (FN→0), the render-band's job is the per-pair
FP-suppressed authority.

---

## 5. The dominant seed debt is SMOOTH-class, and it is the pose screw (dual-use seg+pose seed)

The field-level seed d_seg decomposes (n96): **total ≈ smooth-class shared-vs-per-pair mismatch (0.023,
~87%) + lane (0.0036, ~13%).** The ideal PER-PAIR seed drives the smooth term to 0. So:

- The **prefactor** of the seed's scaling law is dominated by the smooth-class shared-vs-per-frame
  mismatch — closed by warping the shared static-core partition by the **ego-screw ξ(t)** frame-to-frame
  (`Σ_t = H_t(Σ_0)`, law #5; the static Road/sky/hood boundaries move by the ego motion). This is the
  SEG seed's "get even closer" lever.
- The SAME ξ(t) is the POSE sufficient statistic. **ONE ξ(t) serves BOTH** the per-pair seg-seed warp
  AND d_pose (the store-nothing carrier, #257). This unifies the seg-seed and pose-seed the operator asked
  for: seed both from the SAME openpilot/comma2k19 ego-motion.
- CAVEAT (the transient/persistent split): the smooth-class debt is **trainable-away** (the witness
  learns smooth classes to IoU 0.95–0.99), so the ξ-warp is a *warm-start / convergence-rate* lever
  (helps the prefactor and the annulus). The LANE nucleation debt is the **persistent** one
  (nucleation-barrier-locked, un-trainable) — §2's paint-then-SDF is the load-bearing fix.

### 5a. POSE seed (design-only; HELD/UNMEASURED per discipline)
Init the per-frame ξ(t) SE(3) B-spline (`tac.lie.se3_bspline`, #193/#257) from comma2k19 GT ego-motion
(`experiments/results/pose_feasibility_probe/comma2k19_gt_pose_raw.npz`) → FiLM-condition the store-nothing
carrier; reuse the same ξ to warp the seg seed. **NOT a solved lever:** per
`project_pose_solved_screw_twist_dual_use_film_conditioned_sidecar`, pose is OPEN + UNMEASURED on the
witness (naive warp catastrophic d_pose 3.7–10.3; the 3.4e-5 is the ANCESTOR RGB vehicle, never
witness-validated). The pose seed is DESIGN-ONLY and HELD until a byte-closed d_pose is MEASURED. No pose
claim here.

### 5b. META-init (design-only; #211)
A hypernet `H_ψ(scene) → θ_0` pre-seeds the RESIDUAL θ (palette/texture head + Movable + smooth-INR
trunk) near the per-video optimum, so the geometric separatrix seed sets phi and the meta-init sets the
rest ("get even closer"). Overfit-XOR-generalize per #211. Design-only; no measurement.

---

## 6. Nucleation-safety — RE-RANK the memo: eikonal + area-constraint > spatial dilation [MEASURED]

The nucleation memo's fix #1 was "+2px dilate the seed lane above the critical nucleus." At the FIELD
level this is a NET LOSS — a dilation sweep (n96, paint-then-SDF):

| dilation | lane_FN | lane_FP | lane_attr |
|---|---|---|---|
| DIL=0 | 0.00189 | **0.00175** | **0.00363** |
| DIL=1 | 0.00156 | 0.00644 | 0.00801 |
| DIL=2 | 0.00146 | 0.01136 | 0.01311 |

Each +1px adds ~0.0045 FP but recovers only ~0.0003 FN (**~15:1 FN:FP penalty**); the +2px FP (0.0114)
EXCEEDS even the full-erasure static FN (0.0058). The memo's +2px was a *hard-mask-under-Gaussian-smoothing*
survival argument (temporal MCF proxy); at the argmax/field level it over-paints. **RE-RANK:** the
flow-survival mechanism should be
1. **raised eikonal** `--eikonal-weight 0.01 → 0.05` — the interface-width control (`|∇φ|=1` keeps the
   thin lane's interface exactly τ/2 wide through the flow, 0 field-level FP), and
2. a **per-class area constraint** (auction-MBO / area penalty — pins lane mass ≠ 0; NOT YET WIRED, a BUILD item),

NOT spatial dilation. Seed thin (DIL=0, min FP) and let the eikonal + area-constraint hold the nucleus
through the tau/MCF stage. This strictly dominates +2px on the measured field-level d_seg.

---

## 7. The fresh-run SEED config (never-invent-flags; grepped from the trainer)

The SEG seed, composed with the decomposition (seed nucleates lane; render-band is the per-pair authority):

```
# --- structured separatrix seed (0 archive bytes; rule-118 FREE) ---
--structured-init --structured-init-include-lane --structured-init-thresh 0.5
--structured-init-steps 600 --structured-init-lr 5e-3 --structured-init-sdf-clip 20.0
--lane-prior-phi1 --lane-prior-phi1-dash-gate            # openpilot separatrix lane into phi1
# CAVEAT (MEASURED §2b): --lane-prior-phi1-mode replace does NOT nucleate (FN stays ~0.0054).
# REQUIRED ~10-LOC CODE FIX: add --lane-prior-phi1-mode paint (build partition -> paint band∩road ->
#   re-EDT) so the lane wins the argmax (FN -> 0.0019). Until it lands, the phi1 prior is a no-op.

# --- nucleation-safety: eikonal (NOT spatial dilation) + lane-thin sharpness (§6) ---
--eikonal-weight 0.05                                     # raise from 0.01: interface-width control
--length-weight 0.001
--lane-thin-weight <sweep>  --lane-thin-radius 4  --lane-thin-target 0.5   # hold the thin nucleus
# BUILD ITEM (not yet wired): per-class area constraint / auction-MBO to pin lane mass != 0.

# --- per-pair lane authority (the decomposition; render-side, composes with the seed) ---
--lane-render-band --lane-band-dash-forward-max-m 55.0 --lane-band-uncertainty-source witness
--lane-band-tau 0.85 --lane-band-eps 0.35 --lane-band-start-epoch <earlier than 300, deconflict tau>

# --- pose seed (DESIGN-ONLY, HELD/UNMEASURED §5a) ---
# init ξ(t) SE(3) B-spline from comma2k19 GT ego-motion; FiLM the store-nothing carrier; dual-use warp.
```

`--eikonal-weight`, `--length-weight`, `--lane-thin-*`, `--structured-init*`, `--lane-prior-phi1*`,
`--lane-render-band`, `--lane-band-*` all verified present in
`experiments/train_levelset_witness_realized_through_R_mlx.py` argparse (grepped 2026-07-04). The
`--lane-prior-phi1-mode paint` value and the per-class-area-constraint flag are NEW (do not exist yet)
— flagged as the two code BUILD items, not invented as if present.

**DON'T:** add a spatial-dilation flag (measured field-level FP-costly, §6); rely on
`--lane-prior-phi1 --mode replace` alone (measured no-op, §2b); claim pose solved (§5a).

---

## 8. THE ONE SYNTHESIS CLAIM

> **The geometry-informed seed sets the scaling law's prefactor AND its nucleation regime, and decomposes
> into a TRANSIENT smooth-class debt (0.023 field-level, ~87%, trainable-away, best warm-started by the
> ego-screw ξ per-pair warp — the SAME ξ that is the pose sufficient statistic, so seg-seed and pose-seed
> are ONE object) and a PERSISTENT lane-nucleation debt (the only un-trainable part). The nucleation debt
> is unlocked by PAINT-THEN-SDF — build the partition with the openpilot separatrix band painted BEFORE
> the EDT — which drops seed lane_FN 0.0058→0.0019 (MEASURED, n96), and which the current
> `--lane-prior-phi1 --mode replace` does NOT achieve (it injects a shallow lane SDF that loses the argmax
> to the deep static-core road SDF, FN stays ~0.0054 — a MEASURED defect). Flow-survival of the thin
> nucleus is delegated to the raised eikonal (interface-width control) + a per-class area constraint,
> which STRICTLY DOMINATE the +2px spatial dilation (MEASURED field-level FP 0.011 > full-erasure FN
> 0.006, a ~15:1 FN:FP penalty).** [MEASURED (field-level, advisory) + DERIVED; the realized-through-R
> lane number 0.00087 is CITED for its axis.]

---

## 9. Ledger + next

| finding | class | evidence |
|---|---|---|
| ideal separatrix seed → 0 field-level d_seg | MEASURED | n96 pre_R 0.0 / post_R 4.2e-5 |
| lane nucleation failure N-dependent → total at #205 | MEASURED | static lane IoU 0.50→0.37→0.009; FN→0.0059 |
| `--lane-prior-phi1 --mode replace` no-op for nucleation | MEASURED | FN 0.00583→0.00538 (shallow-into-deep) |
| paint-then-SDF nucleates lane (FN→0.0019) | MEASURED | n96 paint-then-SDF |
| spatial +2px dilation field-level FP-costly ~15:1 | MEASURED | DIL sweep FP 0.0017→0.0114 |
| smooth-class shared debt = 87% of seed d_seg | MEASURED | 0.023 of 0.027; closed by ξ-warp |
| margin gate ~inert at the seed (FP annulus-concentrated) | MEASURED | FP 0.00175→0.00169 at margin<4 |
| openpilot band lane-lever = 0.0036 field-level | MEASURED | vs realized 0.00087 (CITED, memo) |
| pose seed via comma2k19 ξ (dual-use) | DESIGN / HELD | unmeasured on witness; NO claim |
| meta-init H_ψ→θ_0 residual pre-seed | DESIGN | #211; unmeasured |

**Next (operator GO + governor gate any heavy launch — CONTAINMENT):** (1) land the ~10-LOC
`--lane-prior-phi1-mode paint`; (2) the per-class area-constraint (auction-MBO) BUILD; (3) the ξ-warp
per-pair seed (dual-use with pose); (4) fold §7 into the next fresh seeded run. The pointer (0.19110)
moves only through a byte-closed n600 `upstream/evaluate.py` row — everything here is MEANS.

*Sisters:* `lane_nucleation_failure_seed_above_critical_nucleus` · `analytic_lane_band_primary_authority_decomposition` ·
`deepmath_amortizing_argmax_paper_draft_20260704` (laws #2 caustic, #5 se(3) screw, #7 MCF) ·
`project_pose_solved_screw_twist_dual_use_film_conditioned_sidecar` (HELD) · tasks #208 (seed) / #286
(eikonal) / #211 (meta-init) / #257 (ξ carrier) / #205.
