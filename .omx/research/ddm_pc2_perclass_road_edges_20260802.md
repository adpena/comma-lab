# ddm_pc2 — Road, the edges, and the floor that was the wrong floor

- **arm:** ddm_pc2 · **date:** 2026-08-02 · **axis:** `[macOS-CPU advisory]` NON-PROMOTABLE.
  `score_claim=false`, `promotion_eligible=false`. **Pointer UNMOVED.**
- **operator scope (verbatim, binding):** *"look closely at the decomposition. We know from our v
  seven, eight, nine, and ten work how each class is most optimally represented and carried and how
  their interactions and edges and boundaries work and are best represented. That is likely signal
  that is useful in the phase faithfulness problem and work.* ***there are no floors. It's all a
  matter of proper deep math and engineering.***"
- **method:** $0. No scorer pass fired (slot held by `ddm_pg1`). Every number is either read from an
  existing custody receipt or computed by reducing cached per-flip / per-cell arrays. Positive
  control on the reduction reproduces the source receipt to `absdiff = 0.0` (§1).
- **denominator for every ΔS:** `tac.canonical_equations.gap_decomposition_against_floor_20260802`
  — gap to PR130 = **0.7263025** (seg 0.4015190 = 55.3% · pose 0.2120155 · rate 0.1127679).
  1% of gap = 10,908 B.

---

## §0 Headline

**`ddm_cv1` §11 ranked Road "sitting EXACTLY on its floor, ratio 1.00 — attacking Road means
piercing a floor." That ranking is an artifact of the reference it chose.** Against the floor that
was actually measured *on the shipping path*, Road is at **3.03×**, and against an exact solve of
our own representation it is at **18.1×**. §11's "22% of the seg residual is ours to take"
(0.09366 S, 12.9% of gap) understates the demonstrated headroom by **3.2×**.

| reference for the FULL seg residual | S | our ratio | headroom | % of total gap |
|---|---:|---:|---:|---:|
| live renderer (tb1 ep399) | 0.38878 | 1.00× | — | — |
| smooth-label GT-flicker floor (`ddm_fl1`, **§11's reference**) | 0.53185 | **0.73×** | *we are already below it* | — |
| oracle-R achievable floor @ our render grid (`probe_PA` / #210) | 0.09100 | **4.27×** | **0.29778** | **41.0%** |
| exact per-cell solve, **same representation** (`ddm_sg1`) | 0.01520 | **25.58×** | 0.37358 | 51.4% |

The operator's "there are no floors" is not rhetoric here — it is the *measured* state. The number
§11 divided by is a **smooth-label reference from a different formulation**, and we are already
**27% below it**. Dividing by it makes every class look finished.

**Two caveats that bound the claim, stated before it is used.**
1. **The oracle-R floor is a LOWER bound, not a promise.** It is measured with the **real frame's
   full texture** as the signal (P-A's own caveat: *"a byte-limited carrier reproduces LESS
   texture"*). It proves **no floor exists at our current value** — a realization through the *same*
   R at 0.09100 S exists — but it does not assert that ~500 KB of tokens can reach it. Every
   "headroom" figure below is *demonstrated distance to a realized point*, never a forecast.
2. **The §11 correction is NOT an endpoint artifact.** §11 read ep641; this memo reads tb1 ep399,
   where Road happens to be better (0.12036 vs 0.18845 S). Checked at §11's *own* endpoint: **at
   ep641 Road is 1.00× the flicker floor and simultaneously 4.74× the oracle-R@384 floor.** The
   reference swap changes Road's reading from "finished" to "4.7× out" at **both** endpoints — and
   the effect is *larger* at §11's.

**And the residual is not five class problems. It is one graph, with one hub.** Road participates —
as the GT side or the realized side — in **87.8% of all 458,738 flips**. A single edge,
**Road↔Lane, carries 49.2%** of them and **22.1% of the entire remaining gap**.

---

## §1 Apparatus validity (checked before anything was read off it)

`ddm_ru1_20260729/atlas_flat.npz` holds 458,738 per-flip rows with `gt_class`, `realized_class`,
`dist_bin`, `gt_flicker`, `m_def`, `gt_margin`. Reducing it reproduces its own receipt exactly:

| control | computed | receipt | absdiff |
|---|---:|---:|---:|
| `on_gt_boundary` frac | 0.9386032986148957 | 0.9386032986148957 | **0.0** |
| `near_3px` frac | 0.0608212094921284 | 0.0608212094921284 | **0.0** |
| `interior` frac | 0.000575491892975947 | 0.000575491892975947 | **0.0** |
| `gt_flicker` frac | 0.4954200436850664 | 0.4954200436850664 | **0.0** |

**Same-endpoint join, no transfer.** `ru1` (458,738 flips) and `sg1` (458,621 flips, d_seg
0.0038878) are the **same tb1 ep399 endpoint** — 0.026% apart, within the interior-pair convention.
So the per-edge (ru1), per-cell (sg1) and exact-solve (sg1) columns join **directly**, with no
cross-endpoint number transfer. `ddm_fl1`'s staleness rule is therefore not engaged inside this memo.

**Endpoint honesty.** tb1 ep399 (0.38878 S) is *not* the live base (0.431 S) and not r1c ep641
(0.4264 S). All three are within 14%; the **structure** below is measured at tb1 ep399 and is
labeled as such. The live-base re-join is owed and belongs to `ddm_rd2`.

---

## §2 Two corrections to my own first draft (both caught by reading source)

Recording these because each would have shipped a wrong headline.

1. **A "render-grid win" of 0.156 S (21% of gap) — VOID.** The oracle report
   (`reports/levelset_oracle_R_floor_n600_20260701.json`) says `render_grid_default: [192,256]` and
   its verdict names "rendering at a higher grid" as "a large representation lever". Both are true
   of the **base** trainer (`train_witness_realized_through_R_mlx.py:3025`, `--render-h default=192`).
   The **live** entry point is `train_levelset_witness_realized_through_R_mlx.py:17953`,
   **`--render-h default=384 / --render-w 512`**, and `tools/levelset_byte_close_and_eval.py:611`
   defaults the same. **The grid win was already taken.** Every oracle comparison in this memo is
   therefore against the **@384 row (0.09100 S)**, never @192.
2. **"Road eats Lane by mean-curvature erasure" — WRONG MECHANISM.** The tr1 lever registry
   (`experiments/train_tr1_partition_renderer_mlx.py:135-139`) states it directly: *"NO scalar
   length/MCF term exists in the tr1 loss ⇒ the Lane-erasure mechanism is absent BY CONSTRUCTION."*
   Correct. The measured mechanism is different and is derived in §5.

A third source-check killed a proposal before it was written: **`#609-v2` KILLED the exact BEV /
ground-frame chart** (Road 39.02 / Lane 47.12 px p50 residual); the registry pins
`row_anisotropic_D_foveation` as **IMAGE-PLANE ONLY**, re-entry only via that memo's F1∧F2
falsifier. MEMORY agrees (`CRUX=realization in IMAGE chart (BEV 39-47px)`). **No ground-frame lane
carrier is proposed here.**

---

## §3 The per-EDGE decomposition (job 2) — the operator's "interactions and edges and boundaries"

**MEASURED** (ru1 atlas, n600, tb1 ep399). Oracle column derived from `probe_PA` RESULT 2's
destination matrix × its per-class shares @384. Edges below 1,000 flips omitted (0.31% total).

| edge | share of ALL flips | live S | oracle-R@384 S | ratio | headroom S | **% of gap** | on-GT-bnd | near-3px | flicker | m_def<0.25 | asym |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Road↔Lane** | **49.23%** | 0.19140 | 0.03099 | **6.18×** | **0.16041** | **22.1%** | 97.3% | 2.6% | **57.6%** | 30.6% | **3.60×** |
| Road↔Undriv | 16.26% | 0.06321 | 0.02054 | 3.08× | 0.04267 | 5.9% | 97.9% | 2.1% | 42.6% | **45.4%** | 1.31× |
| Undriv↔Movable | 11.85% | 0.04606 | 0.01136 | 4.06× | 0.03470 | 4.8% | 80.3% | **19.7%** | 40.4% | 32.8% | 2.28× |
| Road↔Movable | 11.47% | 0.04458 | 0.00805 | 5.54× | 0.03654 | 5.0% | 82.1% | **17.9%** | 37.5% | 35.2% | 1.47× |
| Road↔MyCar | 10.89% | 0.04234 | 0.01942 | 2.18× | 0.02292 | 3.2% | 99.2% | 0.8% | 45.8% | **45.7%** | 3.71× |

- **Road-incident edges = 87.85% of all flips**, 0.34153 S live, **0.26254 S headroom = 36.1% of
  the total remaining gap.**
- **Road NODE participation 87.8%**; Lane 49.5%, Undriv 28.1%, Movable 23.4%, MyCar 11.1%.
- **Road net area bias `+118,775 px`: the vehicle OVER-paints Road.** It gains 260,883 px and loses
  142,108. This is the **exact inverse** of the ep125 run-1 anomaly (`road_anomaly_probe`: Road
  UNDER-painted 0.61×, stolen by Lane 13.8× / Movable 4.6×). That defect is gone; the sign flipped.
- **Interior flips ≈ 0** (0.058% globally; three of five edges exactly 0.0%). At ep125 the
  road_anomaly probe measured **79% of flip mass INTERIOR**. The residual has fully migrated to
  codim-1. **The live vehicle and the irreducible oracle floor are now in the same regime.**

**Why the per-CLASS table hides this.** Charging by GT class splits ONE edge across two rows: §11's
"Road 44%" and "Lane 30%" of the seg residual are largely **the same 225,840 pixels** on the
Road↔Lane separatrix, counted from opposite sides. **The edge is the object; the class is not.**
This is exactly SPEC_v8 §1's binding refinement — *"the decomposition is EDGE-CENTRIC, not
class-naive… one field per adjacency-graph EDGE, never two region fields paying for the same curve
twice"* — and §8(2): *"a class-naive 5-field build is a spec violation, not a variant."*

**Structural agreement with the floor.** Our edge ordering is the oracle's edge ordering. P-A's
oracle destination profile for Road is `Lane 41% / Undriv 25% / MyCar 23% / Movable 10%`; ours is
`Lane 34.6% / MyCar 27.7% / Undriv 22.7% / Movable 15.0%`. **We are not making different mistakes
than the floor. We are making the same ones, 4.3× more.** That is what licenses reading the gap as
attackable rather than as a change of kind.

**The lever split falls out of the last three columns** (this is the actionable part):

| edge group | signature | what it implies |
|---|---|---|
| Road↔Undriv + Road↔MyCar (0.10555 S live) | **45.4% / 45.7% near-tie**, 97.9% / 99.2% exactly on boundary, low `m_def` median (0.283 / 0.280) | **tie calibration.** Nearly half these flips are runner-up by <0.25 logits on a boundary that is already in the right *place*. This is SPEC_v8 §1's ~0-byte `b_c`, edge-resolved. |
| Road↔Lane (0.19140 S live) | **lowest near-tie (30.6%)**, **highest flicker (57.6%)**, worst asymmetry (3.60× Lane→Road) | **not** a tie problem. Real positional/geometric deficit, per-pair. §5. |
| Road↔Movable + Undriv↔Movable (0.09064 S live) | **17.9% / 19.7% `near_3px`** vs 0.8–2.6% elsewhere | **displacement**, not jitter — the silhouette is in the wrong place by 1–3 px, a different failure from the other three edges. |

---

## §4 What v7–v10 says Road should be, vs what the live vehicle is (job 1)

**Live vehicle geometry — MEASURED from the shipped checkpoint's tensor shapes**
(`ddm_r1c_20260731/window_01/checkpoints/stage_seg_trunk_tau_final.npz`):

| tensor | shape | values | share |
|---|---|---:|---:|
| `tokens_base` | (24, 32, 4) | 3,072 | 0.16% |
| **`tokens_delta`** | **(600, 24, 32, 4)** | **1,843,200** | **98.63%** |
| conv0 + up0..up3 + head | (…,3,3,…) ×6 | 25,608 | 1.37% |

`24×32 token grid → 4 × 2× conv upsamples (= grid_downsample 16) → render 384×512 → R → camera`.

**Confirmed independently at the shipping receiver** (`src/tac/optimization/ddm_tr1_runtime.py`:
`SEG_H/SEG_W = 384/512` hard constants L85-88; `_conv_shapes` L274-288 derives
`n_upsample = log2(grid_downsample)` so the upsample factor *identically* cancels the downsample;
`state/selector.sec` on the live archive reads `grid_h: 24, grid_w: 32, code_width: 4,
output_height: 384, output_width: 512`). **Live archive = 360,309 B / 6 ZIP members** (v4d
grammar), of which **`state/tokens.dr7t` = 346,478 B = 96.2% of scored bytes** and
`state/renderer.sec` = 3,341 B. *(MEMORY's "504,736 B / 2 members" is the **eg1 rehearsal
packet**, not the live ship — corrected here.)* Renderer-section size depends only on
`renderer_width`/`code_width`/`n_upsample`, **never** on `grid_h/grid_w`: the entire
grid-proportional cost is tokens.

**Therefore: one shared 4-dim code per 16×16 render cell, one shared conv decoder, one RGB head,
for all five classes. There is no per-class field, no per-edge field, and no contour DOF anywhere
in the representation.** The live `tr1_config.json` (49 keys) confirms it by absence:
`lane_render_band`, `render_aa`, `per_class`, `carrier`, `hood`, `static_mask` — **all ABSENT**.

**What the v7–v10 body prescribes for Road** (`perclass_carriers_design_20260708.md` §carrier-table,
SPEC_v8 §2):

> **Road0** — *bulk-boundary field shared with Undriv2: ONE smooth curve network (road edge).
> **#308 theorem: regularized GRIDS ≥ INR on dense/smooth SDFs; INR wins on contours → grid-bulk +
> INR-annulus hybrid.*** 20–50 KB. And P-A's one-line implication: *"spend the bytes on
> **boundary/annulus precision, not interior texture** (interiors flip ~0)."*

**The delta, stated once.** #308 splits the representation by *what the signal is*: grid for the
dense smooth bulk, INR/contour for the codim-1 set. Our vehicle implements **only the grid half**.
And §3 measures that **our entire residual is the contour half** (93.9% on the GT boundary, 0.058%
interior), while P-A measures that the half we *did* build is already near-free at the floor
(within-class flip Road 0.17%, Undriv 0.03% — the **lowest** of all classes).

> **We are spending 98.6% of the archive on a dense per-pair field over a bulk that is already
> nearly free, and 0% on the contour that is 100% of the error.**

That is the answer to job 1, and it is an *allocation* statement, not a capacity statement.

**The allocation is measurably idle**, from `sg1/cell_flip_mass.npy` (768 cells):

| | |
|---|---|
| cells with **zero** flips | **486 / 768 (63.3%)** |
| 50% of flip mass | top **42** cells (5.5% of grid) |
| 90% of flip mass | top 123 cells (16.0%) |
| **99% of flip mass** | top **206** cells (**26.8%**) |

Every one of the 768 cells receives the same 4 numbers per pair. The hotspot cells (rows 11–12 of
24 → render rows 176–208) reproduce, independently, the tr1 registry's already-`GATE-PASSED-QUEUED`
`row_anisotropic_D_foveation` finding (72.1% of flip-prone mass in rows 160–240, 21% of rows). **My
cell reduction and that gate agree without sharing a method** — treat that as one fact, not two.

---

## §5 The mechanism, and the phase-faithfulness link (job 3)

Since the MCF/erasure mechanism is **absent by construction** (§2.2), the Lane→Road asymmetry needs
a different account. The data gives one, cleanly.

**MEASURED — within-class error rate vs class area** (`sg1`, n600, all five classes):

| class | GT area | **px per 16×16 token cell** | within-class err rate |
|---|---:|---:|---:|
| Undrivable | 49.52% | 126.8 | 0.101% |
| MyCar | 25.43% | 65.1 | 0.036% |
| Road | 23.23% | 59.5 | 0.518% |
| Movable | 1.24% | 3.2 | 4.736% |
| **Lane** | **0.585%** | **1.5** | **25.720%** |

`err_rate ≈ 2.79e-4 · area^(−1.258)`, **Pearson r(log area, log err_rate) = −0.934 (n=5)**, spanning
three orders of magnitude in each variable.

**The mechanism is sub-cell minority-class averaging.** A token cell is 256 scored pixels described
by 4 numbers. **Lane occupies 1.5 of those 256 pixels.** Its appearance must survive being encoded
jointly with ~254 road pixels and then reconstructed by a smooth 16× conv upsample. It does not: it
is averaged toward the majority, and loses the argmax. This is a *quantization/allocation* failure,
not a curvature flow — consistent with the registry's statement, and it predicts the asymmetry
direction and magnitude ordering that §3 measures.

**Two residuals of the law are themselves informative** (and both are predicted by the v7–v10 body):
- **Road sits 2.96× ABOVE the fit.** Area alone under-predicts Road because Road is the **hub** —
  it is the only class bordering all four others (87.8% node participation). The missing term is
  edge-sharing. *This is precisely why the decomposition must be edge-centric.*
- **MyCar sits 0.23× BELOW the fit.** MyCar is static (hood IoU 0.994) and is therefore already
  carried for free by `tokens_base`, the shared, temporally-constant half of the representation.
  *The one class that already has a structure-matched carrier is the one class beating the law.*
  That is a working, in-vehicle existence proof for the per-class-carrier thesis.

**Alternative reading of the law, and why the build conclusion survives it.** Area and *thinness*
are confounded across these five classes (Lane is both smallest and thinnest; Movable is small with
long silhouettes). The regression cannot separate `area` from `perimeter/area`. **It does not need
to:** `px-per-cell` is the quantity both stories reduce to, it is computed from geometry rather
than fitted, and **both** readings prescribe the same cure — a structure-matched non-grid carrier
for the thin classes. Which term dominates is a refinement, not a fork.

**The arithmetic that forces Lane off the grid.** To give Lane the pixels-per-cell Road enjoys
(59.5), the cell must shrink ~40× in area — ~40× the tokens on a member that is **96.2% of the
scored bytes**. The trainer menu offers only `grid_downsample ∈ {8, 16}`
(`train_tr1_partition_renderer_mlx.py:1692`); **ds=8 quadruples the token member (1,843,200 →
7,372,800 codes) and still leaves Lane at ~6 px per cell.** **No grid refinement available at any
affordable rate rescues a 0.585%-area class.** SPEC_v8 §2's *"Lane1 = analytic band ~1–2 KB"* is
not a preference; it is the only affordable option, and this is the quantitative reason.

### Phase faithfulness, made concrete

`ddm_fl1` defines the phase-faithfulness debt as what separates a *per-pair* renderer from a
*smooth-label* one, and ranks **Lane #1 (13.1× corner-C), Road #2 (7.1×)**. §3 shows why those two
rank together: **they are the same edge.** The link the operator asked for is three measured facts
that coincide on Road↔Lane:

1. it is the **largest** edge (49.2% of flips);
2. it is the **most flicker-typed** edge (**57.6%** — the highest of the five). 57.6% of 0.19140 S
   = **0.11025 S of this one edge is on pixels whose GT label changes between frames**;
3. it is the **least tie-like** (30.6% near-tie vs 45.5% on the two bias-fixable edges) — so it
   cannot be bought with a bias.

A flicker-typed flip is a demand for the separatrix to be in a *different place this pair*. Our
per-pair DOF is `tokens_delta`: 4 numbers per 16×16 cell. **Those numbers change what a cell looks
like; they cannot change where, inside the cell, an edge sits — not at the sub-pixel precision a
1.5-px-per-cell structure needs.** The carrier has per-pair **amplitude** DOF and no per-pair
**position** DOF. *Phase faithfulness is exactly positional per-pair DOF on the separatrix* — so
the phase-faithfulness debt and the missing contour carrier are one deficit, not two.

`ddm_ru1` reached the same words from the other end for its tail band — *"positional carrier
needed, not amplitude"* — for 16.9% of flips. §3 extends that reading to the **whole 49.2% edge**,
and §5 supplies the geometric reason.

**And this is where SPEC_v8 §3's channel routing becomes load-bearing rather than decorative.**
SegNet reads full-RGB last-frame (chroma fully argmax-visible); PoseNet reads YUV6 = 4 luma + 2
*subsampled* chroma. The correction Jacobian is near-triangular in (luma, chroma) ⇒ **chroma-first
seg repairs, luma reserved for pose/warp coherence.** The live head is `s_head (3,3,3,24) → 3
channels RGB` with **no channel routing at all**. So every seg repair the vehicle can currently
express also perturbs luma, which is the pose carrier — which is the structural reason seg
corrections keep getting pose-vetoed (`sh1` 74%, `fd1` §12; MEMORY "staging-law violations veto
seg"). Chroma-first routing is what makes a seg repair pose-cheap *by construction*.

---

## §6 Build-ready proposal (job 4) — the Road-hub separatrix carrier

Ordered cheapest-decisive first. Each rung names the surface that already exists; **none of this
proposes a new substrate**, and none of it is a ground-frame chart (§2, `#609-v2`).

**Rung 0 — $0, fires now: the edge-resolved boundary-cell code-width gate.**
The registry already rows `boundary_gated_token_code_width` as `never-fired` with a **"$0 gate
owed"**: *H(cell|neighbors) interior vs boundary cells on GT tokens at (D,c); adopt iff ≥15%
token-stream saving.* §3 supplies the input it was missing: the boundary cells are **not
homogeneous** — Road↔Undriv/Road↔MyCar cells are 45.5% near-tie (cheap), Road↔Lane cells are 30.6%
(expensive), Movable's cells are displacement-limited. Run the owed gate **stratified by edge**, not
just interior/boundary. Feeds the waterfill with a per-edge b/flip rather than one scalar.

**Rung 1 — ~0 bytes: per-EDGE tie calibration.**
`ddm_ru1` tier-1 already measured a byte-neutral token-lattice edit (single ±1 quantum, channel-sign
per atlas): **11.9% of flips, ΔS −0.046, ~0 B, +yield in 17/18 cells.** Its sign is chosen
*per-atlas*. §3 says the required RGB direction differs **per edge** (the Road→Lane margin direction
is not the Road→MyCar one), and that 45.5% of the mass on the two bias-fixable edges is within 0.25
logits. **Edge-resolve the existing sign rule.** This is SPEC_v8 §1's `b_c` in the only form our
RGB-head vehicle can express it.

**Rung 2 — the decisive build: an image-plane positional carrier on the Road↔Lane separatrix.**
`src/tac/boundary_math/analytic_lane_render_band.py` (**101.4 KB, BUILT**) is described in its own
docstring as *"The PRIMARY d_seg lane lever, in its NON-NAIVE form"*, composites over a render
**before R**, and carries three measured FP-killers: (a) `coverage_alpha_from_signed` analytic
sub-pixel coverage, (b) the range-dependent dash gate at the ~55 m SegNet Nyquist, (c) the
witness-uncertainty mask. The `aa_feasibility_reconciliation_20260702` verdict rates it **✓ OPTIMAL**
— *"HELPS… base grid only, O(1) g384, IN budget"* — while **disqualifying** brute supersample.

**It is ABSENT from the live `tr1_config.json`**, and aimed at the edge carrying 22.1% of the
remaining gap. But the `UNWIRED-BUT-BUILT` framing **overstates how cheap it is**, and the source
says so — recording that correction rather than shipping the easier claim:

- **TR1 has NO compose hook.** The band plugs into `render_through_R_mlx`'s
  `compose_fn(rgb_nhwc, code_idx)` — a hook of the **INR witness** lineage. TR1 is a different
  renderer that shares no code with it: `train_tr1_partition_renderer_mlx.py:658-666` imports only
  `_apply_R` and **accepts `render_h`/`render_w` for signature compatibility and never uses them.**
  The shipping receiver renders `render_frame1_float → bicubic_up_to_camera_float` with hard
  geometry guards (`ddm_tr1_runtime.py:1321, 1364`). **This is a trainer change + a receiver change
  + a new counted archive section for the band parameters — not a flag.**
- **There IS a precedent insertion point.** `render_frame1_camera_uint8(parsed, pair_index, *,
  window_solve: bool = False)` (`ddm_tr1_runtime.py:1378-1383`) already performs an **optional
  value-changing step between render and R** — exactly where the composite belongs. Build the band
  as a second such step. This is a real design constraint discovered, not a blocker.
- **Chart risk.** The dash gate reasons in metres (IPM). `#609-v2` killed the exact BEV chart at
  39–47 px p50. The gate plausibly uses range only as a *threshold*, not to place geometry — but
  that must be **verified in code**, or the gate disabled and the band run ungated.
- **A related receiver block worth noting:** the trainer already has a **factorized head**
  (`_head_out_ch`: `rgb=3 | class_field=1 | class_field_photo=2`, QA83) — the closest thing in-tree
  to SPEC_v8's channel routing — but **the receiver hardcodes 3 output channels**
  (`_conv_shapes` head row), so a `class_field` checkpoint cannot compile to a valid TR1 selector.
  The per-class/channel-routed head is trainer-reachable and **receiver-blocked**.

**Rung 3 — the never-fired edge term.** `perclass_pair_surface_tension_sigma_ccprime` (**#382**,
`never-fired`) is the per-class-**PAIR** σ_cc′ the Γ-limit demands. The registry is right that it is
inert today (no curvature/length term exists to carry it). §3 is the measurement that says it should
*become* live: once any boundary regularizer is added, a **scalar** σ would apply one surface
tension to five edges whose measured character differs by 1.5× in near-tie fraction and 25× in
`near_3px` fraction. Per-PAIR is not a refinement; it is the only correct form. Fire it **with**
rung 2, never before — and per the registry's own POOLS LAW, **race, never stack**: rungs 1–3 draw
from overlapping pools.

### Pre-registered falsifier (rung 2 — the one that matters)

Compose the analytic band onto the **frozen** tb1 endpoint's rendered RGB — no retraining — byte-close,
and measure through the exact R + frozen CPU-torch SegNet path at **n600**, reporting **per-edge**
flip deltas, not a composite.

- **PREDICTION:** Road↔Lane flips fall **≥30%** (≥0.057 S) with **collateral / on-edge recovery < 0.25**.
- **FALSIFIED IF** collateral/recovery **≥ 1.0** (net worse), **OR** Road↔Lane flips fall **<10%**.
- **Verdict scope on failure: FORMULATION** (this band, this gating, this endpoint) — not the
  positional-carrier family, and not SPEC_v8.

**Why that collateral bound is the whole test, and why it is not arbitrary.** `ddm_ba31` §B.4
measured QA92's lane-repair collateral at **18.58 : 1** — S created off-target per S recovered
on-target, *"the day's largest measured negative."* Any paint-on-top proposal must clear that or die.
The reason to expect it can is in ba31's own per-class table: **Road's collateral is invariant to
fill content (+0.125 oracle vs +0.130 flat, 4%)** ⇒ the damage is caused by **stroke GEOMETRY**, and
the stroke was a **`+1 px binary dilation`, fixed at one value and never swept** — ba31 flags that
continuum as binding. `coverage_alpha_from_signed` **is** that continuum swept to its analytic limit:
it replaces the binary dilation with continuous sub-pixel alpha. **The proposal is the swept version
of ba31's un-swept knob, aimed by §3's per-edge decomposition.** If the collateral does *not* collapse,
ba31's mechanism attribution is wrong, and that is itself a finding worth the run.

**BLOCKED-ON-SLOT** (`ddm_pg1` holds the single n600 scorer slot). The exact command shape:

```
.venv/bin/python tools/levelset_byte_close_and_eval.py \
    --ckpt /Volumes/VertigoDataTier/pact/ddm_r1c_20260731/window_01/checkpoints/stage_seg_trunk_tau_final.npz \
    --n-pairs 600 --device cpu --per-class-argmax --emit-confusion
```
(`--per-class-argmax --emit-confusion` are **NOT verified against that tool's argparse** — grep it
before use, per never-invent-flags. The per-edge confusion emitter may need to be added; the
reduction in `scratchpad/pc2_edge_decomp.py` is the reference implementation.)

---

## §7 Typed verdicts

| item | verdict |
|---|---|
| per-EDGE decomposition (5×5, undirected, + dist_bin/flicker/m_def per edge) | **MEASURED** (ru1 atlas n600; positive control absdiff 0.0) |
| Road node participation 87.8%; Road↔Lane 49.2% of flips | **MEASURED** |
| Road over-paints by +118,775 px (sign flipped vs ep125) | **MEASURED** |
| interior flips ≈ 0 (0.058%) ⇒ live vehicle now in the oracle's codim-1 regime | **MEASURED** |
| live seg residual 4.27× above oracle-R@384 / 25.58× above exact solve | **MEASURED** (sg1 + P-A + oracle report; sg1's containment caveat ⇒ the exact-solve gap is a **lower bound**) |
| **cv1 §11 "Road at floor, ratio 1.00" is reference-artifactual** | **DERIVED**, with receipt — §11's own scope note already says the flicker floor is formulation-scoped and pierced |
| per-edge oracle S column | **DERIVED** from P-A's destination matrix × per-class shares (rounded percentages ⇒ ±3% on each edge; the 6.18× and 2.18× extremes survive that band) |
| `err_rate ∝ area^(−1.258)`, r = −0.934 | **MEASURED** (n=5 classes — five points, one vehicle, one endpoint; **INSTANCE scope**, a strong regularity, not a registered law) |
| sub-cell minority-averaging as the mechanism | **DERIVED** (law + token geometry + registry's MCF-absence); the falsifier is rung 2 |
| `analytic_lane_render_band` BUILT and ABSENT from live config | **MEASURED** (file exists 101.4 KB; 49 config keys, no match) |
| TR1 exposes a pre-R `compose_fn` hook | **MEASURED FALSE** — `train_tr1_partition_renderer_mlx.py:658-666` takes `render_h/render_w` and never uses them; receiver pins geometry. Rung 2 is a trainer+receiver+section change, **not a flag** |
| live archive = 360,309 B / 6 members; tokens = 96.2% of scored bytes | **MEASURED** (live `v4d_composed_dc1_fold_archive.zip`; MEMORY's 504,736 B / 2 members is the eg1 rehearsal packet — corrected) |
| `renderer_head_mode` ∈ {rgb, class_field, class_field_photo} exists in trainer, receiver hardcodes 3 ch | **MEASURED** — the channel-routed head is receiver-BLOCKED |
| render grid @384 (grid-win claim VOID) | **MEASURED** (argparse defaults + byte-close default) |
| Lane-erasure by MCF | **FALSIFIED at this vehicle** — no length/MCF term exists (registry, source) |

**What this memo did NOT establish.** It did not move the pointer. It did not fire a scorer pass. It
does not re-derive the amortization gap as the crux — `ddm_ph3` §8/§8b and `ddm_ba31` §A.4 own that,
and QA74/QA24/QA75 are the rungs already routed. **Its contribution is the layer under them: the
gap is not five class problems but one graph with one hub, one edge holds 22.1% of everything left,
and the reason that edge fails is measurable in the token geometry.** Rungs 0–3 are aimed by that,
and rung 2 is a lever we already built and did not wire.

---

## NEXT-IF-RESUMED

1. **Rung 2's hook question is ANSWERED (negative):** TR1 has no compose hook; build the band as a
   second optional pre-R step beside `window_solve` (`ddm_tr1_runtime.py:1378-1383`), mirrored in
   `train_tr1_partition_renderer_mlx.py:658-666`, plus a counted band-parameter section. Scope the
   build against that, not against a config edit.
2. **Verify the lane band's chart dependence** — does the dash gate use IPM range as a threshold only, or to place geometry? If the latter, `#609-v2` applies and the gate must be disabled.
3. **Run rung 0** ($0, no slot): the owed `boundary_gated_token_code_width` entropy gate, **stratified by edge** using `pc2_edge_decomp.py`'s reduction.
4. **When `ddm_pg1` releases the slot:** re-run the §3 reduction on the **live** base (`ddm_rd2`'s re-join harness) so the edge table stops being a labeled-structure transfer from tb1 ep399 and becomes the live number.
5. **Do not** propose a ground-frame/BEV lane carrier (`#609-v2`). **Do not** propose brute supersample AA (`aa_feasibility_reconciliation`). **Do not** stack rungs 1–3 (POOLS LAW: race, waterfill winners).

## STORES CONSULTED

`ddm_cv1` §0/§1/§11 · `ddm_fl1` §2/§3/§4 · `ddm_xp1` + `xp1_verdict.json` · `ddm_ru1` +
`atlas_flat.npz` + `atlas_analysis_receipt.json` · `ddm_sg1` `sg1_typing_receipt.json` +
`cell_flip_mass.npy` + `qa24_grid_keep_mask_50.npy` · `probe_PA_paintfloor_perclass_20260708` ·
`road_anomaly_probe_20260708` · `SPEC_v8_perclass_decomposition_20260708` §1/§2/§3/§4/§8 ·
`perclass_carriers_design_20260708` (carrier table, #308) · `ddm_ph3` §8/§8b · `ddm_ba31` §A.4/§A.5/§B.4 ·
`ddm_vh1` · `aa_feasibility_reconciliation_20260702` Q1–Q5 · `reports/levelset_oracle_R_floor_n600_20260701.json` ·
`experiments/train_tr1_partition_renderer_mlx.py` (lever registry L120–200, TR1Config) ·
`experiments/train_levelset_witness_realized_through_R_mlx.py:17953` ·
`experiments/train_witness_realized_through_R_mlx.py:3025` · `tools/levelset_byte_close_and_eval.py:609-615` ·
`src/tac/boundary_math/analytic_lane_render_band.py` (docstring) · ep641 checkpoint tensor shapes ·
`tr1_config.json` · CLAUDE.md (class order, NO-FAKE, THE GOAL) · `docs/operating_manual_craft_handoff.md`.
