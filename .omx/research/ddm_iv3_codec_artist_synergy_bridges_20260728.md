# ddm_iv3 — THE CODEC-ARTIST SYNERGY BRIDGES (composition layer, 2026-07-28)

**Arm:** `ddm_iv3` (codec scientist-artist / composition layer), READ-ONLY. Base: `main@00c0c28fd7`.
**Directive:** operator 2026-07-28 — *"a codec scientist AND artist … looking to bridge and compose
synergy."* Not an enumerator (iv1/iv2 list what plugs in). This memo is the COMPOSITION layer: the
BRIDGES — pairs/triples of already-BUILT assets that FUSE into mechanisms greater than their sum, the
way a real codec's stages compound (each stage shaped BY the next). Aesthetic north star = the pantheon
task-lossy ego-scene codec (`pantheon_synergy_crux_synthesis_20260728.md`): ONE world-program
{image-chart atlas · ξ(t) dual-use · sparse VOP events} expanded by FREE inflate.py into the ONE
384×512 plane both frozen heads read.

**NO-FAKE / calibration.** Every ingredient path was verified to EXIST with the claimed mechanism (see
§0). Every synergy label is MEASURED (parts have receipts; the composition is explicitly noted
unmeasured) / DERIVED (two-line arithmetic shown) / CONJECTURE. NO score claim: the pointer (0.19108,
contest-CPU) moves ONLY through a byte-closed `upstream/evaluate.py` n600 row. `score_claim=false ·
promotion_eligible=false · rank_or_kill_eligible=false`. Contest S = 100·d_seg + √(10·d_pose) +
25·B/37,545,489; box target ≤~200 KB @ d_seg ≤0.00116, d_pose ≤0.00161; effective bar = min(0.15,
official 0.172). **Non-additive pools bind:** same-pool compositions COMPETE / co-measure, they NEVER
sum ΔS — each bridge names its pool.

---

## §0 INGREDIENT LEDGER (verified this arm — path · one-line mechanism · receipt)

| Asset | Path | Verified mechanism | Receipt axis |
|---|---|---|---|
| stratified depth warp | `boundary_math/stratified_depth_warp.py` | `stratified_warp_native(src,ξ,geom, offplane_mask, extra_flow)` — H(ξ) plane warp + per-region 2D parallax `extra_flow` on off-plane cells; bit-parity to A0 when flow=0 | code-exact |
| pose+depth receiver | `optimization/ddm_pc1_pose_stream.py` | `ground_and_movable_depth(movable_mask,…)` = ANALYTIC depth z(row) + Movable contact-depth stratum; `_warp_scorer_frame` = `W_{xi,depth}` writes frame_1 from frame_0 | code-exact; 10 importers |
| context partition codec | `boundary_math/context_partition_codec.py` | causal decoder-replicable range coder; floor Σ N_ctx·H(p_ctx); spatial-25 / temporal-125 templates; per-ctx PMF counts COUNTED | code-exact |
| g4 stationarity | `optimization/ddm_g4_spatial_stationarity.py` | 98.806% flip mass image-stationary; decoder-derivable field | MEASURED n600 (#623) |
| movable site coder | `boundary_math/movable_site_coder.py` | class-3 `(cx,cy,w,h)` boxes + Hungarian track + temporal-delta + REAL coder; render_sites lossy-tell | code-exact; #394 |
| dash-phase carrier | `boundary_math/dash_phase_carrier.py` | class-1 lane dashes (~20.6/frame world objects) δ(s) phase codec @ **2.267 bits/symbol** (jitter d=0 40.4% / ≤1px 72.3%) | MEASURED prior (#425) |
| hood static component | `boundary_math/hood_static_component.py` | class-4 #139 static-clamp bulk generator | MEASURED (34 B → 3.4e-5, G4) |
| road/undriv bulk | `boundary_math/road_horizon_component.py`, `road_undriv_bulk_field.py` | class-0/2 near-clean temporal-constant bulk fields | MEASURED (0.005–0.07) |
| ξ pose coder | `boundary_math/xi_pose_coder.py` | `coder="delta_ar"` per-channel temporal-delta ξ; H derived FREE at decode (rule-118) | MEASURED 474 B @6.8e-4 / 875 B @6.3e-5 |
| ξ temporal delta coder | `boundary_math/xi_temporal_delta_coder.py` | ξ-keyed temporal delta on residual/exception streams | MEASURED 12.6× over zlib (B5) |
| region merge SOLVE | `boundary_math/region_merge.py` | closed-form graph-cut at analytic **1.2727 B/flip** = (100/N)/(25/D) water level | code-exact closed-form |
| laguerre logit offset | `boundary_math/laguerre_logit_offset.py` | **0-byte** per-class logit geometry; Lane↔Road = 57% of flips | MEASURED (#218/#209) |
| blind coordinate | `through_r/blind_coordinate.py` | #401 resize-blind camera pixels = **230,904 px = 22.6969%**, decoder-derivable, NOTHING blind stored | code-exact |
| range(A) projector | `boundary_math/range_a_projection.py` | exact `P_range(A)`; ker(A) ≈ **52%** scorer-invisible render energy (same σ-algebra as the cell partition) | MEASURED (#519/#520) |
| seven-home allocator | `optimization/seven_home_stream_allocator.py` | KKT cross-home byte allocation over EV2's 7 counted homes; NON-additive (co-measures) | code-exact |
| costate organ | `optimization/costate_organ_v3.py`, `ddm_dr2b_tolerance_costate.py` | λ(state)=∂S/∂x live dual (#247) | in-tree |

---

## §1 THE TOP-7 COMPOSED BRIDGES (ranked by leverage = addresses-binding-residual × free-or-cheap × ingredients-real)

### BRIDGE 1 — THE DUAL-USE DEPTH STRATUM (the crown jewel; reopens oc1's PREDICT with FREE 3D)

**Ingredients:** `stratified_depth_warp.stratified_warp_native(… extra_flow)` (PREDICT parallax) ⊕
`ddm_pc1_pose_stream.ground_and_movable_depth(movable_mask)` (ANALYTIC depth + Movable contact stratum)
⊕ `_warp_scorer_frame` = `W_{xi,depth}` (pose leg).

**Mechanism.** oc1 MEASURED the 2D-homography PREDICT task-NEGATIVE — d_seg 0.018672 vs trivial copy
0.008642 (2.16× WORSE) — for ONE physical reason: *forward-driving motion is parallax-dominated (3D),
which a 2D homography cannot model* (oc1 §2). But `ground_and_movable_depth` already builds the exact
3D field that fixes this: an analytic per-row ground depth `z = h·f/(row−horizon)` PLUS a Movable
contact-depth stratum keyed off the `movable_mask`. That depth field is **decoder-derivable at ZERO
counted bytes** — its only inputs are the camera intrinsics (constants, free) and the `movable_mask`,
which Stream C already transmits. Convert z(p) → an SE(3)-reprojection source-flow and feed it as
`stratified_warp_native`'s `extra_flow`: the PREDICT stage now warps with the real 3D parallax. The
IDENTICAL depth field is what the pose leg's `W_{xi,depth}` uses to write frame_1 from frame_0. So ONE
free analytic map serves **three consumers**: (a) PREDICT parallax steering, (b) pose frame_1 synthesis,
(c) Movable-band VOP contact-depth placement. This is the "per-stratum plane+parallax REOPENED"
oc1 flagged (§3b/rung-4) — now with a BUILT, free depth field instead of an unbuilt MPI.

**Label:** DERIVED. Both ingredients are code-exact and MEASURED-to-exist; the composition is
UNMEASURED — the analytic depth field has never been fed as `extra_flow` into oc1's flip-support
measurement. The 2.16× homography-negative is MEASURED; the claim that adding 3D parallax reverses it is
the open datum.
**Cheapest confirming measurement:** re-run `experiments/ddm_oc1_flip_support_measure.py` with the
PREDICT warp = `stratified_warp_native(prev, ξ, geom, offplane_mask, extra_flow=SE3_flow(ground_and_movable_depth))`,
n600, d_seg vs the cached `lstars`. **CONFIRM** if d_seg < copy's 0.008642 (any margin proves 3D parallax
is a task-positive predictor); **KILL** if ≥ 0.008642 (analytic depth insufficient, needs learned MPI).
**Pool:** PREDICT ⊕ pose (dual-use — the depth is paid ONCE via the movable_mask, then free for both legs;
this is precisely why it is not double-counted). **Upside arithmetic:** the Movable band is **27.0% of
flip mass** (1,083,972 flips, pantheon §2) and the #1 structural residual (Movable conditional d_seg
0.988, iv1 §2). If free 3D parallax births even half the Movable contact edges, that is the single
largest attack on the 29× gap to the 0.00116 bar — for **zero counted bytes**.

### BRIDGE 2 — FREE SCENE-CONTEXT FOR THE PARTITION CODER (g4 + ξ-confidence → CPC context, 0 bytes)

**Ingredients:** `context_partition_codec.py` (floor Σ N_ctx·H(p_ctx), causal decoder-replicable) ⊕
`ddm_g4_spatial_stationarity.py` (98.8% image-stationary field, decoder-derivable) ⊕ DAG-801
ξ-warp-confidence.

**Mechanism.** CPC's achievable length is Σ_ctx N_ctx·H(p_ctx): sharper per-context PMFs ⇒ fewer bits.
Today the context is spatial-25 (left,up) or temporal-125 (+prev). Both are decoder-replicable, so
their PMF tables cost the same regardless of how rich they are — **richer context is FREE if it is
decoder-derivable**. g4's stationarity field is a function of the already-transmitted atlas + the free
ξ warp, so it is decoder-derivable; and the ξ-warp prediction (DAG-801) is a free confidence signal.
Extend the context tuple with a 4th and 5th axis — `(left, up, prev, g4_stationary_band?,
ξ_warp_agrees?)` — and the flip PMF collapses toward a spike everywhere the scene is stationary and the
warp agrees (98.8% of the field). The extra context axes cost ZERO counted bytes; only the (unchanged-
schema) PMF counts are transmitted. This is the codec-organism principle: the ENTROPY-CODE stage is
shaped by the free PREDICT/atlas structure upstream.

**Label:** DERIVED. CPC, g4, and the ξ-context are all MEASURED-to-exist; the H-reduction of the
extended context is UNMEASURED. (Honesty caveat: g4's KT rows already show entropy proxies ≠ real coder
— context-free 25.25 Mbit vs causal KT 12.34 Mbit vs boundary-distance proxy WORSE — so the extended
context must be measured through the REAL range coder, not an H proxy.)
**Cheapest confirming measurement:** build the 5-axis context-id array (CPC already has the
`_build_context_ids` seam), measure Σ N_ctx·H AND real range-coder bytes over the n600 partition vs the
temporal-125 baseline. **CONFIRM** if real bytes drop ≥15%.
**Pool:** CODING / support-geometry (competes with the LZMA/contour coder it replaces — never summed with
region_merge's bytes, which are a different granularity). **Upside:** the atlas/support stream is the
dominant 80–130 KB; a near-free 15–30% cut on it is a direct rate win on the largest counted section.

### BRIDGE 3 — ONE WATERFILL AT ONE DUAL λ* (region_merge × box allowance × seven_home × costate)

**Ingredients:** `region_merge.py` (analytic 1.2727 B/flip water level) ⊕ box error allowance (is1:
136,839 allowed vs 17,927 exact = 7.6× headroom) ⊕ `seven_home_stream_allocator.py` (KKT cross-home,
non-additive) ⊕ `costate_organ_v3.py` (#247 live λ).

**Mechanism.** These are FOUR views of ONE KKT dual variable λ* on the SAME rate pool. region_merge's
1.2727 B/flip is *exactly* λ* expressed as a per-flip price — (100/N)/(25/D), the analytic water level.
The box's 7.6× error headroom is the per-stratum TOLERANCE ladder at that same λ*. seven_home is the
cross-STREAM allocation at λ*. costate #247 is the per-DOF PULL = ∂S/∂x = λ*. Compose them and the whole
fix-vs-concede question becomes ONE closed-form knapsack-on-a-graph at a single swept λ* (honoring the
pantheon §7.1 "no static gates — λ SWEPT" mandate): region_merge decides per-region keep/merge, its kept
bytes flow into seven_home's cross-stream allocation, and the box tolerance loosens per-stratum where the
costate says the auth scorer is insensitive. No sweep, no guesswork — the fix-vs-concede SET in closed
form.

**Label:** DERIVED. Each piece is MEASURED / closed-form; the UNIFIED solve (region_merge feeding
seven_home at a swept λ*) is unbuilt.
**Cheapest confirming measurement:** run `region_merge` at 1.2727 on the n600 partition, feed its kept-
region byte stream into `seven_home_stream_allocator`, byte-close ONE row through
`tools/r6cal_byteclose_and_eval.py`. **CONFIRM** if the closed-form allocation beats the current heuristic
support threshold at equal d_seg.
**Pool:** the WHOLE describe/rate pool — every piece draws the SAME dual, so their ΔS are CO-MEASURED on
a merged base, NEVER summed (the cardinal non-additive-pools law). **Upside:** converts the dominant
open axis (which flips to fix vs concede) from a guessed byte budget into a solved λ*-optimal partition —
the mechanism the whole rate endgame rests on.

### BRIDGE 4 — THE BORN-COMPACT PER-CLASS PREDICTION GRAMMAR (eliminate the generic residual coder)

**Ingredients:** `movable_site_coder.py` (class-3 boxes+track) ⊕ `dash_phase_carrier.py` (class-1 dashes
@2.267 bits/symbol) ⊕ `hood_static_component.py` (class-4 #139 clamp, 34 B) ⊕ `road_horizon_component.py`
+ `road_undriv_bulk_field.py` (class-0/2 bulk) ⊕ `stratified_depth_warp` per-stratum plane (class-0/1
ground).

**Mechanism.** Give EACH class the native carrier matched to its geometry, so the generic per-pixel
residual coder is needed only for the transient tail. Movable = compact movers → `(cx,cy,w,h)` boxes +
Hungarian track (births the islands a smooth receiver cannot — attacks the 0.988 residual). Lane = dashes
→ curve-domain δ(s) phase @2.267 bits/symbol on ~20.6 world objects/frame (attacks the 0.437 Lane debt).
Hood = rigid → #139 static clamp (34 B → 3.4e-5). Road/Undrivable = temporal-constant bulk (near-clean
0.005–0.07). Ground = plane+depth (Bridge 1). This IS the pantheon's F1 per-stratum split (F4 VOPs)
realized entirely with BUILT modules — and it is the reason F1's split "is not cosmetic": cb1 measured
that the MyCar carrier and the Lane carrier have OPPOSITE d_pose signs, so per-class factorization is
required, not optional.

**Label:** MEASURED parts (each carrier carries its own receipt) + CONJECTURE composition — the claim
that all five carriers together cover the flip support well enough to DROP the generic residual coder over
most sites is unmeasured.
**Cheapest confirming measurement:** byte-account all five carrier streams + measure the UNCOVERED flip
residual on n600. **CONFIRM** if uncovered support < ~the 1.19% transient floor (pantheon §2); **KILL** if
a generic dense residual is still needed over >10% of sites (the born-compact grammar under-covers).
**Pool:** support-geometry CODING — but the five carriers PARTITION the support spatially (disjoint class
masks), so they are distinct SUB-pools that add over space (unlike same-pool rate competitors). Each
carrier's tolerance is still waterfilled by Bridge 3. **Upside:** directly attacks BOTH structural
residuals (Movable-island birth 0.988 + Lane 0.437) that dominate the 29× gap — the two holes a post-solve
value coder provably cannot reach (iv1 §7).

### BRIDGE 5 — THE NESTED FREE FIBER (blind 22.7% ⊂ ker(A) 52%) AS THE DESIGN MULTIPLIER

**Ingredients:** `blind_coordinate.py` (#401, 230,904 px = 22.6969%, exactly free) ⊕
`range_a_projection.py` (#580, P_range(A), ker(A) ≈ 52% scorer-invisible) ⊕ F5 gauge/null split.

**Mechanism — and the honest correction.** These are NOT a naive union (the seed conjecture asked "do
they union into a bigger mask?"). They are ATOMS OF THE SAME σ-algebra: the #401 blind mask is the
exact-camera-pixel readout of the SAME resize operator A whose null space is the 52% (range_a's own
docstring: "both are projections onto atoms of the same σ-algebra"). The blind 22.7% is the
exactly-readable, closed-form CORE of the 52% scorer-invisible render energy — it is NESTED, not disjoint,
so 22.7% + 52% is a DOUBLE-COUNT fake. The real bridge is a DESIGN MULTIPLIER on every other factor:
route maximal description DOF into ker(A) (≈52% of representational energy costs 0 S by construction,
F5), and use the exact blind mask (22.7%, derivable by one impulse probe = free receiver) as the
GUARANTEED-free carrier region for any side-info Bridges 1–4 need the receiver to have. This is a bridge
that CLARIFIES — it prevents the additive fake AND hands every other stream a free fiber.

**Label:** MEASURED (both fractions exact/measured); the nesting is a code-verified structural fact, not
a conjecture. The DESIGN-MULTIPLIER effect on any specific factor is UNMEASURED per factor.
**Cheapest confirming measurement:** apply `P_range(A)` to a blind-only perturbation — it must be
EXACTLY zero (proves blind ⊂ ker(A), killing the double-count). Then, per factor, measure the counted-byte
reduction when its DOF is projected onto ker(A) first.
**Pool:** gauge/null (F5) — a MULTIPLIER on all other pools, not a competitor. **Upside:** ~half of every
other factor's representational DOF can live free; it is the reason the atlas/pose/VOP budgets below have
headroom under the 154,522 B sub-0.15 line.

### BRIDGE 6 — ξ AS THE UNIVERSAL TEMPORAL KEY (one payload, four consumers)

**Ingredients:** `xi_pose_coder.py` (#257, 474–875 B ξ payload, H free at decode) ⊕
`xi_temporal_delta_coder.py` (#574, ξ-keyed residual delta, 12.6× over zlib) ⊕ the g4/CPC ξ-context
(Bridge 2) ⊕ `stratified_depth_warp` advection.

**Mechanism.** The SAME ξ(t) knots, transmitted ONCE as the pose payload (474–875 B via xi_pose_coder),
key FOUR downstream consumers for free: (1) the pose leg itself (its native job); (2) the residual/
exception temporal-delta coder (xi_temporal_delta_coder, 12.6× over per-frame zlib); (3) the CPC coding
context (Bridge 2's ξ-warp-confidence axis); (4) the advection warp (Bridge 1's PREDICT). This is the
pantheon's F2 dual-use made concrete: ξ is one object; four built modules each consume it; the coding
gain of a SHARED key (transmit once, exploit four times) is the synergy. rule-118 makes the H-derivation
free, so only the ξ trajectory is counted — once.

**Label:** MEASURED parts (each consumer's receipt exists) + DERIVED composition (ξ is provably one
object; the shared-key combined bytes are unmeasured).
**Cheapest confirming measurement:** byte-close the xi_pose_coder ξ payload, then key
xi_temporal_delta_coder's residual streams off the SAME decoded ξ; measure combined bytes vs independent
per-stream coding. **CONFIRM** if the shared key beats independent by the published 12.6× on the residual
stream while holding the pose payload flat.
**Pool:** ξ is paid into the POSE pool once; consumers (2)(3)(4) draw it FREE (the dual-use is exactly why
it is not re-counted). **Upside:** pose 7.2 KB → 875 B (−0.004-class S) AND the residual/exception stream
12.6× smaller on top of the shared key — a compounding win the whole codec-organism is wired around.

### BRIDGE 7 — 0-BYTE HEAD PRE-CONDITIONER × THE SOLVE (laguerre shrinks flips BEFORE region_merge prices them)

**Ingredients:** `laguerre_logit_offset.py` (#218, 0-byte per-class logit geometry) ⊕ `region_merge.py`
(1.2727 B/flip SOLVE, Bridge 3).

**Mechanism.** laguerre_logit_offset fixes class-imbalance under-prediction with per-class logit geometry
at ZERO archive bytes — Lane↔Road is 57% of all flips (#209), and the head systematically drops the
minority Lane/Movable classes. Applying it BEFORE region_merge means fewer flips ENTER the knapsack, so
region_merge's kept-region bytes (which it pays at 1.2727 B/flip) are strictly smaller — the SOLVE is
pre-conditioned for free. This is the purest codec-organism move: a 0-byte upstream stage reshapes the
priced downstream stage. Stage shaped by the next stage, literally.

**Label:** DERIVED. Both MEASURED; the compounding (fewer flips → strictly cheaper merge) is arithmetic
but the joint n600 magnitude is unmeasured.
**Cheapest confirming measurement:** apply the 3 laguerre head levers → recount flip support on n600 →
run region_merge at 1.2727 → compare kept bytes vs the no-offset baseline. **CONFIRM** if flip support
AND kept bytes both drop.
**Pool:** laguerre is a 0-byte HEAD/train pool (does not compete on rate at all); region_merge is the
describe pool. They are orthogonal, so this is a clean free-precondition, not a rate competitor.
**Upside:** the cheapest possible d_seg pre-condition (0 bytes) that ALSO shrinks the priced residual —
it costs nothing and makes every downstream stream smaller.

---

## §2 THE CODEC AS ONE ORGANISM — the single most beautiful coherent composition

The seven bridges are not seven ideas; they are ONE codec whose stages compound. Stated as the
stage-by-stage program of the archive it would produce — each stream shaped by the one after it, with a
projected byte budget (every number labeled):

**The archive = ONE world-program W, decoded by FREE inflate.py into the ONE 384×512 scorer-input plane.**

- **Stream A — ξ(t) trajectory** (`xi_pose_coder`). The universal temporal key. ~**0.5–2 KB**
  [MEASURED anchor: 875 B @ solved-grade 6.3e-5]. Paid ONCE; then keys the pose leg, the residual coder
  (Bridge 6), the CPC context (Bridge 2), and the advection warp (Bridge 1). *Shaped by:* nothing —
  it is the root the rest hangs from.

- **Stream D — depth stratum** (`ground_and_movable_depth`). ~**0 KB** [DERIVED: analytic from Stream C's
  movable_mask + camera constants, decoder-derivable]. Supplies the 3D parallax the 2D warp lacked
  (Bridge 1). *Shaped by:* Stream C (reads its movable_mask); *shapes:* the PREDICT warp AND the pose
  frame_1 synthesis.

- **Stream B — image-chart scene atlas** (region_merge-solved partition + `laguerre_logit_offset`
  0-byte precondition + `context_partition_codec` with g4+ξ free context). ~**80–120 KB**
  [DERIVED target; brackets: generic-AR partition ~114 KB MEASURED external, MDL(MS) ~236 KB upper
  MEASURED, ws1 138 KB @2.85M-err MEASURED — Bridges 2/3/7 target the low end]. The dominant stream.
  *Shaped by:* Stream A (ξ context, Bridge 2), the 0-byte head (Bridge 7), the λ* SOLVE (Bridge 3);
  ~52% of its DOF routed FREE into ker(A) (Bridge 5).

- **Stream C — per-class VOP carriers** (`movable_site_coder` boxes + `dash_phase_carrier` dashes +
  `hood_static_component` clamp). ~**10–25 KB** [MEASURED parts: hood 34 B; dash ~2.267 bits/symbol ×
  ~54 sites/frame; movable box+delta stream real-coded — CONJECTURE total]. Births the Movable islands
  (27.0% flip mass) and Lane dashes (0.437 debt) the smooth receiver cannot. *Shaped by:* Stream D (Movable
  contact-depth placement), Stream A (ξ prediction).

- **Stream E — residual syndrome** (`xi_temporal_delta_coder`, keyed off Stream A ξ, coded over the
  ker(A)-free fiber via Bridge 5). **OPEN** [sized by H(flip | free context) — the tier-moving scalar,
  UNMEASURED; only the transient 1.19% + Bridges 1–4 misses]. *Shaped by:* Stream A (temporal key), all
  upstream streams (only their misses remain).

- **Whole-object lossless transforms** (cc3 + wf7, compound on the container). **−3,422 B** [MEASURED,
  cc3 merged 06845c4582] **− 1,776 B** [macOS-CPU advisory, wf7]. *Shape:* nothing — they polish the
  finished container losslessly.

**The waterfill (Bridge 3) allocates bytes across B/C/E at ONE swept λ*** (no static budget — pantheon
§7.1), with ~52% of each stream's DOF free in ker(A) (Bridge 5). Projected counted total: **~90–150 KB**
[DERIVED, brackets summed at the low-to-mid of each stream: A 1 + B 90 + C 15 + E open − cc3/wf7 5] —
which lands UNDER the 154,522 B sub-0.15 line IF the atlas hits its low bracket and the pose/VOP streams
realize. Every stream is a BUILT module; the organism is the WIRING that makes each stage shape the next.

**The beauty, stated plainly:** the same ξ(t) that IS the pose answer is the temporal key that codes the
residual and the confidence that codes the atlas and the warp that predicts the scene. The same movable
mask that births the car islands IS the depth that fixes the parallax that writes frame_1. The same 52%
null fiber that hides the pose carrier hides half of every other stream. The same 1.27 B/flip water level
that merges the regions allocates the streams. Nothing is used once. That is a codec, not a pile of
levers — and every ingredient already exists in the tree.

**Honest boundary:** the pointer is UNMOVED at 0.19108. Every bridge above is DERIVED or CONJECTURE at
the composition level — the parts are measured, the fusions are not. The single confirming build is the
pantheon §5 km1 joint fit (Gauss-Newton/CG over {atlas, ξ, VOP} with the real coder in-loss), and this
memo's contribution is to show it can be assembled from BUILT modules wired as one organism, with Bridge 1
(free 3D depth) as the highest-leverage, cheapest-to-confirm first move.

## STORES CONSULTED

`CLAUDE.md` (NO-FAKE; §7.1 dynamical gates; inflate-is-free/rule-118; non-additive pools; pointer-only) ·
`MEMORY.md` current-state · `.omx/research/pantheon_synergy_crux_synthesis_20260728.md` (F1–F6, §4/§5/§7)
· `ddm_oc1_xi_temporal_predict_measured_20260727.md` (2D-homography task-NEGATIVE 2.16×; parallax is 3D;
per-stratum plane+parallax REOPENED) · `ddm_iv1_plugin_inventory_sweep_20260728.md` (TOP-10 mechanical) ·
verified module sources (§0 ledger): `stratified_depth_warp.py` · `ddm_pc1_pose_stream.py`
(`ground_and_movable_depth` + `W_{xi,depth}`) · `context_partition_codec.py` · `ddm_g4_spatial_stationarity.py`
· `movable_site_coder.py` · `dash_phase_carrier.py` · `hood_static_component.py` · `xi_pose_coder.py` ·
`xi_temporal_delta_coder.py` · `region_merge.py` · `laguerre_logit_offset.py` · `through_r/blind_coordinate.py`
· `range_a_projection.py` · `seven_home_stream_allocator.py` · `costate_organ_v3.py`.
```
