# T5 CRUCIBLE-3 — DELTA GROUNDING PACKET (2026-07-09) — the v8 optimal-final-form evidence pack

**Everything decision-relevant for v8's optimal final form, every number cited to its measured artifact.**
Six P1 seats work from THIS pack — do NOT re-mine the stores; if a claim you need is missing, it is a
grounding gap, flag it. Every number labelled MEASURED / DERIVED / CONJECTURED / ESTIMATED / ASSUMED with
its SOURCE PATH. A number without a sourceable path is marked **UNSOURCED-VERIFY-IN-P4**, never guessed.
Pointer **0.19110 UNMOVED** — everything here is [macOS advisory · research-signal · NON-PROMOTABLE],
MEANS. Only a byte-closed `upstream/evaluate.py` n600 row moves it. #205 STOPPED (box free); no live run
to protect, but P-C/P-B are heavy governed n600-through-R forwards (operator-GO, memory-gated).

STORES CONSULTED: SPEC_v8_perclass_decomposition_20260708.md · v8_increment1_design_draft_20260709.md ·
v8_roadlane_geometric_rate_20260709.md (441377ddd) · v8_roadlane_ego_compensated_rate_20260709.md ·
v8_movable_residual_rollup_20260709.md · registered equations {v8_geometric_rate_decomposition_v1,
lane_groundframe_xi_transport_no_collapse_v1, laguerre_ot_head_offset, chan_vese_area_constraint_birth_
balance_v1, dash_erasure_homogenization_v1} · crucible-2 DELTA_GROUNDING (rows N-1/N-2/P-1/P-7/V8-1/V8-2)
· DAG FEED-v8-* · SPEC_v75 §8B ALREADY-SETTLED · `src/tac/boundary_math/*` (on-disk, verified) · CLAUDE.md
§SegNet/PoseNet architectures. Label cache: `experiments/results/mlx_fleet_gt_cache/gt_n600.npz` (4.84 GB,
600×384×512 int64 argmax, comma10k order Road0/Lane1/Undriv2/Movable3/MyCar4). ALL v8 rate numbers are
$0 read-only on this cache through a REAL coder with a bit-exact roundtrip — no b/px proxy, no projection.

---

## A. THE ONE-LINE STATE CHANGE SINCE SPEC_v8

SPEC_v8 (2026-07-08) set the ARCHITECTURE with the rate win **PROJECTED** ("−50..75% ≈ 0.049 S
rate-headroom, CONDITIONAL on increment-1's byte measurement"; SPEC_v8 §2). **That measurement now
EXISTS, end-to-end, for all 5 whole-scene edges** (FEED-v8-rollup, real coder, bit-exact). The result
splits the v8 rate story cleanly: **dominant-structure geometric = 0.061 S (1.9× BELOW the 0.118 frontier
rate term — thesis CONFIRMED); lossless-complete = 0.140 S (1.2× ABOVE — ties/slightly over frontier); the
entire 0.079 S gap between them is the residual sidecar coder.** The v8 junction is no longer "does
geometry beat the bitmap" (YES, measured 5.5×) — it is **"can the residual coder + de-sharing close the
0.079 to land the complete number below the frontier, and does d_seg (the shared true blocker) hold."**

---

## B. THE RATE LEDGER — MEASURED, COMPLETE (the core evidence)

**Whole-scene de-shared geometric rate, each inter-class edge counted once, REAL coder, bit-exact
roundtrip through the UNCHANGED inflate decode** (source: `v8_movable_residual_rollup_20260709.md` §C +
`v8_increment1_design_draft_20260709.md` §3 + eq `v8_geometric_rate_decomposition_v1` REGISTERED
`src/tac/canonical_equations/v8_geometric_rate_decomposition_20260709.py`):

| edge | bitmap S | geometric DOMINANT-only S | geometric COMPLETE (+residual) S | × complete | generator (module) |
|---|---|---|---|---|---|
| Road/Lane | 0.204 | **0.0275** (72.5% cover) | **0.0695** (+resid 0.0420, 26.6% unc) | 2.9× | centerline poly + ξ track, Wave-F #234 (`analytic_lane_render_band.py`) |
| Road/Undriv (horizon) | 0.047 | **0.0032** (poly+ξ, 14.6×) | **0.0221** (+resid 0.0189, 25.7% unc) | 2.1× | 1 Laguerre site + deg-3 boundary poly + ξ (`road_horizon_component.py`) |
| Movable (Road/Mov+Undriv/Mov) | **0.0532** (MEASURED region bitmap) | **0.00344** (ONE carrier, 70% cover) | **0.0209** (+resid 0.0174, 30% unc) | 2.5× | sparse object sites, bbox 3.0/frame, Hungarian ξ-track |
| Road/MyCar (hood) | 0.028 | **0.0202** (static model, complete) | 0.0202 | 1.4× | static-model frame0 + rigid ξ-shift (`hood_static_component.py`) |
| Lane/* (3 rows) | 0.007 | 0.007 | 0.007 | 1.0× | already tiny |
| **WHOLE-SCENE TOTAL** | **0.339** | **0.061 (MEASURED)** | **0.140 (MEASURED)** | — | vs **0.118** pointer rate term |

**Consequence (MEASURED, no longer projected):**
- **Geometric DOMINANT-only = 0.061 S** — 5.5× < bitmap (0.339), **1.9× BELOW the 0.118 frontier**. The
  v8 rate thesis is CONFIRMED on dominant structure: geometry ≪ bitmap ≪ current frontier.
- **Geometric COMPLETE (lossless, generic residual) = 0.140 S** — 2.4× < bitmap but **~1.2× ABOVE the
  0.118 frontier**. **With today's generic sparse-coord residual coder, lossless v8 does NOT by itself
  beat the frontier rate.** The gap is ENTIRELY the residual sidecar (0.079 of the 0.140).
- **The true operating point is a RANGE 0.061 → 0.140**, depending on the residual coder + de-sharing.

**Contrast with SPEC_v8's projection:** SPEC §2 conjectured 0.049 S rate-headroom "CONDITIONAL." The
MEASURED dominant number (0.061 vs the 0.118 frontier = 0.057 S headroom on rate) VINDICATES the thesis on
dominant structure but the COMPLETE number reveals the residual coder as the binding term — a refinement
SPEC_v8 did not have. (The increment-1 draft's earlier "~0.02–0.05 projected" for the Road/Undriv edge was
also superseded: the real number is edge-scope + coder-dependent; see §D.)

---

## C. THE PARSIMONY / GENERATOR PRINCIPLE (the v8 representation law — MEASURED)

**The v8 representation law (operator "voronoi or others", MEASURED; source: FEED-v8-voronoi +
v8_increment1 §3 + #284):** the SegNet argmax `P(x)=argmax_c(φ_c(x)+b_c)` IS an additively-weighted
Voronoi = **Laguerre / power diagram** (#284: Laguerre = tropical = curvelet = se(3) — one structure).
Every carrier stores its class's **parsimonious GENERATOR**; the partition falls out of the tropical
argmax composition. **The win is parameter PARSIMONY, not the choice of dual** — curvelet-boundary ≡
medial-axis-centerline ≡ power-diagram-sites are one object.

**MEASURED negatives that keep this honest (do NOT re-open blind):**
- **Dense medial ≈ bitmap: NO WIN.** The dense medial axis (444 pts) ≈ the boundary bitmap (**1.09×**;
  MEASURED, FEED-v8-voronoi). verdict_scope: **FORMULATION** (the *dense* generator; the lever is the
  *few-coefficient* fit). Source: v8_increment1 §3.
- The horizon deg-3 poly (**4 coeffs → 14.6×**; MEASURED, FEED-v8-realmachinery) is the positive proof:
  the horizon IS a low-order curve (a single-scale curvelet atom = the codim-1 sparse basis the theory
  predicts), and 599/600 frames are a near-free ego-warp of frame-0's horizon (cubic/quadratic FROZEN
  |Δ|≈1e-7/6e-5; only the intercept moves ~1.2 px/frame = ego pitch — the vertical ξ already stored for
  pose). Real coder: zlib on delta-coded fp16 coeffs = **4.7 KB@n600 = 0.0032 S**.

**Design consequence:** per-carrier build = the PARAMETRIC GENERATOR (few coeffs), NEVER the boundary
bitmap and NEVER the dense medial axis. The rate win lives in the coefficient count, not the dual choice.

---

## D. PER-CARRIER STATE (measured/derived bytes + module + status — the composition menu)

Source: `v8_increment1_design_draft_20260709.md` §1 (carrier→substrate map) + the rate ledger §B. All
modules verified on-disk 2026-07-09 (`src/tac/boundary_math/`).

| class | carrier | module (STATUS) | bytes / S | basis |
|---|---|---|---|---|
| **MyCar (4)** | static mask clamp | `hood_static_component.py` — **REUSE as-is** | ~0.1–0.5 KB / 0.0202 complete | IoU 0.994 MEASURED #139 |
| **Lane (1)** | analytic ground-frame band | `analytic_lane_render_band.py` + `lane_sdf_component.py` — **REUSE** | 0.0275 dominant / 0.0695 complete | d_seg 0.00087 MEASURED (openpilot band); 72.5% cover, 27.5% resid owed |
| **Movable (3)** | sparse object sites, both edges ONE carrier | Hungarian ξ-track bbox coder (v7.5 homotopy + area-Lagrange reused) — **REUSE v7.5 machinery** | 5.04 KB@n600 = 0.00344 dominant / 0.0209 complete | bbox 3.0/frame MEASURED, 70% cover, bit-exact; region bitmap 0.0532 |
| **Road (0) + Undriv (2)** | ONE edge-centric bulk-boundary field | **BUILT scaffold** `road_undriv_bulk_field.py` (27.1K on-disk) — composes `road_horizon_component` + `lever_b_levelset_generator` (SDF) + `laguerre_logit_offset` (b_c) | horizon dominant **0.0032** (poly+ξ) / 0.0221 complete | edge = 426 px/frame MEASURED (19% of Road perimeter); coder-floor #307-class |
| **tie bias b_c** | Road↔Lane calibration offset | `laguerre_logit_offset.py` — **REUSE** | ~0 bytes | 41% of Road's oracle flips MEASURED (P-A) |

**Net stack ≈ 27–57 KB vs 114 KB incumbent ⇒ −50..75% ≈ 0.049 S — CONJECTURED, DOUBLY CONDITIONAL**
(on the byte number AND on d_seg holding at equal quality; NOT derived — it presupposes the increment-1
result; source: increment-1 §1 review-F). The *only* new module is the Road+Undriv field; four of five
carriers already exist.

**THE SCAFFOLD BYTE-COST BUG (standing catch, MEASURED — operator "707B is big enough to point to
naive"):** `road_undriv_bulk_field.bulk_boundary_byte_cost` = 707 B/frame was naive on TWO axes (source:
increment-1 §1, FEED-v8-bytecost-sharpened): (a) **coder** — brotli-on-bitmap ≈ 1.9× the chain-code floor;
(b) **scope (the bigger one)** — it measured the FULL Road perimeter (2228 px), but the edge-centric field
pays for the **Road↔Undrivable horizon ONLY = 426 px (19%)**; the other 81% (Road↔Lane 47% / Road↔MyCar
23% / Road↔Movable 5%) belongs to those classes' own carriers, and paying for them here IS the risk-1
double-count. **Correct scope + real machinery (horizon poly + ξ): 0.0032 S** (8× below the generic
chain-coder's 0.026, 88× below the naive 0.282). The scaffold's `bulk_boundary_byte_cost` needs a
`mode="horizon_poly_xi"` path measuring the Road↔Undriv shared edge, not the full Road mask. **This is a
build-config bug increment-1 must fix — a standing check for the seal.**

**LIFT-VERIFY (MEASURED, n600 gt argmax, source: increment-1 §2, FEED-v8-liftverify):** Undrivable is
**single-connected** (0.0% of frames ≥2 significant components) ⇒ the signed ±φ_bulk lift's Undriv side is
a clean single background ⇒ one signed field SUFFICES. Road is **multi-component in 37.2% of frames** (mean
1.38, up to 3 blobs) — a signed SDF represents multi-blob Road correctly (zero-set = multiple closed
curves), but the scaffold MUST be **multi-component-Road-aware**, and the multi-curve boundary pushes the
boundary-representation cost toward the HIGH end of the CONJECTURED 20–50 KB.

---

## E. THE RESIDUAL-CODER GAP — THE NAMED ENEMY (0.079 S)

Source: `v8_movable_residual_rollup_20260709.md` §B + §C. The generators carry the DOMINANT structure
(~70–83% of each edge's boundary). The uncovered boundary px (car concavities, faint/occluded lane
fragments, horizon secondary arcs) are the `residual_sidecar_owed`.

**Shared residual coder (MEASURED, bit-exact):** per-frame uncovered boundary px → row-major flat-index →
temporal/spatial delta → zigzag → brotli q11; roundtrip recovers the exact coordinate SET bit-for-bit
(verified all 3 edges). Per-edge residual S: Movable 0.01741 · horizon 0.01892 · Road↔Lane 0.04203.
**The residual coder is the honest bottleneck** — generic sparse-coord coding costs ~0.4–0.6 B/px because
the uncovered px are scattered short fragments. A chain-code residual coder buys only **−13%** (22.3 KB vs
25.5 KB on Movable) — the residual is genuinely near its coordinate entropy, not smooth-curve-compressible.
**Completion is expensive with today's mechanism.**

**Two MEASURED-headroom levers that move 0.140 → 0.061 (both un-exploited, NEITHER built):**
1. **Residual double-count / de-sharing (uncounted).** The horizon's uncovered "secondary arcs" ARE
   objects breaking the horizon (MEASURED 1.6–2.0 crossings/row) = Movable/Undriv px ALREADY carried by
   the Movable sites. Paying for them in the horizon residual DOUBLE-COUNTS; attributing them to the
   Movable carrier shrinks the horizon residual for FREE. Same class for Road/Lane fragments near cars.
   *Currently INFLATING the complete number, not deflating it — a de-sharing headroom.*
2. **Curve-relative residual coder (NOT built).** The residual px sit in a thin band around the generator;
   coding the signed OFFSET-from-generator per residual px (small integers) instead of absolute coords is
   the obvious lever — the generic 0.4–0.6 B/px sparse-coord/chain coder is an UPPER BOUND, not the floor.
   **This is the v8 rate-completion's #1 open real-coder lever.**

**S2/S4 own the residual-coder decision:** close 0.079 via {de-share + curve-relative} to land complete
below frontier, OR accept a lossy operating point (the Road/Lane waterfill knee sits at the finest/lossless
point — every quantization step RAISES S_total, S-4 §Arm 2; so lossy trades are net-negative on that edge),
OR ship the dominant-only number with the residual as a measured-owed sidecar. The choice must be a VALUE,
not a TBD (P2 contract).

---

## F. RECONCILIATION = MERGE → DIFF → CORRECT + the scorer-rule facts

Source: SPEC_v8 §3 + increment-1 §4 + CLAUDE.md §SegNet/PoseNet. The composite must survive the frozen
scorer; the reconciliation loop is how.

**The scorer-rule asymmetry (MEASURED facts, the routing basis):**
- **SegNet** = `smp.Unet('tu-efficientnet_b2', classes=5)`, reads **LAST frame ONLY** (`x[:,-1]`,
  modules.py:108), full-RGB, bilinear→(512,384), distortion = argmax disagreement rate. **⇒ frame0 is
  SegNet-FREE = pure pose territory; DIFF is frame1 ONLY.** Chroma is FULLY argmax-visible.
- **PoseNet** = FastViT-T12, 12-ch = 2 frames × YUV6 (4 luma + 2 SUBSAMPLED chroma), MSE on first 6 pose
  dims. **⇒ pose is luma-dominated; chroma is subsampled/low-passed before pose.**
- **The annulus law (MEASURED #333):** ~97% of d_seg lives in a ~4.7%-area boundary annulus ⇒ the error
  is boundary-JITTER, not region-miss; byte budget → separatrix/annulus precision, not interior texture.
- **P-A oracle bound (MEASURED, n600, probe bf1ee1fa8):** bulk interiors near-flawless through R (Road
  0.17% / Undriv 0.03% within-class) ⇒ "interiors near-free" — but this is the ORACLE/UPPER bound with
  real-frame texture; the generated-paint floor is UNMEASURED (P-C). verdict_scope of "interiors near-
  free": **CONDITIONAL** on P-C.

**The loop (MERGE → DIFF → CORRECT):** (1) MERGE tropical argmax composite → paint RGB frame1. (2) DIFF
frozen SegNet argmax on R(composite) vs intended partition (frame1). (3) CORRECT, channel-routed:
**CHROMA-FIRST seg repairs** (SegNet-strong, PoseNet-quiet in the high-freq annulus band; measured basis
#276) / **LUMA RESERVED** for pose/warp coherence. (4) Iterate to fixed point = **Dykstra alternating
projections onto (argmax-cell ∩ pose-tube) in channel-split coordinates (#73 reborn)**; unpaintable
residual = counted sidecar (#226 margin_conditional_residual).

**SCOPED (increment-1 review-D, the chroma-triangularity caveat):** the (luma, chroma) correction Jacobian
is NOT structurally triangular — PoseNet reads 2 subsampled chroma channels, so triangularity holds only
in the HIGH-spatial-freq band (annulus chroma edits get low-passed before pose); LOW-freq chroma recolors
pass into pose at near-full strength. It is also class-pair-dependent: Road/Undriv IS chroma-separable
(grey/green/blue — chroma-first justified), but **Road/Lane (41% of Road's flips) is LUMA-separable**
(bright lines on dark road) and CANNOT be chroma-repaired. **Increment-1 is safe because Lane is a
separate ANALYTIC carrier, not paint-repaired** — chroma-first is a warm-start heuristic for the Road/
Undriv paint only; the REAL guard is step 4 (MEASURE pose on composites + Dykstra), never the routing.

---

## G. THE 6 NAMED RISKS (SPEC_v8 §4/§5 + increment-1 §5 — binding on increment-1 review; all live at SEAMS)

1. **EDGE-DUPLICATION** → edge-centric single Road+Undriv field (§D). Cure adopted. **Standing catch:**
   SPEC §1 "one field per adjacency EDGE" (41 edges) vs the per-CLASS table (5 fields) — only the ONE
   Road/Undriv edge is edge-centric in increment-1; the "edge-centric" label is aspirational for full v8.
2. **THEFT MIGRATES TO THE COMPOSITE** (end-to-end training through paint→R→SegNet re-couples classes in
   the score gradient) → **STAGED training** (Stage A: fields vs EXACT SDF targets `signed_distance_fields`,
   argmax(sdf)==labels, NO cross-class gradient path; Stage B: paint solved SEPARATELY vs the frozen
   scorer). **FORBIDDEN: end-to-end paint→R→SegNet** (re-opens theft; needs its own measured justification).
   GUARD: b_c calibrated OUTSIDE the scorer-gradient loop (closed-form OT/Menon), never jointly optimized
   against d_seg — else the global-bias coupling re-enters as theft-like behavior (increment-1 §2 review-E).
3. **MASK-OPTIMAL ≠ SCORE-OPTIMAL + oracle-paint gap** → **P-C probe** ($0-labeled but HEAVY n600-through-
   R): flat/procedural fill paint floor per class, runs BEFORE the paint stage is designed. **NEW sub-
   finding (increment-1 §9):** flat-paint FAILS (0.0064 floor) ⇒ adequate texture needs class-typical
   statistics ⇒ those are VIDEO-DERIVED = a nonzero **COUNTED seed floor** that neither P-A nor P-C has
   isolated — P-C must report it. **BLOCKING PRECONDITION: P-C runs before the paint stage is designed.**
4. **TIE-VARIANCE ADDS** (independent fields jitter independently) → **annulus-precision byte allocation**
   (reduce δφ at the tie; increment-1 §9 corrected the cure from temporal-screw, which is the TEMPORAL
   axis and doesn't bound per-frame jitter) + the SDF gauge's tie conditioning + temporal-screw companion
   for the temporal axis.
5. **POSE SEAMS** → chroma routing (§F) + MEASURE pose on composites, never assert; luma stays ONE
   coherent warp-structured field (temporal-screw force is the necessary companion).
6. **APPARATUS ×5 + OPPORTUNITY COST** → gating: **v7.5-first / dual-chain wall** (§J). If v7.5.2's
   measured trajectory reaches target, v8 increments never fire.

---

## H. MEASURED NEGATIVES (do NOT re-open blind; verdict_scope canonical; flip-weighted reformulations noted)

- **N-1 #288 OT / Laguerre head-offsets — MEASURED NEGATIVE** (source: crucible-2 N-1, FEED-otoffset, eq
  `laguerre_ot_head_offset` REGISTERED). mod32cap ep650, realized-through-R, n48+n24 reproduce order+sign:
  no_offset **0.00272** < menon 0.00293 < **ot_newton 0.00487 (WORSE)**. OT enlarges the rare-Lane cell to
  hit its 0.59% GT mass (b_Lane≈+28.7) → OVER-predicts Lane → SegNet penalises. Both offset arms HURT.
  Solver EXACT (7 Newton iters, mass_err 0.0). **verdict_scope: FORMULATION** (mass-matching to raw GT
  frequencies as a d_seg surrogate). **OPEN reformulation = flip-weighted target masses** (match argmax to
  where FLIPS are, not raw area) — the b_c tie-calibration in the v8 carrier table inherits this: b_c must
  be flip-weighted, not area-matched. Larger-n OWED (probe not resumable-chunked).
- **N-2 lane-ξ ego-transport — MEASURED NO-GO** (source: crucible-2 N-2, v8_roadlane_ego_compensated_rate,
  eq `lane_groundframe_xi_transport_no_collapse_v1` REGISTERED, 2 accounting-mode anchors). ξ-advection
  ENLARGES the Road↔Lane stream EVERY arm (best 42,017 B vs identity 41,085 B; both accounting modes).
  WHY: the lane coder stores GROUND-frame coeffs (IPM already quotients ego-forward at FIT time) → residual
  is IRREDUCIBLE curvature evolution, not rigid transport. **Durable LAW (`lane_groundframe_xi_transport_
  no_collapse_v1`): for a chart that has already absorbed the ego DOF, ‖innov_ξ‖₁ ≥ ‖ΔQ‖₁ for all ξ —
  ego-freeze does NOT transfer.** Horizon won 14.6× only because its poly is image-frame (removable
  ego-pitch intercept). **verdict_scope: FORMULATION** (lane rate axis); ξ stays decisive for POSE + the
  image-frame horizon. **Consequence for v8: use 0.0275 S for Road/Lane; do NOT project a horizon-class ξ
  transfer onto any ground-canonicalized carrier.**
- **N-3 dense medial axis ≈ bitmap — MEASURED NO-WIN** (1.09×; FEED-v8-voronoi; §C). verdict_scope:
  FORMULATION (the *dense* generator; the few-coefficient fit is the lever).
- **N-4 waterfill on Road/Lane is net-negative** (source: v8_roadlane_ego_compensated_rate §Arm 2). The RD
  knee sits at the finest/lossless point; every quantization step RAISES S_total (distortion dominated by a
  fixed ~0.52 S band-geometry floor). verdict_scope: FORMULATION (this metric = filled-band vs thin painted
  lines; the coverage/band-width artifact, not a real quantization headroom). **Lossy trades on the lane
  edge are dominated — do NOT propose them as residual-gap closers.**

---

## I. POSE — BANKED + STORE-NOTHING MANDATE (composes with v8, does NOT disturb d_seg)

Source: crucible-2 rows P-1/P-2/P-5/P-7 + SPEC_v8 §7 + memory L68.

- **P-1 R1 dxi BANKED at n600 AUTHORITY:** d_pose **0.001610** → contribution √(10·d_pose) = **0.127**;
  ξ_eff **7,195 B** (rate 0.004791); 20× over no-dxi. [macOS-CPU advisory] NON-PROMOTABLE. Source:
  FEED-238resolved.
- **P-2 pose ⊥ d_seg EXACTLY (structural proof):** ∂d_seg/∂ξ ≡ 0 (SegNet reads ONLY last frame; ξ shapes
  ONLY seg-free frame0) ⇒ the pose carrier CANNOT disturb d_seg. **This is why the v8 reconciliation DIFFs
  frame1 only and reserves frame0/luma for pose.** Source: FEED-posejac.
- **P-5 HONEST FLAG:** pose is BANKED-AS-ARTIFACT but NOT solved-for-the-witness — whether a fresh v8 arm's
  terminal pose-finish CONVERGES to an R1-class dxi from ITS OWN basin is UNVALIDATED (memory refutes cheap
  post-hoc/stored carriers; only JOINT descent from a coherent render crosses the photometric wall). The
  operator POSE ENGAGEMENT GATE binds: pose fires only when d_seg is sufficiently conditioned (conditioning-
  gated EVENT, never epoch; never-reached fallback = ship banked R1).
- **P-7 STORE-NOTHING MANDATE + rate-basis lineage tag:** v8's fresh arm runs the `generated` store-nothing
  path (structurally ~1 KB rate). **Rate basis is lineage-tagged:** {store_nothing / generated = ~1 KB} vs
  {fresh_seeded v1→v5 real_keyframe = COUNTED 697,941 B}. **Any byte-close from a keyframe lineage MUST
  charge the counted-keyframe rate.** Restore store-nothing at the fresh arm (operator MANDATE). #314
  pose-carrier source = operator-routable. Source: FEED-drift-d2-fix.

---

## J. THE BASELINE — the sealed v7.5.2 single trunk (v8 must BEAT or COMPOSE with)

Source: crucible-2 SYNTHESIS_v3 (the launch surface) + ORCHESTRATION_LEDGER + row R-1.

- **v7.5.2 launch-1 ON-set (sealed):** {amber stability-preset precond + directional self-orient + #121
  d_seg-aware taper + AA-ipe + σ_cc′ fitted (all zero loss-share) + Chan-Vese counter-force baseline +
  #360 temporal-screw (the ONE event-gated P0 force) + terminal conditioning-gated pose}; **FRESH default**
  (not warm-start from the floored run-1 basin); pose-gate conditioning-first (rolling-slope-≈0 on de-noised
  σ_min PRIMARY; ship-banked-R1 fallback). Designed wall-clock floor ~6–16 h + ~11 min GPU head-solve.
- **run-1 (the birth-arm, STOPPED) MEASURED:** best d_seg **0.115102 @ ep325** (Road **0.312** + Undriv
  **0.083** DOMINATE; all 5 islands born; CE→tau event-fired @257). This is the birth-arm DIAGNOSTIC
  baseline, NOT the pointer-mover (pre-actuation config). Source: R-1, FEED-205stop.
- **v7.5's advisory S(n600) decomposition (source: P-1):** seg 0.455 (THE blocker) + pose 0.127 (BANKED) +
  rate 0.060 = 0.642 [advisory]. **d_seg is the entire remaining fight for v7.5.2.**
- **The v8 comparison thesis:** v8's rate is DEMONSTRABLY better on dominant structure (0.061 vs v7.5.2's
  ~0.060 is a WASH at dominant-only; v8's ADVANTAGE is that its geometry SCALES to the complete number and
  its d_seg mechanism DECOUPLES the cross-class theft that floored run-1's Road at ~0.40). **But d_seg is
  the shared true blocker** — v8's edge-centric decomposition is a d_seg BET (∂φ_c/∂θ_{c'}=0 kills the
  measured Lane 13.8× / Movable 4.6× theft into Road by construction), UNPROVEN through R until increment-1.
  The dual-chain comparison brief (P8) weighs v8-increment-1's projected S-path + risk register against
  v7.5.2's sealed one.

---

## K. INCREMENT-1 BUILD SURFACE — the REAL modules the config must compile against (argparse-crash check)

All verified on-disk 2026-07-09 (`src/tac/boundary_math/`); the increment-1 config in P2/P7 MUST compile
against these (the F-1 launch-path≠config-tests lesson — config-compilation tests are NOT the launch path):

| module | size | role in increment-1 |
|---|---|---|
| `road_undriv_bulk_field.py` | 27.1K | **the ONE new build** (scaffold) — the edge-centric Road+Undriv bulk-boundary field. **KNOWN BUGS: (1)** `bulk_boundary_byte_cost` measures FULL Road perimeter (2228 px) not the Road↔Undriv shared edge (426 px) — needs `mode="horizon_poly_xi"`; **(2)** must be multi-component-Road-aware (Road multi-blob 37.2% of frames) |
| `road_horizon_component.py` | 20.3K | the Road↔Undriv horizon generator (deg-3 poly + ξ) — composed by the bulk field |
| `hood_static_component.py` | 12.3K | MyCar static clamp — REUSE as-is |
| `lane_sdf_component.py` | 19.7K | Lane SDF band — REUSE |
| `analytic_lane_render_band.py` | 99.3K | the real Road↔Lane byte-close coder (LBND2/LBND3) — REUSE |
| `laguerre_logit_offset.py` | 24.2K | tie bias b_c — REUSE (b_c FLIP-WEIGHTED per N-1, calibrated OUT of the scorer-gradient loop per risk-2 guard) |
| `lever_b_levelset_generator.py` | 56.0K | the SDF coordinate-INR generator the bulk field composes |
| `ego_xi_trajectory.py` | 23.7K | ξ transport (`advect_centerline_coeffs` exact SE(2)) — used by horizon/pose, NOT by the ground-frame lane (N-2 NO-GO) |

**Triality wiring (increment-1 §7):** the Road+Undriv bulk-field lands as a DSL `Lever` factory;
`lever_registry.completeness()` must show 0 unmapped for its new flags (never-invent-flags). The two
council-flagged laws (tropical reconciliation; per-class carrier allocation) STAY FORMALIZATION_PENDING
until P-B/P-C/increment-1 anchors land. Each carrier byte-closed + resumable + bit-exact-at-decode BEFORE
composition (vehicle-OS manifest-per-carrier).

---

## M. MODAL / EXACT-EVAL ENVELOPE (operator addendum, inherited from crucible-2 M-1/M-2)

- **M-1 MODAL ≤$20 HARD CAP (#381):** earmarked for exact-eval rows (paired contest-CPU + CUDA on
  byte-closed increment-1 candidates — the ONLY promotion authority) + the owed CPU-torch n600 verdict
  queue. "Spend the budget to BUY exact rows."
- **M-2 witness training is MLX-LOCAL** (M5 Max), NOT Modal. Modal is for byte-closed exact-eval + the
  CPU-torch verdict authority. The torch-parity-twin-for-A/B-fan-out question was answered NO-GO in
  crucible-2 (S4/P3b: the byte-closed ARCHIVE is the shared invariant, so exact-eval on the archive is
  parity-safe; a parity twin adds build cost + MLX↔torch numeric-parity risk for no throughput the serial
  local byte-close + governed CPU verdict doesn't already give). **Carried forward as SETTLED unless a v8-
  specific reason re-opens it (S4 confirms).**

---

## SETTLED — do NOT re-derive / re-open (SPEC_v75 §8B + SPEC_v8 §8 + crucible-2 negatives)

| settled | authority |
|---|---|
| Class order = canonical comma10k [Road0,Lane1,Undriv2,Movable3,MyCar4]; luma-sort is WRONG (bit us 3×); carriers SELF-DETECT class by spatial/static signature, never hardcode index | CLAUDE.md §SegNet; SPEC_v75 §8B |
| Bulk-paint interiors near-free AT THE ORACLE BOUND; residual = 100% separatrix placement; Road = adjacency hub | probe P-A bf1ee1fa8 (n600) |
| dense medial ≈ bitmap (1.09×, NO WIN) — the few-coefficient generator is the lever, not the dual choice | FEED-v8-voronoi (N-3) |
| lane-ξ ego-transport NO-GO (ground-canonicalized chart) — use 0.0275 S for Road/Lane, no ξ transfer | eq lane_groundframe_xi_transport_no_collapse_v1 (N-2) |
| OT/Laguerre area-mass-matching head offsets HURT (both arms) — only FLIP-WEIGHTED reformulation is open | eq laguerre_ot_head_offset (N-1) |
| Waterfill on Road/Lane net-negative (knee at lossless) — no lossy residual-gap closer on that edge | v8_roadlane_ego_compensated §Arm 2 (N-4) |
| Road floor actuator = birth-stack recall-without-precision, NOT the analytic band (falsified 3×) | road_anomaly_probe b9da25aa6 |
| δ_R = 0.0196 (RE-RUN tools/measure_delta_R_noise_floor.py for n600, never rebuild) | reports/delta_R_noise_floor.json |
| Pose 3.4e-5 is ANCESTOR-BORROWED; this vehicle's d_pose is OPEN; R1 dxi 0.001610/7.2KB is BANKED | L68; crucible-2 P-1 |
| Micro-batch bit-identity-at-speedup IMPOSSIBLE (scorer forward batch-dependent) — bounded n600 A/B is the ONLY admission | frozen_scorer_forward_batch_dependence_v1 |
| Torch-parity-twin for Modal fan-out = NO-GO (byte-closed archive is the parity-safe invariant) | crucible-2 S4/P3b |
| R-phase FOLDS into tie-locus (do NOT build two terms; term already built) | p0_forces_derivation f7209667a |

---

## OPEN QUESTIONS — the 6 sharpest (each P1 seat takes an INDEPENDENT position)

1. **CARRIER COMPOSITION FOR INCREMENT-1 (the central question).** SPEC_v8 §6 says increment-1 = de-share
   the Road separatrix into ONE edge-centric Road+Undriv bulk-boundary field (de-shares ~99% of measured
   flip mass in ONE build) while Movable/Lane/MyCar keep existing carriers. Is that the smallest decisive
   build, or does the residual-coder gap (§E) demand a residual-coder co-build in increment-1? Which
   carriers are truly independent (compose) vs which share separatrix information (edge-centric, must not
   double-pay)? (S1/S2 own; S6 derives the decomposition BLIND.)
2. **THE RESIDUAL-CODER DECISION (the named enemy, §E).** Close the 0.079 S gap via {de-share double-count
   + curve-relative residual coder} to land COMPLETE below the 0.118 frontier — or ship dominant-only
   (0.061) with the residual as a measured-owed sidecar — or accept a lossy operating point (N-4 says lossy
   is dominated on the lane edge)? Which of the two headroom levers is buildable in increment-1, and what
   does each MEASURE-owe? A VALUE, not a TBD. (S2/S4 own.)
3. **STAGED-TRAINING + RECONCILIATION CONTROL (risk-2/risk-3, §F).** Is the merge→diff→correct loop's
   fixed-point (Dykstra) the right control, and does the b_c-out-of-loop guard (calibrated closed-form,
   never joint-vs-d_seg) fully close the theft channel? What does P-C (flat/procedural-fill paint floor +
   counted-seed-floor) have to report BEFORE the paint stage is designed (BLOCKING PRECONDITION)? Is P-C
   the right decisive measurement, and can it run $0 or is it governed-heavy? (S3/S4 own.)
4. **SCORER-RULE ROUTING (§F).** Chroma-first/luma-reserved is a warm-start heuristic scoped to Road/Undriv
   (chroma-separable); Road/Lane is luma-separable and NOT paint-repaired (separate analytic carrier). Is
   the routing over-claimed anywhere in the composite, and is "MEASURE pose on composites + Dykstra" (not
   the routing) the real guard? Does DIFF-frame1-only + reserve-frame0/luma-for-pose hold under the
   composite? (S3/S4 own.)
5. **POSE COMPOSITION + STORE-NOTHING (§I).** v8's fresh arm restores store-nothing (~1 KB, MANDATE). Does
   the terminal conditioning-gated pose-finish converge to an R1-class dxi from a v8 basin (P-5 HONEST FLAG:
   efficacy UNVALIDATED), or does v8 ship the BANKED R1 dxi (0.127/7.2 KB) via the never-reached fallback?
   How does the pose-conditioning gate read d_seg-sufficiency on the v8 decomposed trunk (per-class vs
   aggregate)? (S4 owns; S3 on the gate control law.)
6. **v8-vs-v7.5.2 + THE OPPORTUNITY-COST GATE (risk-6, §J).** v8's rate is a wash-to-better at dominant-
   only and its d_seg mechanism is a decoupling BET (unproven through R). Under the dual-chain wall, what
   projected S-path + wall-clock budget + risk register does increment-1 carry into the comparison brief,
   and what MEASURED evidence (increment-1's byte-close A/B + d_seg-through-R) would make v8 the which-to-run
   pick over the sealed v7.5.2? Is increment-1 worth the apparatus×5 cost given v7.5.2 is launch-ready?
   (S5 pre-mortems the whole stack; S1/S2 own the S-path projection.)

---

**Adversarial self-check (per the anti-blind-spot contract):** cross-checked the rate ledger against the
3 measurement memos + FEED-v8-rollup + the registered equation `v8_geometric_rate_decomposition_v1`; the
0.061/0.140/0.079 numbers reconcile across all three sources. Negatives INCLUDED so a seat does not
re-open them blind: N-1 (OT mass-match — flip-weighted reformulation open) / N-2 (lane-ξ — LAW registered)
/ N-3 (dense medial) / N-4 (lane waterfill). The residual-coder gap (0.079) is surfaced as THE named enemy,
not buried. **UNSOURCED-VERIFY-IN-P4:** the CONJECTURED 20–50 KB Road+Undriv boundary-representation band
and the derived −50..75% ≈ 0.049 S net-stack headroom are DOUBLY-CONDITIONAL estimates (presuppose
increment-1's byte number AND d_seg holding) — NOT measured; a seat citing them must carry the CONJECTURED
tag. The counted-seed-floor (video-derived texture statistics for adequate paint) is MEASURED-owed by P-C,
UNMEASURED as of this pack. If a seat needs a carrier byte number not in §B/§D, it is a grounding gap —
flag it, do not guess.
