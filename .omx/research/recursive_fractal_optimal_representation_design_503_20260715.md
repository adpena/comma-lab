# Task #503 — THE RECURSIVE-FRACTAL-OPTIMAL REPRESENTATION (per-dimension optimal rep + composition law + ranked build order)

**Date:** 2026-07-15 · **Agent:** design subagent (Opus) · **Task #503 (P0, operator 2026-07-14).**
**Axis:** ALL `[macOS-CPU/MLX advisory · research-signal · NON-PROMOTABLE]` MEANS. **Pointer UNMOVED:
submittable contest-CPU 0.19108282; borrowed bank 0.18804 (NON-SUBMISSION).** This is a DESIGN +
$0-feasibility-ranking pass → ONE memo. **NO code edited, NO training/heavy/paid launch, NO archive
mutation, NO exact-eval, NO pointer movement.** The BUILD is sequenced later (needs the DSL, owned by a
live arm — I did NOT touch `src/tac/witness_dsl` or `canonical_equations`).

**STORES CONSULTED (proactive-recall, do-not-redo):** `fullstack_fractal_optimal_synthesis_20260710.md`
(#398 thread-B — the unique-home clause-A/B map + P1–P12 recursive-at-every-SCALE table) ·
`philosophy_pass_v752_20260709.md` + `philosophy_pass_v8_20260709.md` (#392 P1–P12) ·
`waterfill_boundary_spectrum_curvelet_vs_fourier_probe_20260714.md` (#502 GO — the fresh measured
curvelet number) · `island_birth_saddle_node_hysteresis_measurement_20260715.md` (per-class birth-weight
∝ (P/A)_c) · `v8_increment1_design_draft_20260709.md` (Laguerre generators / horizon-poly+ξ rate) ·
`codex_findings_recursive_fractal_optimal_representation_v9_20260714_codex.md` (the sister DCB build +
palette-rank-15 refutation) · `cgauge_master_action_and_parametrization_20260711.md` (the master action
+ obligation matrix) · MEMORY L1/L-v8/L17/L25/L65/L66/L68/L71/L75/L83/L85. Unified level-set flow (L1).

---

## 0. ANSWER FIRST — the recursion IS the composition law

There is ONE scored object: the **textured covariant Laguerre partition on the scorer's obligation
matrix**, `W = (G, ξ, T)` (#398). "Recursive-fractal-optimal" is not a metaphor and not a per-level
grab-bag — it is a **single operation applied self-similarly down the dimension ladder**:

> **At every dimension, store the GENERATOR of that dimension's scorer-relevant structure; DERIVE the
> finer dimension as an evaluation, a warp, or a residual of the coarser generator; store NOTHING a
> coarser generator + a cheap warp already produces.**

This one rule instantiates differently at each dimension but is the SAME rule, and it is exactly the
double-counting cure: the fractal reuse of ONE object (the same `ξ` that warps a pair warps the horizon
intercept and is the pose channel; the same class generators whose tie-loci ARE the boundary; the same
argmax field evaluated at a coordinate IS the pixel). **The archive stores a small set of GENERATORS,
one per dimension, each in its unique geometric home (clause A) at its minimal/waterfilled dimension
(clause B); every finer structure is DERIVED FREE at decode (rule-118).** This is why sub-0.15 is a
non-RGB task-space object, not an RGB codec: RGB stores the finest dimension (pixel) explicitly and pays
for all the coarser structure implicitly and redundantly; the fractal representation inverts that.

The genuinely NEW contribution of #503 over #398 (which organized by P1–P12 SCALE) and the codex level-
table (which listed per-level findings): the **per-DIMENSION recursion + the nesting/warp composition law
that binds them into ONE archive without double-count**, plus a **measured S-impact-per-byte ranking**
for which RGB-replacement to build first.

---

## 1. THE PER-DIMENSION TABLE (measured structure → optimal rep → RGB-replacement → $0 feasibility)

`S = 100·d_seg + √(10·d_pose) + 25·|archive.zip|/37_545_489`. Obligation matrix (MEASURED+from-code,
#398 §1): frame_0 seg-price **exactly 0** (n600 8.5e-9) → ξ-only; frame_1 luma = the ONLY doubly-priced
block → G+T; frame_1 chroma-HF-384 luma-null plane = pose-free (op-null 3.4e-6) → T's home; frame_0⊗
chroma-HF = dead subspace.

| dim | MEASURED structure (anchor) | optimal representation | RGB-replacement object | $0 feasibility |
|---|---|---|---|---|
| **PIXEL** | A pixel is the scorer's **argmax evaluation** at a coordinate; interiors near-oracle (Road 0.17%/Undriv 0.03% within-class through-R, #398 §1.1). BUT pair-0 palette response is **rank 15/15 in Seg decision space** (codex MEASURED, NOT low-rank). | NOT stored: `P(x)=argmax_c(φ_c(x)+b_c)` — the pixel is generated from the coarser class field. Only frame_0 RGB + task-coords are inputs. | coordinate + the class generator field (no per-pixel RGB). | **MEASURED-CAUTION.** Naive "task storage is low-rank at pixel level" is REFUTED (palette rank 15/15). Pixel is not a store dimension — it is the READOUT. No separate build; falls out of CLASS/BOUNDARY. |
| **CLASS** | Argmax partition = a **Laguerre / tropical power-diagram** (#284, L-v8): each class = one site + one weight; the partition and its boundaries are DERIVED. Per-class basin prices MEASURED (flat floors: Movable 0.403 / MyCar 0.147 / Lane 0.138 the enemy; Road wins flat 0.017). | v8 per-class carriers: MERGE→DIFF→CORRECT, **store GENERATORS not boundaries**. Hood static mask (#139 IoU 0.994); Undriv default basin + 3 lateral curves; Road flat colour + horizon arc; Lane analytic band; Movable sparse sites (K=9). | per-class geometric generators (sites/weights/curves/masks) replacing the RGB region content. | **MEASURED/DERIVED.** Carriers BUILT in-tree; Movable coder 6289 B MEASURED (−31%); MyCar/Undriv/Road pins MEASURED. Full decoupling A/B = OWED (v8 1a screen, $0-but-not-yet-run). |
| **BOUNDARY / separatrix** | d_seg lives on a **codim-1 ORIENTED annulus**: ~4.7% area carries ~97% (n16) / **26.8% flip mass (n600 MEASURED)** (L66, codex); boundary margin spectrum **41× anisotropic** (n600), bimodal normal+tangent (#502). Global sinusoid is provably sub-optimal here (Torralba–Weiss: isotropic-Fourier optimal ONLY under stationarity we lack). | **Oriented localized frame** (curvelet / shearlet / steerable), per-orientation waterfilled — NOT more isotropic Fourier. Boundary is a TIE-LOCUS derived from CLASS generators, never independently stored. | oriented boundary atoms (4–8 px along-tangent) replacing RGB edge texture. | **MEASURED GO (#502).** Oriented allocation needs **1.7–2.0× LESS rate** than isotropic to match distortion; win GROWS with budget. This is a linear-basis UPPER bound; realized d_seg through-R is OWED (the build). **Freshest measured $0 GO → the #1 build (§3).** |
| **FRAME** | Frames within a clip are an **ego-rigid warp** of one keyframe: horizon arc = deg-3 poly, cubic/quad coeffs FROZEN (|Δ|≈1e-7/6e-5), intercept moves ~1.2 px/frame = ego pitch (MEASURED, v8 draft §1). | Keyframe generator + tiny per-frame **ξ-warp** + small residual sidecar. Store frame-0's structure ONCE; frames 1..599 = warp. | keyframe task-generators + ξ-warp params (no per-frame RGB). | **MEASURED.** Horizon-poly+ξ store = **4.7 KB@n600 = 0.0032 S**, real coder (zlib on delta-coded fp16 coeffs), **88× below naive 0.282, 8× below generic chain-coder**. Dominant-arc only; residual secondary-arc sidecar OWED. Biggest MEASURED rate win in the stack. |
| **PAIR** | The scored unit is a 2-frame pair; the pair's temporal motion is ONE **se(3) screw twist** `ξ` (Chasles); the SAME `ξ` warps the partition (d_seg regularizer) AND is the pose (d_pose) — dual-use (L1). | ONE 6-dim `ξ` per pair = the pair glue; frame_0 = pure pose territory (seg-free), frame_1 = G+T. | per-pair se(3) twist (6 scalars) replacing pair RGB delta. | **MEASURED (banked).** R1 dxi **d_pose 0.001610 → contribution 0.127, 7.2 KB@n600** (n600 authority through byte-close, L68). Chart-selection LAW bounds where ξ pays (horizon 14.6× YES; ground-frame lane NO). Historical values do NOT transfer to a fresh vehicle (re-measure). |
| **EPOCH / curriculum** | The curriculum IS a **continuation path**; class-occupancy instabilities (island-birth) are **saddle-node/subcritical thresholds** (bistability hint #300: un-gated seed absorbed back). Birth-drive vs MCF homogenization-drain balance ⇒ threshold `W_birth,c* ≈ δ·(P/A)_c`. GT geometry MEASURED. | Continuation schedule (J1 coarse → J2 nucleate-under-area → J3 sharpen → J4 orthogonalize → J5 head-solve → J6 pose-gate), with **per-class birth-weight ∝ (P/A)_c** so rare classes cross nucleation together. | (the epoch dimension has no ARCHIVE bytes — it shapes G/T/ξ; its "representation" is the DSL Lever schedule.) | **DERIVED.** Per-class ratio **Lane ≈ 8.9× Movable** from MEASURED GT isoperimetric geometry (n96). Absolute λ_c/δ needs a live W_birth up/down ramp (operator-GO, NOT $0). Saddle-node NOT CONFIRMED at $0 (frozen sweep is smooth; hysteresis unmeasurable on frozen data — a static f(λ) has no memory). |

Two supporting sub-dimensions (from the obligation matrix, not primary but they fix routing):
- **CHROMA** (a d_seg lever, operator 2026-06-25): SegNet reads RGB → chroma flips the argmax. Optimal
  home = frame_1 chroma-HF-384 luma-null plane (pose-free, authority 2.73e-3 MEASURED); pre-image through
  the exact D kernel (naive camera-res dither leaks 50% → FORBIDDEN). BUILT; full textured-carrier consume
  = NO_VERDICT (codex).
- **SCALE / FREQUENCY** (the recursion's own axis): partition-of-unity + unique-home recursion prevents
  double storage across scales; Fourier interior + genuine localized 4–8 px along-tangent annulus atoms is
  the OPEN optimal form (#502 corrected — the stopped comparison was confounded, NOT ranking evidence).

---

## 2. THE COMPOSITION LAW — nest the generators; nothing at a finer dim the coarser produces

The per-dimension representations compose into ONE archive by the **fractal-nesting rule**, which is a
strict refinement of #398's clause-A unique-home map (`fullstack_unique_home_assignment_v1`, REGISTERED)
made explicit as a recursion down the dimension ladder:

1. **CLASS is the root generator.** Store `{site_c, weight_c, b_c}` per class (Laguerre) + per-class fill
   colour (~16 B) + per-class carrier (hood mask / lateral curves / Movable sites). The **argmax field**
   `φ_c(x)+b_c` is these generators.
2. **BOUNDARY is DERIVED, never stored** — it is the tie-locus `{x : φ_c(x)+b_c = φ_{c'}(x)+b_{c'}}` of
   the class generators. The oriented curvelet atoms (#502) are a **precision residual on the annulus
   only** (4.7% area), stored where the derived tie-locus is not accurate enough — NOT a second copy of
   the boundary. Edge-centric: each boundary is owned by exactly one class-pair (`owns_explicitly`), so
   Road↔Undriv is paid ONCE (81% of Road's other edges belong to those neighbours' carriers, v8 draft).
3. **PIXEL is DERIVED, never stored** — `argmax` readout of the class field at a coordinate; interiors are
   the near-free oracle bound; T (texture trunk, 375 counted params) modulates interiors in the stem
   passband [4..8] render-px only, chroma-HF-first.
4. **FRAME is a WARP of the keyframe** — store keyframe generators ONCE; frame `t` = the SAME per-pair `ξ`
   applied to the horizon intercept (and the whole ego-rigid field), + a small residual sidecar. The
   frozen cubic/quad coeffs are stored once, not per frame.
5. **PAIR is the `ξ` glue** — ONE se(3) twist per pair, DUAL-USED for both d_pose (the stored pose target)
   and d_seg (the partition warp / temporal-screw regularizer, stop-grad). **The same ξ appears in FRAME
   (warp), PAIR (glue), and pose (d_pose) — stored ONCE, read three ways. This is the load-bearing
   double-count cure.**
6. **EPOCH shapes, does not store** — the curriculum's per-class birth-weight ∝ (P/A)_c decides WHICH
   generators nucleate and WHEN; it produces the trained G/T/ξ but adds zero archive bytes.

**Rule-118 split (the counted/free boundary):** FREE in inflate.py = generator ALGORITHMS + deterministic
tables (Gabor bank, Fourier/curvelet frame, curve rasterizers, Laguerre argmax, ξ-warp). COUNTED in
archive.zip = the video-derived MINIMAL statistics: per-class sites/weights/colours + curve coeffs + W_tex
(375) + ξ stream (7.2 KB) + Movable site stream (6.3 KB) + curvelet annulus residual + horizon coeffs
(4.7 KB). The **pairwise non-derivability audit at byte-close is the standing gate** (no counted byte
reconstructible from {other sections + rule-118 generator}). Parsimony (L-v8) is the win: whole-scene
generator store ≈ 0.02–0.05 S vs 0.118 incumbent — because we store the ~8-dim generators, not the RGB.

**Fractal-conformance (recursive at every scale, from #398 §5, inherited verbatim):** the SAME
{one-fact-one-store · noise-floor · tolerance-budget · falsifier · floor · no-proxy · composition · unique-home ·
min-or-waterfilled-dim} discipline holds at constant → lever → section → config → chain. #503 adds the
orthogonal DIMENSION axis (pixel→class→boundary→frame→pair→epoch) to that SCALE axis — the two axes are
independent and both recursive.

---

## 3. RANKED BUILD ORDER (bias: land a lower exact score soonest — S-impact-per-byte)

The sub-0.19 gap is **ENTIRELY d_seg** (pose banked at 0.001610, L68). So the ranking weights d_seg
levers first, rate levers where they are large-and-measured, and defers dimensions that need a non-$0
probe.

| rank | dimension | build target | measured S-impact basis | feasibility |
|---:|---|---|---|---|
| **1** | **BOUNDARY** | **oriented curvelet/shearlet annulus frame (#502)** | d_seg is 100% of the gap (L68); ~97%(n16)/26.8%(n600) of d_seg flip mass is in the 4.7% oriented annulus (L66); oriented allocation is **1.7–2.0× more rate-efficient** there (#502 MEASURED GO), distortion gap growing to ~47% at high budget. Freshest measured $0 GO. | GO'd; build = genuine localized frame (NOT more Fourier); realized d_seg through-R OWED (the build itself). |
| **2** | **FRAME** | **horizon-poly + ξ carrier** (`mode="horizon_poly_xi"` on the bulk-field scaffold) | **4.7 KB@n600 = 0.0032 S** real-coder, **88× below naive / 8× below generic chain-coder** (MEASURED, v8 draft). Largest MEASURED rate win; ego-amortized. | MEASURED; dominant-arc done, secondary-arc residual sidecar OWED. Parallel to #1 (rate-side, disjoint). |
| **3** | **CLASS** | v8 per-class decoupling screen (1a, $0) → carrier composition | Movable is the coverage enemy (flat 0.403); carriers MEASURED (6289 B Movable −31%; hood/Undriv/Road pins). Decoupling closes MEASURED gradient-theft (Lane 13.8×/Movable 4.6× stealing Road). | $0 screen not-yet-run; carriers BUILT; full A/B OWED. |
| **4** | **EPOCH** | per-class birth-weight `∝ (P/A)_c` as a DSL Lever (Lane 8.9× Movable) | DERIVED from MEASURED GT geometry; fixes island-starvation (the plateau cause, L2/L3 — NOT capacity). Zero archive bytes; improves d_seg via nucleation. | DERIVED ratio ready to wire; absolute λ_c needs a live ramp (operator-GO). |
| **5** | **PAIR** | pose finish vs banked 0.001610 (rollback-guarded) | pose already banked (contribution 0.127); an in-basin finish is OPTIONAL improvement over a real floor. | rides the launch; not a gap-mover. |
| — | **PIXEL** | (no separate build) | palette rank 15/15 → not a store dimension; it is the argmax readout of #1/#3. | derived; naive pixel-task-storage REFUTED. |

**#1 RANKED BUILD TARGET: the oriented curvelet/shearlet boundary-annulus frame (#502).**
**Measured S-impact basis:** the entire sub-0.19 gap is d_seg (L68); d_seg concentrates ~26.8% (n600) of
its flip mass in the 4.7%-area oriented boundary annulus (L66); on OUR measured frozen-SegNet boundary
spectrum (41× anisotropic, n600) an orientation-adaptive frame reaches the same distortion at **1.7–2.0×
lower rate** than the isotropic Fourier features the witness currently uses (#502 GO), with the win
GROWING with budget. It is the only dimension carrying a fresh, robust, $0 MEASURED go-signal that
targets the exact quantity blocking the pointer. (The #502 number is a linear-basis UPPER bound; the
realized d_seg must be MEASURED through-R on the exact bytes — that measurement IS the build, per the
optimal-form + no-fake disciplines.)

---

## 4. HONEST GAPS — DERIVED vs $0-probe-owed vs live-run-owed

- **MEASURED, feasible now:** BOUNDARY curvelet capacity (#502 GO); FRAME horizon-poly+ξ rate (0.0032 S);
  PAIR pose bank (0.001610); the obligation-matrix routing; CLASS carrier bytes (Movable 6289 B etc.).
- **DERIVED, ready-to-wire, no $0 probe needed:** EPOCH per-class birth-weight ratio ∝ (P/A)_c (Lane 8.9×
  Movable); the composition/fractal-nesting law (it composes already-measured anchors, no new equation).
- **$0-probe-owed (design done, measurement not yet run):** CLASS full decoupling screen (v8 1a); CHROMA
  384-band legibility (Fable E-1); the FREQUENCY optimal-form fresh-start curvelet-vs-shearlet family
  comparison (#502's stopped comparison was confounded — a clean $0 basis screen is owed BEFORE the build
  commits to curvelet vs shearlet vs steerable).
- **Live-run-owed (NOT $0, operator-GO):** the absolute island-birth λ_c/δ (needs a quasi-static W_birth
  up/down ramp on a resumed EMA-BEST); the realized d_seg of the built curvelet frame through-R; **the
  composed launch itself — THE pointer question**. PIXEL: naive task-storage-is-low-rank REFUTED at
  palette level (codex, rank 15/15) — pixel is confirmed a readout, not a store dimension, but the full
  DecisionCarrierBundle rate verdict is NO_VERDICT (no alternate receiver-closed n600 archive A/B exists).

**Codex sister-status caveat (503 lane):** the codex arm returned `V9_INTEGRATION_BLOCKED_OWNER` /
`NO_VERDICT_RECEIVER_RATE_CUSTODY` — the DCB surfaces are built-default-OFF but not live-wired, and no
byte-saving is proven. My design does not assert otherwise; the decisive falsifier for the whole recursion
is the actual alternate receiver-closed n600 archive A/B, not any proxy rank/capacity narrative.

---

## Triality legs + pointer honesty

- **DAG:** FEED-fractal503 appended to `sub015_DAG_exp_linear_reparam_warmstart_20260714.md`.
- **equations:** N/A-with-rationale — the fractal-nesting composition law COMPOSES the already-registered
  `fullstack_unique_home_assignment_v1` + component anchors (obligation matrix · Laguerre/#284 · #502
  curvelet capacity · horizon-poly+ξ · banked ξ · (P/A)_c birth-balance); no NEW law is asserted at design
  time (per SPEC_v8 §5 discipline — candidate composites stay council-flagged until anchors land). The
  registry is owned by a live arm; I did not touch it.
- **DSL:** N/A-with-rationale — this is a DESIGN authority output; the per-class birth-weight Lever and the
  curvelet-frame lever fold into the DSL at BUILD time (the DSL is a hot file owned by a live arm; no
  silent orphan — the two owed levers are named here + in the #398/#502 ledgers).

**Pointer UNMOVED (submittable 0.19108 / bank 0.18804). This memo is MEANS — a design + $0-ranking
authority output; no launch fired, no archive built, no exact-eval, the pointer moves ONLY through a
byte-closed `upstream/evaluate.py` n600 exact row < 0.19108.**
