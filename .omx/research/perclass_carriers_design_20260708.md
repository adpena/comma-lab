# PER-CLASS CARRIERS + SCORER-RULE RECONCILIATION — the argmax-native decomposition (task #359, design derivation)

**Date:** 2026-07-08 · **Axis:** design-only, $0, no trainer edits, no launches; run-1 (pid 63069) UNTOUCHED ·
**Pointer 0.19110 UNMOVED (means).** · **Operator riff (verbatim):** *"What if we did separate classes
separately and then took masks and then reconciled according to upstream auth eval scorer?"* + *"Exactly
engineered correctly aligned with deep math all falls out perfectly and optimally."*

**Relation to the campaign:** v7.5 counter-force (FP-area penalty + completion events, in-flight) = the
near-term IN-PLACE fix for the measured Road-floor theft bug; THIS memo = the v8-class ARCHITECTURE that
structurally obsoletes the bug class. v7.5 launches first; v8 build is gated on this design + crucible
sequencing (FEED-perclass).

## STORES CONSULTED
- CLAUDE.md §WITNESS CAPSTONE + §unified level-set flow + canonical class order
  (Road0/Lane1/Undriv2/Movable3/MyCar4, MEASURED — never luma-sort) + docs/operating_manual_craft_handoff.md.
- `.omx/research/road_anomaly_probe_20260708.md` — THE theft measurement: Lane 13.8× / Movable 4.6×
  over-paint, EXACT mass conservation (rare excess 0.1191 ≈ majority deficit 0.1189), 71% flip mass GT-Road,
  79% interior; actuator = birth-stack recall-without-precision.
- DAG FEED-perclass + FEED-roadfloor + FEED-af (symbolic MDL floor) + FEED-ly (#210 oracle-R gate) +
  FEED 2026-06-26ba (4-part witness structure) + FEED 2026-07-08 sweep (#308 grids-vs-INR).
- Built components: `src/tac/boundary_math/lane_sdf_component.py` (self-detecting; paint-then-SDF injection),
  `hood_static_component.py` (self-detecting; `hood_mask_byte_cost`), `lever_b_levelset_generator.py`
  (`signed_distance_fields`: argmax(sdf)==labels EXACTLY; `apply_R_to_fields`),
  `legal_frame_bridge.py` (palette rasterizer; frame0-is-SegNet-free pose_carrier strategy),
  `context_partition_codec` family (#180 Morse-Smale/partition codec lineage).
- `src/tac/witness_dsl/curriculum_dsl.py` ~2523 (`DirectionalBasisRebalance`, the two-regime allocation law
  + `persistence_classes_for_basis_regime` coherence coupling).
- MEMORY L68/L69 (pose OPEN+UNMEASURED on witness; 3.4e-5 = ANCESTOR; store-nothing ξ derive-H #257 PROVEN).
- v7.5 memo `.omx/research/v75_birth_counterforce_20260708.md`: **NOT YET LANDED** at write time — composition
  claims below are against its DAG-FEED-roadfloor spec (FP-area penalty · post-birth completion event ·
  logit-adjust softening · birth ramp), to be reconciled when it lands.

---

## 1. The reconciliation operator (the deep math — it does all fall out)

**The scorer's own rule IS the reconciliation.** `evaluate.py` scores `d_seg` on the SegNet **argmax** —
a partition, i.e. the tropical (max-plus) projection of K scalar fields. Our witness already represents the
partition as `argmax_k φ_k` of K=5 signed-distance fields (`lever_b_levelset_generator`). The operator's
decomposition is the statement that the **max-plus semiring is the ONLY coupling the scorer imposes**:

```
P(x) = argmax_c ( φ_c(x) + b_c ),   c ∈ {Road0, Lane1, Undriv2, Movable3, MyCar4}
```

- **Five decoupled scalar fields.** Each φ_c has its OWN parameter set θ_c and its OWN carrier (heterogeneous,
  §3). The composite partition is DERIVED at reconciliation; the **separatrix is the tie locus**
  `{x : φ_c(x)+b_c = φ_c'(x)+b_c' ≥ all others}` — codim-1, computed, **never represented**. This is the
  argmax-NATIVE chart: the scorer never sees fields, only the tie structure.
- **Theft-impossibility (structural).** The measured Road-floor bug (road_anomaly_probe: Lane paints 13.8×
  its GT area, mass conserved exactly out of Road) requires a SHARED substrate: recall pressure on Lane moves
  shared-trunk features/logits and drags Road with it. With per-class-independent parameters,
  `∂φ_c/∂θ_{c'} = 0 for c ≠ c'` — a growth loss on Lane **cannot move Road's field**. Cross-class
  interaction survives ONLY at the argmax comparison, where it is governed by the K-vector of biases b_c and
  the fields' common SCALE — an explicit, ~0-byte, per-class-measurable calibration instead of an emergent
  loss pathology. The bug class is obsolete BY CONSTRUCTION, not by counter-force.
- **Common scale = the SDF gauge.** argmax needs commensurable fields. The 1-Lipschitz SDF parametrization
  (eikonal-constrained, `|∇φ_c| = 1`) puts every field in PIXEL units: ties are equidistance loci, and the
  per-pixel flip budget is the geometric margin `m(x) = φ_(1)(x) − φ_(2)(x)` (top1−top2). Interior pixels
  have deep-SDF margins (error-tolerant); the annulus has small margins — the error budget concentrates
  exactly on the measured Fisher/margin geometry (Pearson 0.978, CLAUDE.md). Nothing new is needed: the SDF
  gauge we already use IS the reconciliation calibration.
- **Per-class area-Lagrange + completion become trivially per-field.** The v7.5 counter-force constrains
  `E[1{φ_c wins}] = a_c` (GT area) with multiplier λ_c. Under argmax, class masses satisfy `Σ_c a_c = 1`
  STRUCTURALLY (every pixel assigned once), so the 5 constraints have 4 DOF and the per-class multipliers
  are the exact Lagrangian dual of the mass-conservation identity the probe measured (deficit ≡ excess,
  0.1189 ≈ 0.1191). In the shared trunk this is a fight between coupled losses; per-field it is five small
  independent constrained flows.
- **Per-class τ_c/β_c annealing = five independent small flows.** Each carrier anneals its own smoothing
  (τ_c homotopy, MCF/viscosity window, birth/completion events) on its own schedule — the operator's
  "precisely desired annealing behavior." Rare classes (Lane, Movable) can be born aggressively without
  de-weighting the majority-class recall (the logit-adjust −5.14/−4.39 damage in run-1 becomes unnecessary:
  Menon adjustment was only needed because one softmax coupled the classes).
- **Training target per field is exact and local.** `signed_distance_fields(labels, K)` already produces
  per-class SDF targets with `argmax(sdf) == labels` EXACTLY. Fit φ_c to GT-SDF_c (regression in the
  annulus band, don't-care deep interior) — no softmax, no cross-entropy coupling, no class competition in
  the loss at all. Competition exists only where the scorer creates it: at the tie.

**d_seg decomposition under the operator (first-order geometry).** A flip at x requires the LOCAL ordered
pair (c_win, c_2nd) to swap: `δφ_{c_win}(x) − δφ_{c_2nd}(x) < −m(x)`. So
`d_seg = Σ_{pairs (c,c')} ∫_{tie(c,c')} P[pairwise swap]` — d_seg is a sum of **pairwise tie-locus
displacement terms**, each depending only on the TWO adjacent fields' errors near their shared separatrix.
Distant classes are exactly decoupled (Movable errors cannot flip Road↔Undriv pixels). This is the
Morse–Smale complex read as a codec: regions = cells, separatrices = pairwise ties, and the flip measure
factorizes over the adjacency graph (measured: 34.65 regions / 41 RAG edges per frame, FEED-af).

## 2. Two-stage reconciliation: mask-space (exact, free) → appearance-space (the paint problem)

The scorer does not consume our label map — it consumes FRAMES: `d_seg` is argmax disagreement of
`SegNet(R(F₁))` vs GT, where R = bicubic↑874 → uint8 → bilinear↓384, and **SegNet reads frame1 ONLY**
(`x[:, -1]`, modules.py:108 — frame0 is SegNet-free). So reconciliation is TWO-STAGE:

**Stage 1 — mask space (ours, exact, $0 at inflate).** `L̂ = argmax_c(φ_c + b_c)` computed deterministically
from the decoded per-class carriers. Pure geometry; rule-118 FREE compute (the rasterizers/EDT/argmax are
generic algorithms; only carrier payloads are counted).

**Stage 2 — appearance space (the evaluator-inverse PAINT problem).** Synthesize frame1 F(L̂) such that the
FROZEN SegNet through R reproduces the intended argmax:

```
min bytes(paint params)   s.t.   argmax SegNet(R(F(L̂))) = L̂     (per pixel, or d_seg(F) ≤ ε_paint)
```

Structure of the solution (all components already in-tree):
- **Region interiors:** class-typical appearance. The feasible cell is LARGE — 95% of pixels carry >2 logits
  of free margin (legal_frame_bridge, closed-spec §4). Flat palette alone is OOD for SegNet (measured
  0.0064 flat-paint floor, FEED-ba) → the paint must be class-typical TEXTURE, which lives in SegNet's null
  space (argmax depends on class statistics, not the realization) → procedural/near-0-byte (Perlin/VQ/
  exemplar), FREE generator + tiny counted seed.
- **The codim-1 annulus:** boundary treatment is the hard part and is R-dominated: the #210 gate measured
  AA/supersampled boundary rendering lifts lane recall **0.56 → 0.94 (+0.38 at ~0 rate)** — anti-aliased
  sub-pixel boundary placement is what survives the uint8 knife-edge. The paint's boundary operator = the
  AA-SDF observation render (`aa_sdf_observation_render.py`) evaluated on the DERIVED tie loci.

**MEASURED evidence that paint-induces-argmax (per class where it exists):**

| evidence | number | scope |
|---|---|---|
| Oracle-R floor (#210): IDEAL partition → paint → R → SegNet, LIVE grid @384 | d_seg **0.00091** (below sub-0.15 need ~0.00087… at the gate's paint) | COMPOSITE all-class — the strongest global proof the paint problem is solvable |
| same @192 / @camera | 0.00247 / ≈0 | render-resolution law |
| AA lane recall (#210) | 0.56 → 0.94 | Lane1 boundary through R |
| Hood static clamp (#139) | IoU 0.993–0.994; 19 flips in 25% of frame; "clamp saves ~0" (already near-perfect) | MyCar4 |
| Lane analytic band (FEED-dj / v7) | shape FN 0.00046; band composite 0.00087; dash FP 0.00396 (mask recon, range-dependent) | Lane1 mask fidelity |
| Flat-palette paint floor (FEED-ba) | 0.0064 (OOD artifact — texture is the cure) | composite, falsifies flat paint |

**UNMEASURED gaps (named honestly — the decisive $0 probe to pre-register):**
- **Per-class attribution of the 0.00091 oracle residual.** We know the composite floor; we do NOT know which
  classes contribute it. Road0/Undriv2 paint fidelity through R has never been decomposed per class — and
  Road+Undriv are 63% of the flip-prone mass (50% + 13%, CLAUDE.md measured margin distribution).
  **Probe P-A ($0, pre-registered §5):** rerun `tools/levelset_gate_discriminators_n600.py` (the #210
  tooling) with `d_seg_by_class` decomposition of ideal-paint-through-R. Output = the per-class paint floor
  vector ε_paint,c — the achievability bound each carrier composes with.
- **Textured (non-flat) class-typical paint for Road/Undriv through R** — the 0.0064 flat floor is measured,
  the textured cure is asserted from the null-space argument (FEED-ba) but not yet measured per-class.
  P-A covers this by running the gate at the live grid paint AND a textured-paint arm.

## 3. Heterogeneous carriers + rate (the per-class waterfill)

**The carrier table** (GT areas from road_anomaly_probe part_frac; flip-prone shares from the CLAUDE.md
measured margin distribution 50/19/13% for classes 0/1/2):

| class | GT area | carrier (optimal form) | byte anchor | d_seg anchor | status |
|---|---|---|---|---|---|
| MyCar4 (hood) | 0.2541 | ONE static mask, span-encode+brotli, amortized ÷600 (`hood_mask_byte_cost`) | ~0.1–0.5 KB total (DERIVED from span table; measurement tool built, number owed) | ~1e-4 (#139: 19 flips/frame in the region; IoU 0.994 MEASURED) | 2/5 built (v7 clamp) |
| Lane1 | 0.00585 | analytic ground-frame poly band + dash gate (~35 floats/frame, AR-coded); rule-118 rasterizer FREE | ~1–2 KB (FEED-dj DERIVED from float count) | 0.00087 composite MEASURED (FN 0.00046; dash FP range-dependent) | 2/5 built (v7 band) |
| Road0 | 0.2323 | **bulk-boundary field** (shared with Undriv2): ONE smooth curve network (road edge). #308 theorem: regularized GRIDS ≥ INR on dense/smooth SDFs; INR wins on contours → grid-bulk + INR-annulus hybrid | 20–50 KB (DERIVED, the decisive unknown — see waterfill) | floor bounded below by ε_paint,Road (P-A owed); flip-prone share 50% | THE build (v8 increment 1) |
| Undriv2 | 0.4952 | same bulk-boundary field (horizon + road-edge complement); horizon line measured "cheap" (DAG L208) | (included above) | flip-prone share 13%; run-1 converging 0.074 even under theft | THE build |
| Movable3 | 0.0124 | sparse per-island codes (position/scale/shape, AR-tracked) OR keep homotopy+area-Lagrange INR head | 2–6 KB (DERIVED; unmeasured) | birth PROVEN trainable (run-1: 1.0→0.0069) | keeps v7.5 machinery |

**The 5-regime waterfill (generalizing the two-regime law).** The registered
`anisotropic_basis_two_regime_allocation_v1` law assigns basis budget by whether lane is CARRIED or
OFFLOADED. The decomposition generalizes it to a per-class assignment vector: each class c is either
**offloaded** to a near-free analytic/static carrier (MyCar, Lane — their marginal d_seg per byte ≈ 0 beyond
tiny B_c) or **carried** by a trained field with its own rate-distortion curve d_c(B_c). The KKT waterfill

```
min Σ_c d_c(B_c)  s.t.  Σ_c B_c = B    ⟹    d_c'(B_c*) equal across CARRIED classes
```

then routes essentially ALL trained bytes to the bulk-boundary field (Road/Undriv) + Movable — which is
exactly where the flip mass is (63% + tail). The per-class coherence coupling
(`persistence_classes_for_basis_regime`) becomes the general rule: a class's LOSSES and its BASIS budget are
derived from the same carried/offloaded bit — per-class, compile-time asserted, never regime-blind.

**Net-bytes estimate (DERIVED; endpoints MEASURED).** Incumbent shared trunk: 114 KB INR @ d_seg 0.002017
(FEED-ad, MEASURED) / 72 KB dense @ 0.0231 (MEASURED). Decoupled stack: 0.5 + 2 + 4 + (20–50) ≈ **27–57 KB**
counted, i.e. **net-NEGATIVE bytes (−50% to −75%)** IF the bulk-boundary field lands in its derived band —
because near-free carriers replace the INR capacity that was spent representing static/analytic structure,
and the remaining trained object is ONE effective binary boundary instead of a 5-class coupled field.
Rate-term scale: 25·B/37,545,489 → 114 KB = 0.0759 vs 40 KB = 0.0266 ⇒ **≈0.049 S of headroom** at equal
d_seg. Bounding sanity: the exact-partition symbolic floor is 255 KB at d_seg=0 (FEED-af MEASURED) — the
decoupled stack does NOT contradict it because the carriers are LOSSY (per-class ε_c > 0) and amortize via
generative structure (weights/analytic forms), which is precisely how FEED-af said the floor must be beaten
("amortize below 255KB by weight-sharing"). **The 20–50 KB Road/Undriv figure is the decisive unmeasured
number** — it is what increment 1 + probe P-A exist to measure; every other row is anchored or near-free.

## 4. Pose composition (the honest coupling)

PoseNet reads BOTH frames (2-frame YUV6, 12-ch); SegNet reads frame1 only. Consequences:

- **Decomposition is pose-NEUTRAL relative to the incumbent, not pose-free.** Any witness (shared-trunk or
  decoupled) synthesizes frame1 wholesale, so the paint stage perturbs PoseNet's input either way. The
  constraint is identical in both architectures: the frame PAIR must hold the PoseNet tube. Per-class masks
  change WHAT paints frame1, not the fact that frame1 is painted.
- **The pose carrier rides ξ, orthogonal to the seg carriers.** Per MEMORY L68/L69: pose on the witness is
  OPEN + UNMEASURED (warp-real-luma d_pose 3.7–10.3 measured BAD; the ancestor 3.4e-5 is NON-TRANSFERABLE);
  the designated mover is the store-nothing ξ carrier (derive-H #257 PROVEN) consumed via FiLM conditioning.
  Nothing in the per-class decomposition touches ξ derivation or its consumption.
- **The one real coupling: frame0.** The legal-frame bridge's measured strategy — frame0 is SegNet-invisible,
  so it can carry real/derived motion for PoseNet while frame1 serves the argmax — composes cleanly with the
  decomposition (frame1's paint is the per-class composite; frame0 is untouched by it). Known hazard
  (MEASURED, legal_frame_bridge): if BOTH frames were palette-painted, PoseNet collapses — the decomposition
  must never paint frame0 with the seg palette.
- **Honest bottom line:** d_pose remains the campaign's open measured risk regardless of this design; the
  decomposition neither solves nor worsens it. Any v8 verdict must measure d_pose through the real
  byte-closed decode (recursive-review axis 9), never cite the ancestor number.

## 5. Build scoping — the incremental path from v7 (EV-ranked)

v7 already separates 2/5 classes (Lane→analytic band, MyCar→static clamp). The ladder:

1. **P-A ($0, pre-registered, FIRST):** per-class oracle-R paint-fidelity decomposition — #210 gate tooling +
   `d_seg_by_class`, ideal partition, live-grid paint + one textured-paint arm, n600. Output: ε_paint,c
   vector. DECISIVE: bounds every carrier's achievable floor; tells us if stage-2 paint (not the carriers)
   is binding for Road/Undriv. No trainer edits; read-only vs run-1.
2. **P-B ($0):** decoupled-theft falsification check — fit toy per-class fields to GT SDF targets
   (`signed_distance_fields`, no scorer, no shared params) with the run-1 birth-stack losses transplanted
   per-field; measure part_frac ratios. Prediction: ~1.0× by construction (vs 13.8×/4.6× measured on the
   shared trunk). Falsifiable: if theft appears WITHOUT shared parameters, the §1 derivation is wrong.
3. **v8 increment 1 (the SMALLEST decisive build):** split Road/Undriv into a dedicated bulk-boundary field
   — one binary Road↔Undriv separatrix carrier (grid-bulk + INR-annulus per #308), trained per-class-decoupled
   on SDF regression targets, reconciled by argmax against the v7 band/clamp + Movable head. Movable KEEPS
   the v7.5 homotopy + area-Lagrange machinery (now per-field). This single increment: (a) removes the
   shared trunk for 99% of the measured flip mass, (b) measures the decisive 20–50 KB unknown, (c) is the
   structural theft fix. Everything heavier (full 5-way split, per-class τ_c schedules, waterfill at
   byte-close) sequences AFTER its verdict.
4. **v8 increment 2:** per-class τ_c/β_c/completion annealing schedules (requires 3).
5. **v8 increment 3:** per-class KKT waterfill at byte-close + the b_c tie-calibration sweep (~0 bytes).

**Composes with v7.5 (and eventually subsumes it):** v7.5's FP-area penalty + completion events are the
counter-force WITHIN the shared trunk — needed exactly because theft is possible there. Under increment 1+,
the area-Lagrange machinery is REUSED as the per-field area constraint (§1: it becomes the exact dual of
mass conservation), the completion event becomes a per-field birth-completion event, and the FP-area penalty
becomes unnecessary for decoupled classes (theft structurally impossible) while remaining correct for any
still-shared head (Movable). **No structural conflict.** Sequencing per the crucible: v7.5 launches first
(in-place fix on the live lineage); v8 increment 1 is gated on this memo + P-A/P-B + crucible results.

## 6. Candidate canonical equations (council-FLAGGED, not registered)

Per triality discipline — derivations complete, empirical anchors OWED, so these are flagged for the council
and register only after their named anchors land:

- **`tropical_perclass_reconciliation_v1` (candidate):** partition = argmax_c(φ_c + b_c) with per-class-
  independent θ_c ⟹ (i) ∂φ_c/∂θ_{c'} = 0 (theft-impossibility, structural); (ii) d_seg factorizes over
  pairwise tie-locus displacements on the region-adjacency graph; (iii) area constraints Σa_c=1 have K−1 DOF
  and the per-class λ_c are the dual of mass conservation. Anchor owed: P-B (theft check) + increment-1
  per-class d_seg rows.
- **`perclass_rate_waterfill_v1` (candidate; generalizes `anisotropic_basis_two_regime_allocation_v1`):**
  carried/offloaded assignment vector + KKT equal-marginal-d_seg-per-byte across carried classes. Anchor
  owed: increment-3 byte-close rows.
- The FEED-roadfloor candidate law (recall-without-precision majority floor) remains council-flagged
  separately (its anchor = run-1 telemetry, diagnostic-grade).

## 7. Hostile round-1 self-review (own; findings + resolutions)

1. **"Theft-impossibility is oversold — the argmax still couples classes."** Partially fair: a MISCALIBRATED
   φ_c (globally too high) still over-claims area at reconciliation. Resolution: the coupling surviving at
   argmax is a K-vector scale/bias problem (measurable, ~0-byte b_c correction, and bounded by the SDF gauge
   + eikonal constraint), NOT a gradient pathway — the run-1 mechanism (loss-gradient theft through shared
   parameters) is genuinely closed. §1 wording now says "obsolete BY CONSTRUCTION" only for the gradient
   mechanism, with the residual argmax coupling named. P-B is the falsification probe.
2. **"Net-bytes rests on an unmeasured 20–50 KB guess."** Correct — labeled DERIVED and named THE decisive
   unknown; increment 1 exists to measure it; the claim is conditional and the memo says so. The MEASURED
   endpoints (114 KB incumbent, 255 KB exact floor, near-free carriers) bracket it honestly.
3. **"Oracle-R 0.00091 was measured at the gate's paint, not your per-class composite paint."** Correct —
   it proves the composite paint problem solvable at ONE paint; the per-class decomposition of the residual
   and the textured-paint arm are exactly what P-A pre-registers. Table row annotated.
4. **"Pose section could hide behind 'orthogonal'."** Fixed: stated pose-NEUTRAL-not-pose-free, named the
   both-frames-painted collapse hazard, and repeated the OPEN+UNMEASURED status with the ancestor-number
   prohibition.
5. **"Does this contradict FEED-af (explicit spline codec NOT the lever)?"** No — FEED-af killed explicit
   symbolic curves for the LOSSLESS d_seg=0 store (context-arith already optimal there). The carriers here
   are LOSSY, generative, per-class amortizers — precisely FEED-af's prescribed route ("beat 255 KB by
   weight-sharing"). §3 cites the floor as the bounding sanity check.
6. **"v7.5 memo not landed — composition claims unverifiable."** Flagged in STORES CONSULTED; claims are
   against the FEED-roadfloor spec of v7.5; reconcile-on-landing noted.

**verdict_scope (on every negative herein):** the flat-paint 0.0064 floor is FORMULATION-level (flat palette
only; textured paint unjudged pending P-A); the shared-trunk theft finding is FORMULATION-level (this
birth-stack composition on a shared trunk; island-birth as a paradigm is CONFIRMED); no family/paradigm
kills are made or implied by this design.

## FINAL STATE
Design memo only; no trainer edits; no launches; run-1 untouched; pointer **0.19110 UNMOVED** (this unit is
means — the pointer moves only through a byte-closed `upstream/evaluate.py` n600 exact row).

---

## Observability surface

*(OBSERVABILITY-ADDENDUM 2026-08-25 — APPEND-ONLY per Catalog #110/#113. This
section is an INDEX into this memo's own content per Catalog #305's 6 facets;
it adds no new claim. Facets with no counterpart in this memo say so plainly.)*

1. **Per-layer inspection** — §1 "The reconciliation operator" and §2 "Two-stage reconciliation: mask-space (exact, free) -> appearance-space (the paint problem)" separate the two stages, each inspectable on its own; per-class carriers are separable by construction.
2. **Per-signal decomposition** — §3 "Heterogeneous carriers + rate (the per-class waterfill)" is a per-class rate decomposition; §4 "Pose composition (the honest coupling)" separates the pose coupling from the seg term.
3. **Run-to-run diff** — because carriers are per class, an arm that changes one class's carrier differs from its base on that class only, which is what makes the §5 EV-ranked increments attributable.
4. **Post-hoc query** — named surfaces are `lever_b_levelset_generator.py`, `aa_sdf_observation_render.py`, `hood_static_component.py`, `legal_frame_bridge.py`; the authority is `upstream/evaluate.py`.
5. **Cite-chain** — the "STORES CONSULTED" section is the recall chain; §6 "Candidate canonical equations (council-FLAGGED, not registered)" keeps the equation debt visible rather than silently registered.
6. **Counterfactual hooks** — §7 "Hostile round-1 self-review (own; findings + resolutions)" is the adversarial counterfactual pass; §5 "Build scoping — the incremental path from v7 (EV-ranked)" orders the ablations.
