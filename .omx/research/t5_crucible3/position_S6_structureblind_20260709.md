# T5 CRUCIBLE-3 — POSITION S6 (STRUCTURE-BLIND) — 2026-07-09

**Seat charter:** derive the v8 carrier DECOMPOSITION + stage skeleton BLINDED — from the tropical-argmax /
Laguerre energy + the MEASURED rate ledger + the region-adjacency graph + the scorer-rule/annulus facts
ALONE — then compare the blind-derived SHAPE to the incumbent's design class to expose aspirational-label
residue (the elementwise-audits-launder-structural-cargocult LAW, operator 3rd recurrence).

**STORES CONSULTED (read):** `docs/operating_manual_craft_handoff.md` (craft bar) · `.omx/research/t5_crucible3/
CONVENING_20260709.md` (charter/junction/bindings ONLY) · `.omx/research/t5_crucible3/ORCHESTRATION_LEDGER.md`
(S6 blind constraint, seats, standing checks) · `.omx/research/t5_crucible3/DELTA_GROUNDING_20260709.md` (the
MEASUREMENTS: §A state-change · §B rate ledger · §C parsimony/Laguerre/adjacency-hub · §E residual-coder gap ·
§F reconciliation/scorer-rule/annulus · §H measured negatives N-1..N-4 · §I pose banked/store-nothing · §J
baseline v7.5.2 · §SETTLED · §OPEN QUESTIONS) · CLAUDE.md §SegNet/PoseNet architectures + §SETTLED class order.

**STORES FORBIDDEN (not read, per the binding blind constraint):** `SPEC_v8_perclass_decomposition_20260708.md`
§1 (tropical-argmax architecture statement), §2 (carrier table), §6 (increment-1 plan) · `v8_increment1_design_
draft_20260709.md` §1–2 · ALL sibling positions S1–S5 (crucible-3) · ALL crucible-2 position docs.

**HONEST DISCLOSURE (blindness is partial by construction):** DELTA_GROUNDING is an ALLOWED read, and its §D
(per-carrier state table) + §K (8 build modules) DO encode the incumbent's answer (a 5-row carrier table; the
`road_undriv_bulk_field` module). I read them. My derivation below proceeds from §B (the EDGE-keyed rate ledger)
+ §C (Laguerre/parsimony/Road-hub) + §F (annulus/scorer) + the energy — NOT from §D's carrier COUNT or §K's
module NAMES. The value of the blind pass is that I reach the decomposition SHAPE from the scored energy on its
own logic; §D/§K then become the object I AUDIT, not the premise I inherit. Where I cite §D it is as a
measurement (bytes, connectivity), never as the design.

Pointer contest-CPU **0.19110 UNMOVED**; remaining gap to sub-0.15 target = **0.0411 S**. Everything here is
[macOS advisory · research-signal · NON-PROMOTABLE] MEANS. `[no-triality]` (apparatus/position doc).

---

## 0. THE METHOD — what the score actually indexes

d_seg = per-pixel argmax-disagreement rate on **frame1 only** (SegNet reads `x[:,-1]`; §F, CLAUDE.md). The
argmax is `P(x)=argmax_c(φ_c(x)+b_c)` = an additively-weighted Voronoi = **Laguerre / power diagram** (§C, #284:
Laguerre ≡ tropical ≡ curvelet ≡ se(3), one object). Two MEASURED facts fix what the representation must be
GOOD at:

- **The annulus law (§F #333):** ~97% of d_seg mass lives in a ~4.7%-area boundary annulus. The scored object
  is the **separatrix** (the inter-class boundary set), NOT the interiors.
- **Interiors near-free at the oracle bound (§F, P-A bf1ee1fa8):** Road 0.17% / Undriv 0.03% within-class
  through R. verdict_scope: CONDITIONAL on P-C (generated-paint floor UNMEASURED).

So the representation's whole job is to place the separatrix. The energy of a boundary point x on the interface
between classes c and c′ is set by `φ_c(x)+b_c = φ_{c′}(x)+b_{c′}`, i.e. by the **DIFFERENCE field**
`(φ_c − φ_{c′}) + (b_c − b_{c′})`. **A boundary is a function of a class-PAIR, not of either class alone.**
This is the first-principles fork: the natural unit of a separatrix representation is the **edge** {c,c′} of the
adjacency graph, not the class. This derivation is entirely energy-driven; I have not consulted the incumbent's
choice.

---

## 1. (a) THE DERIVED DECOMPOSITION — how many carriers, keyed to what

### 1.1 Enumerate the REAL adjacency edges (from §B, the edge-keyed rate ledger)

5 nodes {Road0, Lane1, Undriv2, Movable3, MyCar4} ⇒ C(5,2)=10 possible edges. §B measures which carry mass:

| edge | bitmap S (§B) | significant? | physical meaning |
|---|---|---|---|
| Road–Lane | 0.204 | **YES (dominant)** | lane markings embedded in the road plane |
| Road–Undriv | 0.047 | **YES** | the horizon (road plane meets sky/background) |
| Road–Movable | ⊂ 0.0532 | **YES** | cars standing on the road |
| Undriv–Movable | ⊂ 0.0532 | **YES** | cars silhouetted against the sky above horizon |
| Road–MyCar | 0.028 | **YES** | ego hood meets road at frame bottom |
| Lane–{Undriv,Movable,MyCar} | ⊂ 0.007 | no | lanes touch only the road (coder-tiny already) |
| Undriv–MyCar, Movable–MyCar | ≈0 | no | hood touches only road |

**MEASURED result: 5 significant edges** {Road-Lane, Road-Undriv, Road-Movable, Undriv-Movable, Road-MyCar};
the other 5 possible edges are negligible (Lane is single-neighbor-to-Road; MyCar is single-neighbor-to-Road).

**Road is the HUB** — it is one endpoint of 4 of the 5 significant edges (§C "Road = adjacency hub"). The ONLY
significant non-Road edge is Undriv–Movable (a car poking above the horizon). DERIVED.

### 1.2 Collapse edges → distinct GENERATORS (the parsimony law, §C)

§C's law: store each carrier's **parsimonious GENERATOR** (few coefficients), never the boundary bitmap and
never the dense medial axis (N-3: dense medial ≈ bitmap, 1.09×, NO WIN). The 5 edges do NOT need 5 generators,
because generators are shared:

1. **Movable's TWO edges share ONE generator.** A car is one region O with one closed silhouette ∂O. That
   silhouette is partitioned into a Road-arc (below horizon) and an Undriv-arc (above) by WHERE the car sits —
   but the CURVE is one object. Store the object sites/bboxes once; both edges fall out of the horizon
   crossing. → **1 Movable generator covers Road-Movable + Undriv-Movable** (§B treats them as one 0.0532 row
   with one carrier — a measurement that VINDICATES this collapse, not a premise I assumed).

2. **Road & Undriv need NO positive interior generator — they are the two "seas."** Undriv is
   **single-connected** (§D lift-verify: 0.0% of frames ≥2 components) = one sky/background blob above the
   horizon. Road is the plane below the horizon. The Road–Undriv boundary is ONE curve (the horizon). Once you
   have the horizon curve + the feature cutouts (lanes, cars, hood), **Road = "everything below horizon not
   claimed by a feature" and Undriv = "everything above horizon not claimed by a car."** Neither is an object;
   both are COMPLEMENT regions. Interiors are near-free (§F P-A). → Road and Undriv contribute exactly ONE
   generator between them: **the horizon curve.**

3. **Road's multi-component-ness (37.2% of frames, ≤3 blobs; §D) is a CONSEQUENCE, not a generator.** The road
   plane is cut into disconnected pieces BY the lane bands / car silhouettes / hood that overlay it. That
   structure FALLS OUT of the feature generators at composition time; it does not demand its own field.

### 1.3 THE BLIND COUNT

**4 boundary generators + 1 tie-bias vector**, edge-keyed, Road-as-implicit-complement:

| # | generator | edge(s) it resolves | frame-invariant form | MEASURED rate (§B/§C/§D) |
|---|---|---|---|---|
| G1 | **Horizon curve** | Road-Undriv | deg-3 poly + ξ vertical intercept (599/600 = ego-pitch shift of frame0's poly; cubic/quad FROZEN) | dominant **0.0032** (14.6×, 4 coeffs) |
| G2 | **Lane band** | Road-Lane | openpilot analytic centerline poly, GROUND-frame, NO ξ transport (N-2 LAW) | dominant **0.0275** (72.5% cover) |
| G3 | **Movable silhouettes** | Road-Movable **+** Undriv-Movable (2 edges, 1 gen) | sparse object sites/bboxes 3.0/frame + Hungarian ξ-track | dominant **0.00344** (70% cover) |
| G4 | **Hood** | Road-MyCar | static frame0 silhouette (IoU 0.994) + rigid ξ shift | complete **0.0202** (static, no residual) |
| B | **b_c tie-bias** | ALL (global argmax tie-break) | 5-vector, flip-weighted, closed-form, out-of-loop | ~0 bytes |

**Road & Undriv carry NO positive generator** — they are the complement regions the horizon + features cut out.
Lane-tail edges (0.007) are already coder-tiny; fold into G2's carrier or leave as a fixed cheap sidecar.

This is my blind-derived SHAPE: **4 generator families keyed to boundary structure + a global tie-bias, with
Road as the hub-complement and Movable's two edges sharing one silhouette.** It is FRAME-INVARIANT (the
generators are parametric families; the number of boundary arcs they emit varies per frame).

---

## 2. (b) THE RESIDUAL CHART — where the uncovered bytes go, rate-optimally

§B: generators cover 70–83% of each edge's separatrix; the residual (17–30% uncovered boundary px) is the
NAMED ENEMY (§E, 0.079 S gap between dominant 0.061 and complete 0.140). Relative-significance: **0.079 S is
1.9× the ENTIRE remaining gap to sub-0.15 (0.0411).** Closing it is not polish — it is larger than the whole
distance to target. Even a half-close (~0.04) ≈ the entire gap. This is THE rate lever, sized honestly.

The residual chart falls out of the same geometry, two levers, both DERIVED here independently (they match §E's
two headroom levers — a vindication that the cure is structural, not a lucky guess):

- **Residual chart = curve-relative offset coding (DERIVED; §E lever 2, "NOT built").** By the annulus law, every
  residual px is a boundary px the generator missed — a THIN set hugging the generator curve. The generator
  supplies a 1-D arc-length parametrization; code each residual px as **(arc-length s along the owning
  generator, signed normal offset n)**. n is a small signed integer (residual px sit within a few px of the
  generator, since the generator already captured the dominant boundary) ⇒ ~2–3 bit entropy after zigzag; s is
  delta-compressible along the curve (residual px cluster). This strictly dominates the generic absolute-coord
  flat-index coder (§E's current 0.4–0.6 B/px, which pays ~full image entropy on 2 coords) because the
  generator is the PRIOR and the residual is coded as its innovation. Byte outcome: CONJECTURED headroom toward
  0.061 — OWED a real bit-exact roundtrip (S2/S4 + P4 recess; I do NOT assert a measured number).

- **De-sharing partition (DERIVED; §E lever 1, "uncounted").** Because carriers are edge-keyed, every uncovered
  separatrix px belongs to exactly ONE true edge (the class-pair actually meeting there). Attribute each
  residual px to the carrier whose generator owns that pair — a PARTITION, so no px is coded twice. The MEASURED
  instance: the horizon's "secondary arcs" ARE cars breaking the horizon (§E, 1.6–2.0 crossings/row) = Movable
  px already carried by G3. My blind derivation PREDICTS this double-count before reading §E's statement of it:
  a car straddling the horizon is claimed by both G1 (as Undriv/Road split) and G3 (as silhouette); the px
  belongs to G3's edge, so subtract it from G1's residual for FREE.

**Residual verdict:** the rate-optimal residual is per-generator curve-relative (s, n) coding under a global
de-sharing partition. Both are structural consequences of edge-centrism + generator-normal geometry, NOT new
mechanisms. **Do NOT propose lossy residual trades on the Road-Lane edge** — N-4 MEASURED the RD knee at the
lossless point (every quantization step RAISES S_total; distortion pinned by a ~0.52 band-geometry floor).
verdict_scope of N-4: FORMULATION (filled-band-vs-thin-lines metric artifact) — but binding for increment-1:
lossy is dominated there, so the gap MUST be closed by de-share + curve-relative, not by quantization.

---

## 3. (c) POSE COMPOSITION — free w.r.t. d_seg, shared ξ, gated engagement

§F/§I P-2, structural proof: **∂d_seg/∂ξ ≡ 0 EXACTLY** — SegNet reads only frame1; ξ shapes only the
SegNet-blind frame0. So pose is ORTHOGONAL to the d_seg carriers. DERIVED composition:

- **No new carrier.** Pose is the **se(3) screw ξ trajectory** — and ξ is ALREADY consumed by G1 (horizon
  ego-pitch intercept), G3 (Hungarian ξ-track), and G4 (rigid hood shift). ONE ξ trajectory serves both the
  per-carrier temporal advection (d_seg coherence) AND pose (dual-use screw, memory L10/L68). Pose is a READ of
  the shared ξ on frame0, not a stored field.
- **Store-nothing MANDATE (§I P-7, operator binding 5):** the fresh arm runs the `generated` path ≈ 1 KB. Any
  keyframe lineage MUST charge the counted-keyframe rate (697,941 B) — lineage-tagged. At the fresh arm,
  store-nothing is restored.
- **Engagement gate (operator binding 2).** WHY gate, given orthogonality? Because the SHARED ξ is fit to serve
  d_seg temporal-boundary coherence FIRST; firing terminal pose-finish before the per-carrier boundaries are
  temporally stable fits ξ to a jittering partition ⇒ the screw is noise. So the gate reads **per-carrier
  temporal-boundary stability** (frame-to-frame boundary jitter of G1–G4 below a derived threshold), a
  CONDITIONING-GATED EVENT (never an epoch). Provenance of the threshold: DERIVED from the carrier jitter
  statistic (owed at increment-1; a VALUE, config-conditional, not a TBD). Never-reached fallback = **ship
  BANKED R1 dxi** (d_pose 0.001610 → contribution 0.127; ξ_eff 7,195 B; §I P-1, [macOS-CPU advisory]
  NON-PROMOTABLE). P-5 HONEST FLAG stands: fresh-arm terminal-finish efficacy is UNVALIDATED; the fallback is
  the safe default.

---

## 4. (d) RECONCILIATION — overlapping carriers → ONE argmax

The carriers make OVERLAPPING claims (a car straddling the horizon is claimed by both G1's "Undriv-above" and
G3's silhouette). Reconciliation = tropical argmax `P(x)=argmax_c(φ_c(x)+b_c)`, which I derive as a **layered
override with a precedence that falls out of the generator hierarchy**:

1. **Base layer:** the horizon (G1) splits top(Undriv) / bottom(Road).
2. **Feature overrides:** G2 lane band overrides Road→Lane; G3 silhouette overrides Road/Undriv→Movable; G4
   hood overrides Road→MyCar at the bottom. Each feature owns a DISJOINT arc-set (post de-sharing), so claims
   do not fight except at genuine triple-points (3 classes meeting — a car on a lane at the horizon), which are
   near-measure-zero and resolved by b_c.
3. **b_c is the ONLY global coupling — so it MUST be closed-form, out-of-loop (risk-2 guard).** Since the
   argmax composition is the sole place classes interact, calibrating b_c against d_seg jointly re-couples ALL
   classes through the shared bias = theft re-entering. So b_c is SOLVED from flip statistics, not trained.
   **DERIVED requirement: b_c is FLIP-WEIGHTED, not area-matched** — the score is flip-rate (annulus error), so
   the tie-bias minimizes flip mass, not area error. This reaches N-1's open reformulation BLIND: N-1 MEASURED
   that area-mass-matching head offsets HURT (no_offset 0.00272 < menon 0.00293 < ot_newton 0.00487, both arms
   worse; solver EXACT). verdict_scope of N-1: FORMULATION (mass-matching to raw GT frequencies) — the
   flip-weighted target is the open, un-refuted reformulation my derivation independently demands.
4. **Fixed point (merge→diff→correct, §F):** MERGE the layered argmax → paint frame1; DIFF frozen SegNet on
   R(composite) vs intended partition (frame1 ONLY, since frame0 is SegNet-free); CORRECT via the residual
   sidecar (§2 curve-relative), channel-routed. **Chroma-first is a warm-start ONLY for the chroma-separable
   Road/Undriv paint** (grey/green/blue). It is NOT valid for Road-Lane (luma-separable, bright lines on dark
   road, 41% of Road's flips) — and my decomposition is SAFE there BY CONSTRUCTION because Lane is the separate
   analytic carrier G2, never paint-repaired. The REAL guard is step-4 MEASURE-pose-on-composites + Dykstra
   alternating projection onto (argmax-cell ∩ pose-tube), never the routing heuristic (§F review-D).

---

## 5. THE STRUCTURE-MISMATCH TEST — blind-derived SHAPE vs the "5-per-class-field" design class

I do not know the incumbent's internal structure (blind to §1/§2/§6). I reason about the DESIGN CLASS the
charter names as the residue target: a decomposition presented as **"5 per-class fields"** and/or labelled
**"one field per adjacency EDGE (41 edges)"**. Three findings:

### 5.1 The "41 edges" label cannot be class-pair edges — it is instance-scoped arc count
5 classes admit at most C(5,2)=10 class-pairs; §B measures **5 significant**. **41 cannot be pairwise adjacency
edges.** It is most consistent with the number of connected boundary-ARC SEGMENTS in a typical frame's
separatrix planar subdivision (road + lanes + horizon + few cars + hood ⇒ dozens of arcs) — which is
FRAME-DEPENDENT and INSTANCE-SCOPED (toy-isolation risk: a per-arc field count varies per frame and does not
generalize). My decomposition keys carriers to the **4 frame-invariant generator FAMILIES**, not to the ~41
per-frame arcs. **Reconciliation of 41-vs-5: they are different objects.** 41 ≈ per-frame arc-instances; 5 (or
my 4) = generator families. Storing "one field per arc" is the wrong (instance-scoped) axis; the correct axis
is the generator family. This is a numbered-cross-ref-drift catch: the "41" is aspirational/instance-level and
must never become a carrier count. verdict_scope: FORMULATION (the per-arc field framing), not a paradigm kill.

### 5.2 For 4 of 5 classes, "per-class field" and "edge-carrier" COINCIDE — the label tension is harmless there
Lane, MyCar are single-neighbor-to-Road ⇒ their "class field" IS their "Road-edge" (one boundary). Movable is
two edges but one silhouette generator. So for {Lane, Movable, MyCar} the per-class vs edge-centric distinction
**dissolves** — a per-class field there is already de-facto an edge-carrier. The class-naive-vs-edge-centric
tension is NOT load-bearing for 3 of the 4 non-hub generators.

### 5.3 The tension is REAL and LOAD-BEARING at exactly ONE place: Road+Undriv — and my blind shape says the honest generator is the HORIZON CURVE, not a bulk field over Road∪Undriv
This is the sharp catch. §D (measurement) shows the Road+Undriv carrier as **"ONE edge-centric bulk-boundary
FIELD"** composing a signed-SDF `lever_b_levelset_generator` over the Road/Undriv region. My blind derivation
(§1.2 item 2) says Road and Undriv are the two COMPLEMENT seas separated by ONE curve; the load-bearing
generator is the **horizon poly** (G1), and Road's multi-component structure is a CONSEQUENCE of the OTHER
generators cutting the plane — not an intrinsic field. Therefore:

- A **signed SDF field spanning Road∪Undriv** is at risk of RE-ENCODING the very boundaries (lane bands, car
  silhouettes, hood) that G2/G3/G4 already carry — the double-count that edge-centrism was supposed to KILL
  (risk-1). It re-enters one abstraction layer up: the "bulk field" wrapper pays for the multi-blob Road
  structure, which is authored by the features. MEASURED corroboration: §B's Road-Undriv COMPLETE = 0.0221
  (+0.0189 residual, 25.7% uncovered), and §E MEASURED that residual's "secondary arcs" ARE Movable px — i.e.
  the Road/Undriv carrier is ALREADY double-counting Movable's boundary today. My shape predicts and removes it
  (de-share to G3, §2).
- N-3 sharpens the warning: a DENSE field representation of the Road/Undriv boundary ≈ bitmap (1.09×, NO WIN).
  A bulk SDF only wins if compressed to few coefficients — but those few coefficients ARE the horizon poly +
  the object boundaries, which G1/G3 already hold. So the bulk field is either redundant (re-encodes existing
  generators) or the dense-medial no-win. **verdict_scope: FORMULATION** — the *bulk-field WRAPPER* is
  over-built; its `road_horizon_component` SUB-generator IS the lever, and it already exists on-disk (§K). I am
  NOT killing the Road/Undriv carrier; I am asserting its minimal honest form is horizon-curve + de-shared
  Movable straddles + the two seas as complement, and any capacity spent on a Road/Undriv "bulk interior field"
  must justify itself against interiors-near-free (§F P-A) and the double-count it risks.

**Net structure-mismatch implication:** a design that stores **5 per-class fields as the primary form** would
(i) waste a φ_Road field (Road is the hub-complement — no positive generator), and (ii) require an explicit
de-sharing rule to avoid ~2× rate (every boundary paid in both incident fields). The mismatch localizes
ENTIRELY to the Road/Undriv pair; there the honest generator is a CURVE, not a field. **If the incumbent's
Road+Undriv carrier is a full bulk SDF, that is the aspirational-label residue to strip; if it is already
horizon-curve + de-share in disguise, my derivation VINDICATES it and the "bulk field" is just a naming
overreach.** Only reading §1/§2 (post-blind, at synthesis) resolves which — that is the designed comparison.

---

## 6. KNOB TABLE — every value on the provenance ladder (TBD FORBIDDEN)

| knob | VALUE | provenance |
|---|---|---|
| # significant adjacency edges | 5 (Road-Lane, Road-Undriv, Road-Movable, Undriv-Movable, Road-MyCar) | DERIVED from §B (S > ~0.007) |
| # distinct boundary generators | **4** (G1 horizon, G2 lane, G3 movable[2 edges], G4 hood) + b_c | DERIVED (edge→generator collapse, §1.2) |
| Road / Undriv representation | implicit COMPLEMENT (no positive interior generator) | DERIVED: Road=hub (§C) + Undriv single-connected (§D) + interiors-near-free (§F P-A) |
| G1 horizon | deg-3 poly + ξ vertical intercept | MEASURED §C (14.6×, 4 coeffs, 0.0032 S; cubic/quad FROZEN 1e-7/6e-5) |
| G2 lane | openpilot analytic band, GROUND-frame, NO ξ transport | MEASURED §B (0.0275) + N-2 LAW ξ-NO-GO (registered) |
| G3 movable | sparse sites/bboxes 3.0/frame + Hungarian ξ-track | MEASURED §B (0.00344 dominant, 70% cover, bit-exact) |
| G4 hood | static frame0 silhouette + rigid ξ | MEASURED §D (IoU 0.994 #139, 0.0202 complete) |
| b_c tie-bias | FLIP-WEIGHTED, closed-form, out-of-scorer-loop | DERIVED (score=flip-rate) + N-1 (area-match HURTS; flip-weight open) |
| residual chart | per-generator curve-relative (arc-length s, signed normal offset n) | DERIVED §2 (annulus + §E lever 2); byte outcome CONJECTURED/owed |
| de-sharing | global partition: residual px → its true-edge carrier | DERIVED §2 (§E lever 1; horizon↔Movable double-count MEASURED) |
| lossy residual on Road-Lane | FORBIDDEN | MEASURED N-4 (RD knee at lossless; every step raises S) |
| pose | shared ξ trajectory, store-nothing ~1 KB, engagement-gated, fallback banked R1 (0.127/7.2 KB) | MEASURED §I (banked) + DERIVED gate (carrier temporal stability) |
| reconciliation | layered tropical-argmax override (horizon base → feature overrides), closed-form b_c, chroma-first ONLY for chroma-separable Road/Undriv paint, guard = MEASURE-pose-on-composites + Dykstra | DERIVED §4 + §F |

**Grounding gaps flagged (do not guess):** the curve-relative residual byte number is CONJECTURED (coder
UNBUILT, §E) — owed a bit-exact roundtrip in P4 recess. The counted-seed-floor (video-derived texture stats for
adequate paint) is MEASURED-OWED by P-C (heavy governed n600-through-R), UNMEASURED as of DELTA §E/§F — a
BLOCKING precondition for the paint stage, not resolvable blind. The pose-gate threshold value is
config-conditional-DERIVED (owed the carrier-jitter statistic at increment-1).

---

## 7. THE STAGE SKELETON (blind-derived; matches SPEC_v8's risk-2 STAGED intent without reading §6)

Theft (risk-2) re-couples classes through paint→R→SegNet gradients. My decomposition kills the CONSTRUCTION-time
coupling (∂φ_c/∂θ_{c′}=0 across generators) but the paint composite can re-open it. Staged skeleton:

- **Stage A — generators vs EXACT SDF/analytic targets, NO cross-class gradient.** Each of G1–G4 fit to its own
  boundary target (argmax(sdf)==labels per class-pair); b_c solved closed-form flip-weighted OUTSIDE any
  scorer-gradient loop. Per-carrier byte-closed + bit-exact-at-decode + resumable BEFORE composition
  (vehicle-OS manifest-per-carrier). FORBIDDEN: end-to-end paint→R→SegNet (re-opens theft).
- **Stage B — paint solved SEPARATELY vs the frozen scorer**, GATED behind P-C (flat/procedural-fill floor +
  counted-seed-floor MUST report first — §F risk-3 BLOCKING precondition). Merge→diff→correct to Dykstra fixed
  point; residual → curve-relative sidecar.
- **Stage C — terminal pose-finish, engagement-gated** on per-carrier temporal-boundary stability; fallback
  ship banked R1.

Every event (per-carrier fit convergence, paint fixed-point, pose-gate) ships BACKTEST + LIVE-trainer INJECTION
test (fires-when-should + silent-when-shouldn't, NOT a unit stub — the F-1 launch-path≠config-tests lesson) +
FAIL-SAFE cap (SYNTHESIS HARD REQ B). The increment-1 config compiles against the REAL §K modules; the
Road+Undriv carrier lands as a DSL `Lever` factory with `lever_registry.completeness()` = 0 unmapped.

---

## 8. THE 3 SHARPEST COMMITMENTS (for the synthesis)

1. **The carrier axis is the GENERATOR FAMILY (4), not the class (5) and not the per-frame arc (~41).** Road is
   the hub-COMPLEMENT and carries no positive generator; Movable's two edges share one silhouette. "5 per-class
   fields" over-counts Road and needs an explicit de-share to avoid ~2× rate; "41 edges" is an instance-scoped
   arc count that must never become a carrier count. The class-vs-edge tension is HARMLESS for
   {Lane,Movable,MyCar} and LOAD-BEARING only at Road/Undriv.

2. **At Road/Undriv, the honest generator is the horizon CURVE + de-shared Movable straddles + two complement
   seas — NOT a bulk SDF field over Road∪Undriv.** A bulk field risks re-encoding G2/G3/G4 boundaries
   (double-count risk-1 one layer up) or the dense-medial no-win (N-3). MEASURED corroboration: §B's
   Road-Undriv 0.0189 residual IS Movable px (§E) — the carrier double-counts TODAY; de-share removes it free.
   verdict_scope: FORMULATION (the bulk-field wrapper), not a kill — `road_horizon_component` is the lever.

3. **The residual coder is THE game, sized honestly: 0.079 S = 1.9× the entire 0.0411 gap to sub-0.15.** Its
   rate-optimal chart is per-generator curve-relative (s, n) under a global de-sharing partition — both DERIVED
   independently and matching §E's two un-built headroom levers. Lossy trades on Road-Lane are dominated (N-4);
   the gap MUST be closed by de-share + curve-relative, and both are structural consequences of edge-centrism,
   so they are BUILDABLE, not aspirational. The byte outcome is CONJECTURED/OWED a bit-exact roundtrip — the
   #1 P4-recess measurement.

---

## Adversarial self-check (operating-manual: attack your own conclusion)

- **Did I launder the incumbent's shape?** I saw §D's 5-row table and §K's modules (allowed reads). Mitigation:
  every structural claim traces to §B (edge-keyed ledger) / §C (Laguerre/hub) / §F (annulus/scorer) / the
  energy, cited inline; §D/§K appear only as MEASUREMENTS (bytes, connectivity, module existence), never as the
  design premise. My 4-generator count DIVERGES from §D's 5-row framing (I drop Road as complement; I demand
  horizon-CURVE not bulk-FIELD) — divergence is evidence the derivation ran independently, not a launder.
- **Is "4 generators" a real reduction or a naming trick vs §D's 5 rows?** §D's 5 rows are {hood, lane, movable,
  Road+Undriv, b_c}. Mine are {horizon, lane, movable, hood, b_c} — same four objects PLUS the structural claim
  that the Road+Undriv row's honest generator is the horizon curve (not a bulk field) and Road needs no field.
  The reduction is 5-fields→4-generators (Road drops), and the SHAPE claim (curve not field) is the load-bearing
  part, not the count.
- **Could Road genuinely need its own field?** If interiors were NOT near-free, yes. §F P-A measures them
  near-free but CONDITIONAL on P-C (generated-paint floor UNMEASURED). So my "Road = complement, no generator"
  is CONDITIONAL on P-C confirming generated interiors stay near-free. Flagged: if P-C shows a large counted
  seed-floor, Road interior texture (not boundary) becomes a real cost — but that is a PAINT cost (Stage B), not
  a boundary GENERATOR, so it does not resurrect a Road boundary field.
- **verdict_scope discipline:** every negative scoped FORMULATION (N-1 area-match, N-3 dense-medial, N-4 lane
  waterfill, the bulk-field wrapper) — none is a paradigm/family kill; each names its open reformulation.
- **relative-significance discipline:** the one magnitude claim (0.079 residual gap) is sized against the 0.0411
  remaining-gap — 1.9×, i.e. the residual coder is larger than the whole distance to target, so it is THE lever,
  not a small delta to defer.
- **What I did NOT resolve (honest boundary):** whether the incumbent's Road/Undriv carrier IS a bulk field or
  is already horizon-curve+de-share — I am blind to §1/§2; the synthesis resolves it. My job was to state the
  energy-honest shape so the comparison has a fixed reference; I have.
