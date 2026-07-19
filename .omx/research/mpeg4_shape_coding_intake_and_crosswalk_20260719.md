# MPEG-4 Part 2 Binary Shape Coding — deep intake + crosswalk to our setting — 2026-07-19

**Agent:** MPEG-4 SHAPE-CODING DEEP INTAKE (isolated worktree, WebSearch/WebFetch). NO launches, no paid
dispatch. Advisory only; pointer **0.19108 UNMOVED** — everything here is a MEANS, nothing is a score.

**Mandate (operator elevation 2026-07-19, "that mp4 work sounds very interesting and promising as well"):**
deep-dive adoption #2 of `.omx/research/generator_description_online_survey_20260719.md` (the ξ-keyed
temporal-delta / MPEG-4 INTER-CAE import) and design its crosswalk to our stack. This is the **research/design
half of task #574**; the build/measure half stays gated on the PDW1 first in-box point (Aurenhammer
power-diagram-detection LP). Coordinator mid-task addendum folded in: a **third temporal arm** —
kinetic regular-triangulation event grammar (PDW survey §PDW P1) — is priced against the two CAE arms in §5.

**Our setting (the crosswalk target).** 600 pairs / 1200 frames of a 5-class road-scene partition
(0=Road, 1=Lane, 2=Undrivable, 3=Movable, 4=MyCar) at 512×384. The ego-motion screw **ξ per pair is ALREADY
stored in the archive for pose** → temporal-predictor bytes derived from ξ are **FREE** (rule 118: the warp
is generic decode-time code, not counted). Decode compute is **FREE** (30-min budget; only archive payload is
counted). Receiver is our **frozen-fp32 C1 spine** (exact integer-lattice). Budget box TOTAL archive
**≤ ~286,682 B (~477.8 B/pair)** with **d_seg ≤ ~3.39e-4**. Error bytes concentrate on **Road–Lane edges
(61%)**. Current open RD axis (per `seg_and_pose_solved_exact_lattice_realization_one_rd_axis`): **bytes
(generator + band-slack)** — the exact-residual axis is RATE-DEAD (~336 KB/pair floor). **So a temporal delta
that shrinks the generator description attacks the ONE open axis directly.** Incumbent to beat: margin-preserving
**power-diagram packet ≤138 B/pair (INTRA, per-pair)** + ~8-dim lane polynomials.

---

## §1 — The exact algorithm (not the abstract)

MPEG-4 Part 2 (ISO/IEC 14496-2) codes a per-object **binary alpha plane** (opaque=255 / transparent=0). The
plane is tiled into **Binary Alpha Blocks (BABs) of 16×16** (macroblock-aligned to the texture MBs). Each BAB
is coded independently in one of 7 modes; the boundary/interior pixels are entropy-coded by **Context-based
Arithmetic Encoding (CAE)** with a fixed causal template indexing a **precomputed static probability table**
(no adaptation, no side info — the tables are baked into the standard, hence FREE decoder for us).

### 1.1 Intra CAE — 10-pixel causal template → 1024-entry table

The pixel being coded (`?`) is predicted from a **10-pixel causal template** (labels C0…C9 in the standard),
drawn from the two rows above and the current row to the left — all already-decoded within the frame:

```
        row y-2:      C9  C8  C7
        row y-1:  C6  C5  C4  C3  C2
        row y:        C1  C0  ?
```

(3 pixels two rows up + 5 pixels one row up + 2 left-neighbors = **10 causal pixels**.) The 10 bits form a
context index `c ∈ [0, 1023]`; a **static 1024-entry probability table** (`intra_prob[1024]`, published in the
standard) gives `P(? = 1 | context)`, fed to a binary arithmetic coder. Off-plane template pixels use a
border-extension rule. **Cost per pixel ≈ H(p_context)** — small where the 10-neighborhood is unanimous
(deep interior, all-0 or all-255 → near-free), large only at the boundary.

### 1.2 Inter CAE — 9-pixel template (4 current + 5 motion-compensated reference) → 512-entry table

INTER mode predicts from BOTH the current frame's causal neighbors AND the **motion-compensated previous alpha
plane** (shifted by the shape MV). The template is **9 pixels**: 4 from the current BAB (C0–C3, causal) + 5
from the aligned reference (C4–C8: the co-located pixel C6 plus its 4-neighbors — up/down/left/right — which
are all available because the reference frame is fully decoded, so the reference half is NON-causal / a full
plus-stencil):

```
   current frame (causal):        reference frame (motion-compensated, non-causal plus-stencil):
        row y-1:  C3  C2                     C5
        row y:    C1  C0  ?              C4  C6  C7
                                             C8
```

(4 current + 5 reference = **9 pixels** → context index `c ∈ [0, 511]`, **static 512-entry table**
`inter_prob[512]`.) Because the reference term collapses the entropy wherever the motion-compensated previous
alpha already matches, INTER CAE spends bits only on the **shape innovation** (newly (dis)occluded boundary).

### 1.3 The 7 BAB modes (mode = the per-block RD decision)

| Mode | Name | Payload | When it wins |
|---|---|---|---|
| 1 | MVDs==0 & No Update | nothing (BAB ≡ MC reference) | boundary unchanged, MV predicted exactly |
| 2 | MVDs!=0 & No Update | MVD only | boundary translated, predicted MV wrong |
| 3 | all_0 (transparent) | 1 flag | BAB entirely outside object |
| 4 | all_255 (opaque) | 1 flag | BAB entirely inside object |
| 5 | intraCAE | 10-context CAE bits | new/uncovered boundary, no good predictor |
| 6 | interCAE, MVDs==0 | 9-context CAE bits | boundary innovation, MV predicted exactly |
| 7 | interCAE, MVDs!=0 | MVD + 9-context CAE bits | boundary innovation + translated block |

Modes 1–2 are the "copy from motion-compensated reference" fast paths (the bulk of a smoothly-moving object);
3–4 are the trivial interior/exterior fast paths; 5 is the intra fallback; 6–7 are the motion-compensated
innovation coders. The encoder picks the mode minimizing bits subject to the lossy criterion (§1.5).

### 1.4 Shape motion-vector prediction

The shape MV (`MVs`) is predicted (`MVPs`) from the first available candidate among neighboring **shape** MVs
(left, top, top-right BABs); if none available, it falls back to the co-located **texture** MVs. Only the
difference **MVDs = MVs − MVPs** is coded (VLC). Search is a small window around the predictor. For us this
matters structurally (§4b): the "predictor" is what ξ becomes.

### 1.5 The lossy dial — size conversion / conversion ratio (CR) + AlphaTH acceptance

The **rate-distortion knob** the classical standard had (and our task-lossy stack currently lacks a
standardized form of): before CAE, a BAB may be **down-sampled by a conversion ratio CR ∈ {1, 1/2, 1/4}**
(so a 16×16 BAB is coded as 16×16, 8×8, or 4×4), then the decoder **up-samples back** to 16×16 with a
context-based binary interpolation filter. The encoder tries the coarsest CR first and **accepts it iff the
reconstruction error stays within a threshold**: the plane is partitioned into pixel sub-blocks (4×4), and a
CR is rejected if any sub-block has more than **AlphaTH** erroneous pixels vs the original, where
**AlphaTH ∈ {0, 16, 32, …, 256}** is a VOP-level quality parameter (0 = lossless; larger = coarser/cheaper).
CR=1/4 is known to be visibly "irritating" at QCIF, so 1 and 1/2 dominate in practice. **This down/up-sample +
per-sub-block error-threshold accept/reject IS the operational R-D knob** — distortion is measured as
boundary-pixel disagreement, NOT (crucially for §4d) as an argmax flip.

### 1.6 Scan order / transposition

Each BAB may be coded **row-scan or transposed (column-scan)**, whichever yields fewer CAE bits; a 1-bit
transpose flag is sent. The CAE context is always computed in the (possibly transposed) raster order so the
causal template is well-defined. Minor (~1-bit-flag) but a real free lever.

---

## §2 — Published numbers (honest about what I could and could not extract)

**What the literature establishes (order-of-magnitude, corroborated across sources):**
- Intra CAE ≈ **~0.2–1.0 bit per boundary pixel** (a 10-state Markov source; interior is near-free, cost lives
  on the boundary). This is the standardized floor the survey's Q2 cites.
- **INTER mode cuts shape bits ~40–50% vs intra** on moving video objects — the headline temporal number and
  the reason ξ-keyed delta is the strongest UNBUILT rate axis (survey Q6).
- Lossy CR + AlphaTH trades boundary fidelity for rate monotonically; CR=1/4 is the aggressive extreme.
- Modern anchor (survey Q2, for calibration only): **CAECC (arXiv 2603.03073)** full-map lossless
  **367.5 B/frame (DAVIS-480p), 2661.7 B/frame (Cityscapes)** — 6–20× over our budget because it is
  reconstruction-lossless and never drops the ker(A)-invisible boundary.

**What I could NOT extract (flagged, not fabricated).** The exact per-VOP bits tables and the intra-vs-inter
sequence-by-sequence RD curves live in the Ostermann TNT-Hannover tutorial (368_1.pdf), the Katsaggelos
"MPEG-4 and rate-distortion-based shape-coding techniques" review, and the Columbia "Optimal buffered
compression" paper (ip01-jbl.pdf) — **all three are scanned/binary PDFs that did not OCR through WebFetch**
(saved locally under tool-results if a future pass wants to OCR them). I am NOT quoting a bits/VOP number I
could not read. The load-bearing imports (10/9-pixel context, 40–50% inter cut, CR/AlphaTH dial) are
corroborated across the readable secondary sources cited in §Sources; the precise per-VOP table is an
**open lookup** for the build, not a blocker for the design.

---

## §3 — OSS / reference code assessment

**Verdict: nothing modern/runnable implements CAE shape; the tables are published; reimplementation is cheap.**

- **MoMuSys / ISO/IEC 14496-5 reference software** — the historical MPEG-4 VM reference (C), which DOES
  implement intra+inter CAE with the full probability tables. It is the canonical source of truth, but it is
  distributed through ISO (not on modern package managers/GitHub as a maintained repo); provenance/licensing
  for direct reuse is murky and it is not build-friendly today. **Treat as a spec-equivalent reference to read,
  not a dependency to link.**
- **ffmpeg** — its MPEG-4 Part 2 decoder does **not** decode binary shape (BABs); it handles rectangular
  VOPs / SA-DCT machinery only. No CAE.
- **OxideAV/oxideav-mpeg4video** (Pure-Rust MPEG-4 Part 2, MIT) — explicitly **rejects non-rectangular shapes
  with typed errors**; SP subset only. No BAB, no CAE tables. Not usable for shape.
- **Academic reimplementations** (hardware-architecture papers, the multisymbol-CAE architecture, the York
  bandwidth-efficient encoder) describe the algorithm precisely but are not drop-in software.

**Reimplementation cost from spec (the actionable finding):** the two CAE probability tables
(**1024-entry intra + 512-entry inter**, published verbatim in ISO/IEC 14496-2) plus a standard binary
arithmetic coder + the 10/9-pixel context extractors + BAB mode logic is **~300–600 LOC** — an afternoon, not
a project. **But** (critical) — for OUR crosswalk we do **not** want the pixel-domain BAB codec at all except
as the Movable-class fallback (§5). The valuable import is the CAE *machinery* (static-table context arithmetic
coding + motion-compensated context) lifted to **parameter space**, not the binary-alpha-plane codec itself.

---

## §4 — The crosswalk (the design meat)

### §4(a) — What we code: three candidate representations for the temporal-delta format

Our "shape" is a **5-class partition**, not one binary alpha plane per object. Three ways to bring the CAE line
across, priced in §5:

**Arm A — nested / ordered binary planes + pixel-domain INTER-CAE.** Encode the 5-class partition as **4 ordered
binary planes** (the standard MPEG-4 multi-VOP composition trick: plane_k = "class ≥ k" or a fixed class
priority `Undrivable ⊃ Road ⊃ Lane ⊃ Movable ⊃ MyCar`), each coded with motion-compensated CAE. This is the
literal MPEG-4 import. **Property:** reconstruction-lossless per plane; pixel-domain; exploits temporal
redundancy via MC but **NOT** ξ (would re-derive per-block MVs the encoder searches) and **NOT** the 52% ker(A)
invisibility. Directly comparable to the CAECC 2661 B/frame anchor. **This is our incumbent's competitor, and
it loses on rate** (§5) — but it is the natural **fallback for the Movable class** (real non-rigid motion where
the parametric arms have no good warp).

**Arm B — boundary-chain CAE + shared-facet skip.** Code the partition **boundaries** as context chain codes
(survey Q2: ~1 bit/boundary-px) with **shared-boundary skip-coding** (encode each Road–Lane facet once, run-
length-skip the neighbor — attacks the 61% dominant term). Reconstruction-lossless boundary; same pixel-domain
order as Arm A. Loses on rate for the same reason (never drops ker(A)-invisible boundary).

**Arm C — kinetic power-diagram event grammar (coordinator addendum; the parametric arm).** Code OUR
parametric description (power-diagram generators `(site_x, site_y, weight)` + lane polynomials), and make it
**temporal** by ξ-advecting the generators between frames. The power diagram's dual is a **regular
triangulation**, whose combinatorial topology is **CONSTANT between flip events** (kinetic-data-structures
result). So the temporal payload is:
  - **(a) nothing** while the triangulation topology holds (the generic case under smooth ego-motion — the
    generators move but no cell adjacency changes);
  - **(b) an event record** at each combinatorial flip (an InCircle/InPower certificate fails → one local edge
    flip): ~2–3 B/generator/frame *envelope* in the KDS literature, but flips are RARE under smooth motion
    (a handful per frame across the whole diagram, not per generator);
  - **(c) small parameter residuals** — the ξ-prediction error on each generator (§4b).

This is the arm that can **beat the incumbent**, because it is the only one that stays in the argmax-lossy
parametric space our incumbent already exploits (it inherits the ≤138 B/pair intra cost and only pays the
*innovation*), and it uses the FREE ξ warp instead of a coded per-block MV search.

### §4(b) — ξ replaces block MVDs (the derivation)

In MPEG-4, each BAB codes `MVDs = MVs − MVP` after a local search — the motion is *unknown* and must be
signalled. In OUR setting the global motion is a **known screw ξ + ground-plane homography H(ξ)**, already in
the archive. So the per-BAB motion search collapses to a **closed-form prediction with zero coded MV**:

- **Road / Undrivable / MyCar (planar, rigid to ego):** these classes are (near-)planar in the ground/hood
  frame; the ground-plane homography `H(ξ)` advects their generators **exactly** (up to the planarity
  approximation). Predicted MVD ≈ 0 → the generator residual is near-zero → **these classes cost ~nothing per
  inter-frame**. This is Arm C's "(a) nothing while topology holds" for the majority of the frame.
- **Lane (thin, on the ground plane):** also advected by `H(ξ)`, but lane *dashes* appear/disappear at frame
  edges and lane geometry curves → **small non-zero residual** on the lane polynomials + occasional topology
  flips (a dash being born/killed = a generator insert/delete event). Small but non-zero.
- **Movable (cars, non-rigid, independent motion):** ξ does **NOT** predict them — the residual is the object's
  own motion. This is where Arm C degrades to Arm A (pixel-domain INTER-CAE fallback for Movable BABs only).

**Cross-check against the Road–Lane 61% prior:** this predicts the inter-frame bytes concentrate exactly where
the intra bytes already do — **Lane innovation (dashes, curvature) + Road–Lane shared facets** — because Road,
Undrivable, MyCar advect for free. Consistent with the measured 61% Road–Lane error concentration: the temporal
delta doesn't move WHERE the bytes go, it shrinks the rigid-class share to ~0 and leaves the Lane/Road-edge
share as the residual. **The temporal axis is a rate cushion on the rigid classes, redeployable to the
Road–Lane correction budget.**

### §4(c) — Decode-free asymmetry: richer context than 1999 hardware allowed

MPEG-4's 10/9-pixel templates were sized for **1999 real-time silicon** (the hardware papers: 9200 gates + 3 KB
ROM). We have a **FREE 30-min decoder** — we can run a far richer context model:
- **Larger template** (more neighbors → lower conditional entropy) — no gate budget.
- **Mixed context = template pixels + the ξ-warped prediction confidence** — i.e. condition the arithmetic
  coder not just on decoded neighbors but on *how confident the ξ-warp is at this generator* (residual
  magnitude, distance to a flip certificate). This is the modern move the CAECC anchor (2603.03073, **243
  context tables**, −14–25% vs CC-SMC) demonstrates pays single-to-double-digit % over the classic small
  context. For us the context tables are **deterministic / generic → FREE** (rule 118) as long as they are not
  trained on the contest video (they are combinatorial, keyed on the geometry, not learned). Expect a
  **~10–25% entropy trim** on the residual/event stream from a modern context vs the 1999 template — a second
  cushion on top of §4b.

### §4(d) — The lossy dial → OUR task-lossy criterion (the RD knob our stack lacks)

MPEG-4's CR/AlphaTH accepts a BAB simplification iff **boundary-pixel error ≤ AlphaTH per sub-block**. Map this
onto OUR distortion: accept a parametric simplification (skip an event, coarsen a generator residual, drop a
lane dash, merge two cells) **iff the measured through-C1 argmax flip count stays within the band**
(d_seg ≤ 3.39e-4). Concretely the build's inner loop is an **AlphaTH-analog sweep**:
  - candidate simplifications ranked cheap→expensive (by bytes saved),
  - each verified through the frozen C1 receiver for its d_seg cost (or the IDSE Jacobian surrogate from survey
    Q5 as a 7× cheaper pre-rank),
  - accept greedily while the *cumulative* d_seg stays inside the band.
This is the **task-lossy replacement for CR/AlphaTH**: their dial is reconstruction-boundary error, ours is
argmax flips — the same operational-R-D machine (Schuster–Katsaggelos DP, survey Q3) retargeted to the only
distortion the scorer sees. **This is the single most important structural import**: it is the RD knob that
lets the temporal delta spend its saved bytes *back* onto the Road–Lane correction term, or bank them toward
sub-0.15.

**Synergy note (phase/flicker):** independent per-frame fits are a known flicker source
(`feedback_flicker_floor_not_hard...`: GT-floor flicker 0.005318 binds only smoother-than-GT). A ξ-consistent
temporal delta produces a **temporally coherent** generator trajectory → *fewer* independent-fit jitters → a
plausible flicker reduction on top of the rate win. Worth measuring, not assuming.

---

## §5 — Priced verdict

### Priced B/pair for the temporal-coded description (the three arms)

Incumbent (intra, per-pair): **≤138 B/pair** for the full generator set (~17–23 generators × ~3 scalars +
lane polys), inside the ~477.8 B/pair box.

| Arm | Mechanism | Priced inter cost | Verdict |
|---|---|---|---|
| **A — pixel-domain INTER-CAE (4 nested planes)** | MC + 10/9-context CAE, reconstruction-lossless | **~250–400 B/frame → ~500–800 B/pair** (CAECC-class; ~0.5 bit/boundary-px × ~4k boundary px) | **LOSES** vs 138 B intra — dominated by our parametric incumbent; keep ONLY as Movable-class fallback |
| **B — boundary-chain + shared-facet skip** | context chain code ~1 bit/boundary-px + skip | **~200–350 B/pair**, reconstruction-lossless | **LOSES** — never drops ker(A)-invisible boundary |
| **C — kinetic power-diagram event grammar (ξ-advected)** | rigid classes advect free; pay Lane residual + flip events + Movable fallback | **~40–130 B/pair** (see below) | **WINS / the build target** |

**Arm C priced range: ~40–130 B/pair**, decomposed:
- **Best case ~40–60 B/pair** (~2.3–3.5× under the 138 B incumbent): smooth ego-motion, ground-homography holds,
  rigid classes (Road/Undrivable/MyCar) advect to ~0 residual, Lane residual ~1 B/generator, few flip events,
  Movable sparse. This matches/beats the MPEG-4 INTER-CAE 40–50% headline because our ξ-warp is free and
  near-exact for the rigid majority.
- **Central estimate ~70–90 B/pair** (~1.5–2× under incumbent): moderate Lane innovation + a handful of
  Movable BABs on pixel-domain fallback.
- **Worst case ~130 B/pair** (marginal, ~1.06× under incumbent): frequent Movable objects + high lane-topology
  flip rate (merges, dashes) push Movable onto full pixel-domain INTER-CAE and inflate the event stream.

**The dominant uncertainty** (states the assumption that owns the range): **the Movable-class real-motion
residual + the lane-topology combinatorial-flip rate** — i.e. how much of the scene is NOT rigidly advected by
ξ, and how often the regular-triangulation topology actually flips under real dash/merge dynamics. Everything
else (rigid-class advection ≈ free, modern-context trim ≈ 10–25%) is comparatively certain. The KDS worst-case
event bound is near-quadratic in generators, but under *smooth algebraic* ego-trajectories the stable-Delaunay
regime gives few events — the empirical flip rate is the number the build must measure.

**Framing against the goal (honest):** rate is **not** the currently-binding S axis (the ~477.8 B/pair box is
already met by the 138 B incumbent). The temporal delta's value is that **generator bytes ARE the ONE open RD
axis** (`seg_and_pose_solved_exact_lattice_realization_one_rd_axis`): shrinking generator description from
~138 → ~40–90 B/pair frees ~50–100 B/pair to redeploy onto the Road–Lane argmax-correction term (via the §4d
task-lossy dial) or to bank toward sub-0.15. It is a **cushion on the open axis**, not a threshold-crosser by
itself.

### Top-3 design decisions #574's build must make

1. **Representation of the delta: commit to Arm C (kinetic ξ-advected generators) as primary, Arm A
   (pixel-domain INTER-CAE) reserved for the Movable class ONLY.** Do not build the 4-nested-plane pixel codec
   as the main path — it is dominated. Build the parametric event grammar; measure whether Movable genuinely
   needs the pixel fallback or a cheap parametric Movable model suffices.
2. **ξ-warp fidelity model: ground-plane homography `H(ξ)` for the planar classes (Road/Undrivable/MyCar),
   full screw SE(3) for the rest, per-class.** This choice sets the residual magnitude — the dominant
   uncertainty above. A too-crude single-warp model inflates the residual and kills the win; the per-class
   planar/screw split is what makes the rigid classes advect to ~0.
3. **Event/keyframe cadence + task-lossy acceptance (the §4d dial): AlphaTH-analog greedy accept, ranked by
   bytes-saved, verified through-C1 (IDSE pre-rank) against the d_seg band; keyframe every N frames to bound
   drift + error resilience.** Decide N and the per-class flip-event vs residual-refit crossover. This is where
   the saved bytes get spent back onto Road–Lane correction.

### Honest kill-condition

**Kill the temporal axis (for the parametric arm) if a measured through-C1 build shows the ξ-warped
inter-frame parametric residual + event bytes ≥ ~0.7 × the intra generator bytes at equal through-C1 d_seg**
(i.e. inter ≥ ~95 B/pair) — that means ego-motion does NOT predict the *argmax-relevant* generators well
enough and you are paying nearly the full per-frame fit anyway (temporal redundancy in the argmax-relevant
description is too low to monetize). **Equivalently**, kill if the Movable-class pixel-fallback + lane
flip-event stream ALONE exceeds ~100 B/pair (the non-rigid + topology-churn share swamps the rigid-class
savings). Either measured outcome shows the argmax-relevant scene description is not temporally compressible
under a free ego-warp, and #574 should bank the geometry-fit (Arm C intra) without the temporal delta. The
falsifier is a single n600 through-C1 measurement of inter-frame residual bytes vs intra generator bytes at
matched d_seg — cheap, and it is the first thing the build should measure before investing in the full event
grammar + Movable fallback.

---

## Sources
**Primary algorithm (MPEG-4 Part 2 / ISO 14496-2):**
- Ostermann, *Coding of Binary Shape in MPEG-4*, TNT Hannover 368_1.pdf (scanned — not OCR'd; load-bearing
  context/mode details corroborated across secondary sources)
- Brady, "MPEG-4 standardized methods for the compression of arbitrarily shaped video objects" (1999)
- ISO/IEC 14496-2 (the 1024-entry intra + 512-entry inter CAE probability tables; CR/AlphaTH)
- ISO/IEC 14496-5 MoMuSys reference software (C; canonical CAE reference; ISO-distributed, not maintained OSS)
- "MPEG-4 Natural Video Coding — An overview" (img.lx.it.pt/~fp/cav) ; IDC-online MPEG-4 Standard PDF
- Katsaggelos et al., "MPEG-4 and rate-distortion-based shape-coding techniques" (review; scanned)
- "Optimal buffered compression and coding mode selection for MPEG-4 shape coding" (Columbia mmsp ip01-jbl.pdf;
  PubMed 18249659; scanned) ; "Operational R-D optimal bit allocation between shape and texture" (Katsaggelos)
- Hardware/architecture corroboration: "A multisymbol context-based arithmetic coding architecture for MPEG-4
  shape coding" (IEEE 1391002) ; "An architecture for MPEG-4 binary shape decoder" (RG 3855292) ; York
  "A Bandwidth and Memory Efficient MPEG-4 Shape Encoder" ; USPTO 6058213 / 6285795 / 6049631 / 6081554 (mode
  and MVP/MVDs definitions) ; Google Patents WO2000021295A9 (CR bit-rate control)

**OSS assessed:**
- OxideAV/oxideav-mpeg4video (Rust, MIT — rejects shape) ; ffmpeg mpeg4 (no binary shape) ; MoMuSys (ISO ref)

**Crosswalk / kinetic third arm:**
- CAECC — arXiv 2603.03073 (243 context tables; 367.5 B/frame DAVIS-480p, 2661.7 B/frame Cityscapes) [survey]
- Kinetic data structures: Kinetic triangulation (Wikipedia) ; "Kinetic Stable Delaunay Graphs" (arXiv
  1104.0622; SoCG 2010) ; CGAL Kinetic Data Structures manual — O*(n²) events, few under smooth algebraic motion
- In-tree: `generator_description_online_survey_20260719.md` (§PDW P1 kinetic event grammar; adoption #2) ;
  `seg_and_pose_solved_exact_lattice_realization_one_rd_axis_20260719.md` (the ONE open RD axis = generator
  bytes) ; Chasles 1830 / `tac.lie` (ξ screw, free predictor) ; Aurenhammer power diagrams (1987; PDW1 gate)
