# Generator-Description Survey — classical + modern, year-blind — 2026-07-19

**Agent:** GENERATOR-DESCRIPTION ONLINE RESEARCH (isolated worktree, WebSearch/WebFetch).
**Mandate:** survey the literature — **no recency floor; 1800s → 2026** (per operator addenda) — for the
"describe two 512×384×3 scorer-input planes/pair in fewest TOTAL bytes, free deterministic decoder, frozen
known consumer" problem. NO launches, no paid dispatch. Advisory only; pointer **0.19108 UNMOVED** —
everything here is a MEANS, nothing is a score.

## The box we price against (the only number that matters)

- Budget to beat S=0.19108 on rate: **~264 KB TOTAL ≈ 440 B/pair** (600 pairs), with `d_seg ≤ ~1.5e-4–6e-4`
  through the frozen SegNet **argmax**; errors concentrate on **Road–Lane edges (61% of necessary bytes)**;
  SegNet head is **rank-4 linear**; **~52% of input energy is in ker(A)** (resize null-space, scorer-invisible).
- **Incumbents any source must beat:** margin-preserving **power-diagram packet ≤138 B/pair**, lane
  polynomials **~8-dim/frame**, **exact integer-lattice receiver (C1)**.
- **Headline finding (year-blind):** the strongest *rate* answers are CLASSICAL, exactly as the operator
  predicted. The decode-compute-free + frozen-known-operator regime is the one the 19th-c geometers and the
  1990s operational-R-D / MPEG-4-shape coders built for. **But even the best classical shape coder
  (Schuster–Katsaggelos operational-R-D vertex DP; MPEG-4 CAE) is reconstruction-distortion-bounded, not
  task-lossy against a known argmax** — so our incumbents remain ~1–2 orders ahead on rate. The classics'
  value is EXACT machinery + EXACT rate tables the modern papers lack: a DP that provably minimizes
  bits-for-a-given-boundary-distortion, an LP that decides whether a partition IS a power diagram (and fits
  the generators), and a standardized motion-compensated INTER shape mode = our unbuilt temporal move.

## Precedent that makes "classical = first-class" our own house style

Our **pose axis was solved by 1800s screw theory** — Chasles (1830, rigid motion = screw), Ball (1876,
screw theory), Plücker line coordinates — now living in `tac.lie`, carrying **d_pose ≈ free**. Our
**power-diagram** rests on Laguerre geometry (1880s) over Dirichlet (1850)/Voronoi (1908) tessellations.
So 19th-c geometry is not nostalgia here; it is the tier that already delivered a solved axis. This survey
weights sources purely by **the NUMBER vs the 440 B/pair box, year-blind**.

## Prior-work de-dup (not re-surveyed)

- `carrier_sota_online_survey_20260611` + `lane_coeff_tracking_denoising_optimal_survey_20260702` already
  logged **MapTRv2 (2308.05736)** and **Coding-for-Machines RD theory (2305.17295)**. Treated as KNOWN.
- Those were carrier-focused (NeRV). The six questions here are the orthogonal vector-codec / frozen-consumer
  / classical-geometry axis — fresh in-tree.

---

## Q1 — Vectorized/parametric boundaries (bytes/frame, params/frame)

**Classical (strongest):**
- **Dickmanns 4D approach (1980s–90s)** — coded road edges as a **handful of clothoid/curvature parameters
  per frame** for real-time AV, EXACTLY our regime (tiny parametric road description, heavy known model).
  The historical existence proof that a road scene is ~O(10) parameters/frame.
- **Ramer (1972) / Douglas–Peucker (1973) polygonal approximation** — the ε-bounded min-vertex polyline with
  a hard **max-deviation ≤ ε** guarantee. Gives a *distortion-bounded* vertex count directly; O(n log n).
- **Active contours / snakes (Kass–Witkin–Terzopoulos 1988)** and **B-spline / clothoid road-edge fitting**
  (1990s AV) — energy-minimizing boundary families; a few control points/road-edge.
- **Classical envelope / caustic theory (19th c.)** — a *family* of boundaries as the envelope of a
  one-parameter curve family; relevant if Road–Lane edges are describable as one envelope + parameter.

**Modern (calibration):**
- **BeMapNet (CVPR23, 2306.09700, `er-muyue/BeMapNet`)** — piecewise **cubic Bézier**, "consistent-degree,
  dynamic-piece": ~**4–8 control pts/instance**. **MapTRv2** ~20 pts/instance [prior-logged].

**NUMBER:** distortion-bounded polylines are **~4–20 (x,y) vertices/instance** (RDP ε-bound; Bézier
consistent-degree). At fp16 ~16–80 B/instance raw. **Our ~8-dim/frame lane-poly is already ≤ this.**

**VERDICT: TEST (RDP ε-bound + Dickmanns clothoid), SKIP as codec.** Adopt the **RDP max-deviation guarantee**
to bound lane-poly vertex count against a pixel-distortion budget, and the **clothoid** parameterization
(curvature-linear-in-arclength) as a cheaper-than-polynomial road-edge family. **Integration:** lane-poly AR
coder. No rate win over incumbent; a distortion-guarantee + fewer params on curved lanes.

## Q2 — Mask / contour coding floors (bits per boundary-pixel)

**Classical (strongest — exact rate tables):**
- **MPEG-4 Part 2 binary shape coding, Context-based Arithmetic Encoding (CAE)** (Ostermann; Brady 1999) —
  the ONE standardized video-object-mask codec. Intra: **10-pel causal template** (shape as 10-state Markov
  source); **INTER: motion-compensated CAE**, context = 4 target + 5 reference pels from the previous alpha
  map. Order-of-magnitude (spec/literature): lossless CAE ≈ **~0.2–1 bit per boundary pixel**; INTER mode
  cuts shape bits **~40–50% vs intra** on video objects. (Exact per-VOP table needs the ISO 14496-2 spec;
  the *mechanism* — 9–10-bit context + motion-compensated INTER — is the load-bearing import.)
- **Freeman chain code (1961)**: raw **2–3 bits/boundary-px**. **Kaneko–Okudaira differential chain codes
  with context (1985)** and **Schindler/Moffat / Fränti context-tree chain codes**: push to **~1–1.3
  bits/boundary-px**. **Crack codes / Eden–Kolers.** **JBIG/JBIG2** region+halftone context coding.
- **Digital straightness (Klette–Rosenfeld; Freeman)** — provably minimal descriptors for straight boundary
  runs; a straight Road–Lane edge segment costs O(log length), not O(length).

**Modern (calibration):**
- **CAECC — Context Adaptive Extended Chain Coding for Semantic Maps (arXiv 2603.03073, 2026)** — extended
  chain code (**<½ the symbols of F8**), 243 context tables, **shared-boundary skip-coding**. **Full-map
  bytes/frame: DAVIS-480p 367.5 B, Cityscapes 2661.7 B**; −14–25% vs CC-SMC.

**NUMBER:** road-scene FULL partition, lossless: **Cityscapes 2661 B/frame (CAECC)**; boundary floor **~1
bit/boundary-px (context chain codes)**. Both are ~**6–20× over our budget/incumbent** — because they are
reconstruction-lossless, not argmax-lossy, and never drop the 52% ker(A)-invisible boundary.

**VERDICT: TEST (MPEG-4 INTER-CAE + shared-boundary skip), SKIP as codec.** Two imports: (a) **CAECC/JBIG2
shared-contour skip-coding** — Road–Lane edges are shared facets (61% of our cost); encode a facet once +
run-length-skip the neighbor; single-digit-% trim on ≤138 B, aimed at the dominant term. (b) **MPEG-4
INTER-mode motion-compensated shape** is the standardized precedent for Q6. **Integration:** C1 receiver /
power-diagram payload (shared-facet skip) + temporal predictor (Q6). Calibration only: our sub-1-bit/boundary
spend is legitimate ONLY because we drop ker(A)-invisible boundary — no chain-code paper can.

## Q3 — Rate-distortion for shapes / frozen-consumer coding (the exact "min bits within D")

**Classical (strongest — this IS our problem statement, solved):**
- **Schuster–Melnikov–Katsaggelos, "Operationally Optimal Vertex-Based Shape Coding" (IEEE SP Mag 15(6),
  1998; Kluwer book 1997)** — approximates a boundary by a polygon/low-order curve and solves BOTH duals
  **exactly** via DP over an admissible control-point band: *min distortion for B bits* AND **min bits for a
  given distortion D**. This is the classical incumbent of our exact-lattice C1 receiver — a provably
  operational-R-D-optimal boundary coder with published bits/object curves. Follow-ons: chord-length /
  arc-length distortion, variable-width admissible band, interframe vertex encoding (PubMed 18262906 — the
  temporal version).
- **Kolmogorov complexity of polygons / minimum-description curve families** — the theoretical floor.

**Modern (the axis we already exceed):**
- **Feature-Preserving RDO in ICM (arXiv 2408.07028 / 2504.02216)** — Taylor-expands the **frozen** feature
  extractor: IDSE `‖f(x)−f(x̂)‖²≈‖J_f(x)(x̂−x)‖²`, Jacobian-weighted quadratic; **−7.77% seg / −8.34% det mAP
  (COCO Mask R-CNN), 7.06× FLOP cut.** **Explicitly does NOT use rank-collapse/null-space.**
- **Privacy-Preserving Feature Coding (2210.00727)** — adversarial anti-inversion, >30% bits (opposite goal).
  **CRATE white-box (2311.13110).**

**NUMBER:** best published frozen-consumer codec = **~8–10% bitrate saving, full-rank Jacobian, NO null-space
use**; best classical shape coder = **operationally optimal bits-for-distortion via DP (Schuster–Katsaggelos)**
but distortion is *geometric boundary error*, not *argmax flip*. We run **rank-4 head + 52% ker(A) + exact
lattice** — categorically beyond BOTH: we optimize *argmax-flip* distortion (task-lossy), which neither the
classical vertex-DP nor the modern IDSE codec does.

**VERDICT: ADOPT the Schuster–Katsaggelos DP framing + IDSE as a cheap proxy; SKIP both as codecs.**
(a) **Re-cast C1** in the Schuster–Katsaggelos operational-R-D DP language: our lattice receiver IS a
"min-bits-within-distortion" solver — adopt their **admissible control-point band + Lagrangian sweep** to
choose which boundary vertices/generators to spend bytes on, but with **d_seg (argmax flip) as the distortion
D** instead of geometric error. This is the single most aligned classical result in the survey.
(b) **IDSE Jacobian** = a 7× cheaper first-order surrogate for through-R d_seg/byte to rank candidates before
exact C1 verify. (c) **This pair is the citation that proves our null-space rate mechanism is beyond SOTA
CFM** (NO-FAKE #7 originality). **Integration:** C1 receiver DP + generator-fitting ranker.

## Q4 — Power diagrams / Laguerre tessellations fitted to segmentation fields

**Classical (strongest — exact fit machinery):**
- **Laguerre geometry (1880s) / Dirichlet (1850) / Voronoi (1908)** — the generator-with-weight cell family
  our packet already uses.
- **Aurenhammer, "Power Diagrams: Properties, Algorithms and Applications" (1987)** + **"A criterion for the
  affine equivalence of cell complexes … convex polyhedra in R^(d+1)" (1987)** — power diagram = lifting to a
  convex polyhedron in one higher dim; **2D build O(n log n)**; generator = (site, weight) ≈ **3 scalars/cell**.
- **Power Diagram Detection (Aurenhammer criterion → JOTA 2018 10957-018-1442-y; constrained-clustering-via-
  diagrams, arXiv 1703.02867)** — **a simple LINEAR PROGRAM decides whether a given partition IS a power
  diagram, and recovers the weights.** This is the exact-feasibility + exact-fit tool under our packet.

**Modern (scalable solvers):**
- **Combinatorial semi-discrete OT (NeurIPS 2024, 2d950a2c)**; **inverse Laguerre — ESAIM:M2AN 59 (2025)
  841–871** (fit power diagram whose **cell volumes AND centroids match targets**); **distributed Voronoi
  SDOT (2406.04192)**; **higher-dim power diagrams (2106.14730)**.

**NUMBER:** each generator = **~3 scalars** (site x,y + weight); fit is **exact** — an LP for representability
(Aurenhammer/JOTA) + volume+centroid inverse (ESAIM). Our ≤138 B packet fitted heuristically (Lloyd) becomes
an **exact convex solve**, so the same generator count lands a margin-optimal partition needing fewer
correction bytes.

**VERDICT: ADOPT (the exact fit).** Replace heuristic Lloyd relaxation with **Aurenhammer's power-diagram-
detection LP** (is our target argmax partition a power diagram? if not, which cells need extra generators?)
+ **semi-discrete-OT / inverse-Laguerre** volume+centroid solve. **Best chance in the survey to shave the
≤138 B packet AND lower d_seg at equal bytes** — a better-fitted generator set needs fewer correction bytes,
and the LP tells us the *minimum* generator count to represent the partition exactly. **Integration:**
power-diagram payload builder (offline; free decoder unchanged).

## Q5 — Program-style / MDL description (free decode compute = our asymmetry)

**Classical (strongest):**
- **Rissanen MDL two-part codes (1978)** — description length = model bits + residual bits; the exact ledger
  for "generator program + correction bytes." Our packet IS a two-part code; MDL is the accounting law.
- **IFS / fractal coding (Barnsley 1988; Jacquin 1992)** — **payload-light, decode-HEAVY** (iterate a
  contractive map to a fixed point). The precise free-decoder asymmetry we have: ship the map's few
  parameters, let the free 30-min decoder iterate. Road/lane self-similarity is weak, but the *paradigm* is
  our exact situation.
- **Grammar-based image coding / classical shape grammars** — a scene as a short production; decode = derive.

**Modern (mostly SKIP — learned codebooks are COUNTED bytes):**
- **Hierarchical SVG Tokenization (2604.05072)**, **VectorArk rounded-polygon (2605.24398)**, **SVGFusion
  (2412.10437)**, **NS Program Synthesis (PLDI 2025 / 2508.15750)** — human-fidelity token counts
  (hundreds–thousands/image); a learned SVG tokenizer ships a **dataset-trained codebook = COUNTED
  video-derived bytes (rule 118)** → net loss vs our deterministic generators.

**NUMBER:** no modern paper reports byte scales at our tolerance; classical **MDL gives the exact two-part
floor**, and **IFS/fractal proves payload-light+decode-heavy is a real regime** (fractal image codecs shipped
kilobytes for whole images in the 1990s). Not below our incumbent, but the accounting is ours.

**VERDICT: ADOPT MDL as the accounting law; SKIP program-synthesis codecs; TEST VectorArk rounded-polygon
shape.** Use **Rissanen two-part MDL** as the explicit ledger for generator-program-bits + correction-bytes
(it disciplines when to add a generator: only if it saves more residual bits than it costs). Keep **VectorArk
rounded-polygon** as a candidate free-decoder shape for the MyCar/Movable cells (hood/car outline). Reject
learned SVG tokenizers (rule-118 counted codebook). **Integration:** power-diagram payload (MDL stop rule +
rounded-polygon cell variant).

## Q6 — Temporal amortization (1200 frames; delta-coded control points)

**Classical (strongest — standardized + exact):**
- **MPEG-4 INTER-mode CAE (motion-compensated binary shape)** — the standardized temporal mask codec:
  predict the current alpha block from the motion-compensated previous alpha, arithmetic-code the residual;
  **~40–50% shape-bit cut vs intra.** Directly our move.
- **Kalman snakes / classical contour tracking (Terzopoulos–Szeliski; Blake–Isard CONDENSATION 1998)** —
  predict boundary control points across frames, store only the innovation. **Interframe vertex-based shape
  coding (Schuster–Katsaggelos lineage; PubMed 18262906)** — the temporal version of the Q3 DP.
- **Ego-motion = our stored se(3) screw (Chasles/Ball, `tac.lie`)** — already in the archive for pose;
  dual-use to warp the vector description frame-to-frame.

**Modern (accuracy, not rate):**
- **PrevPredMap (2407.17378)**, **StreamMapNet**, **MemFusionMap (2409.18737)**, **TopoStreamer (2507.00709)**
  — previous-prediction / working-memory temporal fusion (optimize mAP, publish no temporal-delta bytes).

**NUMBER:** MPEG-4 INTER-CAE **~40–50% shape-bit reduction**; no vector-map paper publishes temporal-delta
bytes. Our 600 pairs currently pay each lane/generator fit in full — **temporal redundancy is the one
unexploited rate axis.**

**VERDICT: TEST (build) — the highest-leverage UNBUILT rate idea.** Predict frame t+1 generators/lane
control-points from frame t via the **already-stored ego-screw ξ** (Chasles/`tac.lie`), AR-code only the
residual — the MPEG-4 INTER-CAE structure with our free known warp. 1200 frames → ~1 keyframe + tiny
residuals instead of 600 independent fits. **Integration:** lane-poly AR coder + power-diagram payload:
ξ-keyed inter-frame predictor + residual arithmetic coder.

---

## Ranked top-5 adoption list — year-blind, priced against the 440 B/pair box

| # | Import | Source(s) (year-blind) | What it buys (vs 440 B/pair box, ≤138 B incumbent) | Verdict | Integration point |
|---|--------|------------------------|-----------------------------------------------------|---------|-------------------|
| 1 | **Aurenhammer power-diagram-detection LP + inverse-Laguerre generator solve** | Aurenhammer 1987; JOTA 2018 10957-018-1442-y; NeurIPS 2024 semi-discrete OT; ESAIM:M2AN 59 2025 p.841 | EXACT: does our argmax partition = a power diagram? LP recovers the **minimum** generator set + weights (vol+centroid matched). Fewer generators / fewer correction bytes at equal d_seg → plausibly shaves ≤138 B AND lowers d_seg. Highest-confidence rate win. | **ADOPT** | power-diagram payload builder (offline; decoder unchanged) |
| 2 | **ξ-keyed temporal delta (MPEG-4 INTER-CAE structure over stored ego-screw)** | MPEG-4 INTER-CAE (Ostermann/Brady); interframe vertex DP (PubMed 18262906); Chasles 1830 / `tac.lie`; PrevPredMap 2407.17378 | Only UNEXPLOITED rate axis: 600 independent fits → 1 keyframe + 1199 residuals; MPEG-4 measures ~40–50% shape-bit cut. Largest potential B/pair reduction; uses an ξ we already store. | **TEST (build)** | lane-poly AR coder + power-diagram payload: ξ predictor + residual coder |
| 3 | **Schuster–Katsaggelos operational-R-D vertex DP, retargeted to d_seg distortion** | Schuster–Melnikov–Katsaggelos IEEE SP Mag 1998; Kluwer 1997; RDP 1972/73 ε-bound | The classical "min bits within distortion D" solved by DP — our exact problem. Retarget D = argmax-flip. Principled Lagrangian byte-allocation over boundary vertices/generators; disciplines WHERE bytes go (the 61% Road–Lane term). | **ADOPT (framing)** | C1 receiver DP + admissible-band byte allocator |
| 4 | **Shared-boundary skip-coding (JBIG2 / CAECC) + Rissanen MDL stop-rule** | CAECC 2603.03073; JBIG2; Rissanen MDL 1978; digital-straightness Klette–Rosenfeld | Road–Lane edges are shared facets (61% of cost): encode facet once + run-length skip; MDL two-part code decides when a generator earns its bytes. Single-digit-% trim directly on the dominant term. | **TEST** | C1 receiver / power-diagram payload: shared-facet skip + MDL add-generator gate |
| 5 | **IDSE Jacobian surrogate loss (frozen-scorer RDO)** | Feature-Preserving RDO 2408.07028 / 2504.02216 | Not a byte win — a **7× cheaper** first-order proxy for through-R d_seg/byte to rank candidate generators before exact C1 verify. Also the **citation proving our null-space rate mechanism is beyond SOTA CFM** (NO-FAKE #7). | **ADOPT (proxy) / SKIP (codec)** | generator-fitting inner loop / margin-saliency ranker |

### Honest "nothing beats what we have"

- **No codec in ANY era is task-lossy against a frozen known argmax.** MPEG-4 CAE, Schuster–Katsaggelos
  operational-R-D DP, CAECC (2661 B/frame Cityscapes lossless), context chain codes (~1 bit/boundary-px),
  feature-preserving RDO (−8%) — all are reconstruction-lossless or geometric-distortion-bounded or
  human-fidelity. Our **≤138 B power-diagram + ~8-dim lane-poly + exact-lattice receiver sit ~1–2 orders
  below** because we drop the **52% ker(A)-invisible** signal and fix only **argmax-flipping Road–Lane edges**
  — a move no surveyed source makes.
- **Q3 confirms our originality:** the closest classical (vertex-DP) minimizes *geometric* boundary bits;
  the closest modern white-box CFM explicitly *avoids* null-space coding. Our rate mechanism is beyond both.
- **Net:** the corpus offers **exact fitting machinery (Aurenhammer LP / inverse-Laguerre), an exact
  byte-allocation DP (Schuster–Katsaggelos), a standardized temporal structure (MPEG-4 INTER-CAE), an
  accounting law (MDL), and a cheap surrogate (IDSE)** — not a smaller payload. The imports with a plausible
  B/pair cut are **#1 (exact generator solve)** and **#2 (temporal delta)**; #3 is a byte-allocation
  discipline; #4 a marginal trim; #5 a proxy + originality citation.

## Sources
**Classical / foundational (year-blind first-class):**
- Laguerre geometry (1880s); Dirichlet (1850); Voronoi (1908); Chasles rigid-motion-as-screw (1830); Ball, *Theory of Screws* (1876); Plücker line coordinates — [in-tree precedent: `tac.lie`, pose solved]
- Aurenhammer, *Power Diagrams: Properties, Algorithms and Applications* (1987) + affine-equivalence criterion (1987); Power Diagram Detection — JOTA 10.1007/s10957-018-1442-y (2018); constrained-clustering-via-diagrams — arXiv 1703.02867
- Schuster, Melnikov, Katsaggelos, *Operationally Optimal Vertex-Based Shape Coding*, IEEE Signal Processing Mag. 15(6):91–108 (1998); *Rate-Distortion Based Video Compression*, Kluwer (1997); interframe vertex shape — PubMed 18262906
- MPEG-4 Part 2 binary shape / CAE — Ostermann (TNT Hannover 368_1.pdf); Brady, "MPEG-4 standardized methods for the compression of arbitrarily shaped video objects" (1999); ISO/IEC 14496-2
- Freeman chain code (1961); Kaneko–Okudaira differential chain codes w/ context (1985); Fränti context-tree chain codes; JBIG/JBIG2; digital straightness (Klette–Rosenfeld)
- Ramer (1972) / Douglas–Peucker (1973); Kass–Witkin–Terzopoulos snakes (1988); Dickmanns 4D road model (1980s–90s); Blake–Isard CONDENSATION (1998); Kalman snakes
- Rissanen MDL two-part codes (1978); Barnsley IFS (1988); Jacquin fractal image coding (1992)
**Modern (2024–2026):**
- CAECC — arXiv 2603.03073 ; Feature-Preserving RDO in ICM — arXiv 2408.07028 / 2504.02216 ; Privacy-Preserving Feature Coding — arXiv 2210.00727 ; CRATE — arXiv 2311.13110
- BeMapNet — arXiv 2306.09700 (`er-muyue/BeMapNet`) ; MapTRv2 — arXiv 2308.05736 [prior-logged] ; StreamMapNet ; PrevPredMap — arXiv 2407.17378 ; MemFusionMap — arXiv 2409.18737 ; TopoStreamer — arXiv 2507.00709
- Semi-discrete OT — NeurIPS 2024 (proceedings 2d950a2c…) ; inverse Laguerre — ESAIM:M2AN 59 (2025) 841–871 ; distributed Voronoi SDOT — arXiv 2406.04192 ; higher-dim power diagrams — arXiv 2106.14730 ; OT for ML learners — arXiv 2505.06589
- Hierarchical SVG Tokenization — arXiv 2604.05072 ; VectorArk — arXiv 2605.24398 ; SVGFusion — arXiv 2412.10437 ; NS Program Synthesis — PLDI 2025 / arXiv 2508.15750
- Prior in-tree: `carrier_sota_online_survey_20260611`, `lane_coeff_tracking_denoising_optimal_survey_20260702` (MapTRv2 + Coding-for-Machines RD 2305.17295 logged)
