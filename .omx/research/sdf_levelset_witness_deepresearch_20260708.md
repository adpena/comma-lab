# SDF LEVEL-SET WITNESS — deep-research + deep-math stress test of v7.5 / v8 vs the external frontier

**Date:** 2026-07-08 · **Axis:** design-only online research + re-derivation, **$0, no launches, no trainer
edits**; run-1 (pid 63069) + all run dirs READ-ONLY, UNTOUCHED. **Pointer contest-CPU 0.19110 UNMOVED —
this unit is MEANS** (the pointer moves ONLY through a byte-closed `upstream/evaluate.py` n600 exact row).
Author: SDF-witness deep-research agent (operator-dispatched). Every external claim is CITED; every number
is labeled MEASURED (ours, n600) / DERIVED / lit-CITED / ASSERTED; ours-vs-borrowed separated (NO-FAKE #7);
verdict_scope on every negative.

## STORES CONSULTED
- **Internal (authority):** `t5_crucible/SPEC_v75_optimal_single_trunk_20260708.md` · `SPEC_v8_perclass_
  decomposition_20260708.md` · `perclass_carriers_design_20260708.md` · `counterforce_insufficiency_
  deepmath_20260708.md` · `probe_PA_paintfloor_perclass_20260708.md` (n600, the measured floor) · DAG
  FEED-perclass/PA/v8risks/mergediff/missingforces/roadfloor · CLAUDE.md §OPERATOR PRIORITY + §WITNESS
  CAPSTONE + §SegNet/PoseNet architectures · MEMORY L65/L66/L68/L69/L74/L75/L17.
- **External (all cited inline + in §7):** tropical DNN geometry; Maslov dequantization; Candès-Donoho
  curvelets + shearlet cartoon approximation; Kim-Fridovich-Keil grids-vs-INR (#308 source); COIN/COOL-CHIC
  INR compression; Dubois sufficient-statistic compression; indirect/CEO rate-distortion + f-separable;
  coding-for-machines (VCM) frontier; RAG boundary coding; deep active contours; SE(3) ego-motion.

---

## 1. HEADLINE VERDICT

The v7.5/v8 **mathematical skeleton is unusually well-aligned with the current literature** — more so than
most of our past vehicles. Four of the load-bearing equivalences survive re-derivation cleanly; two survive
only in REFINED form; the framing that is genuinely *decorative* is a small minority. The literature does
**not** hand us a codec that dominates the split-vehicle / edge-centric design — because our binding
constraint (a **frozen** scorer + payload-only archive, with a **non-quadratic argmax-0-1** distortion) sits
outside the trainable-codec setting almost all VCM/coding-for-machines work assumes. That is simultaneously
(a) confirmation the design is on genuinely novel ground (good for NO-FAKE #7 originality) and (b) a warning
that the SOTA rate machinery we *can* borrow (overfitted-INR entropy coding) lives on the **rate** axis, not
the seg/pose-distortion axis.

**The one place the design and the literature agree the vehicle is not yet a sub-0.19 pointer-mover: pose.**
The lit is unambiguous (§4) that reproducing two-frame flow from a single rigid twist is under-parameterized;
this is independent of the seg decomposition and binds v7.5 AND v8 equally.

---

## 2. CONFIRMED-SOUND vs FRAGILE/MISSING (with citations)

| # | Design claim | Verdict | Basis (ours = MEASURED n600 · lit = cited) |
|---|---|---|---|
| C1 | **Argmax partition = tropical (max-plus) object; separatrix = tie locus, derived not stored** | **CONFIRMED** | The SegNet ReLU-net decision boundary *is* a tropical hypersurface (Zhang 2018; Alfarra 2020, arXiv:2002.08838). `argmax_c(φ_c+b_c)` is literally a tropical polynomial's max. Sound by construction. |
| C2 | **Theft-impossibility: ∂φ_c/∂θ_{c'}=0 with per-class-independent params** | **CONFIRMED (trivial)** | Parameter decoupling ⇒ zero cross-gradient is definitional. The *residual* argmax coupling (b_c calibration) is correctly named in the design's own §1/§7. verdict_scope: the gradient-theft mechanism is closed; the P-B probe is still the right falsifier. |
| C3 | **Edge-centric decomposition: one field per adjacency-graph EDGE, not per region** | **CONFIRMED** | RAG boundary-coding canon: "the most efficient way of coding group membership is to code the *boundary*; an edge = a shared boundary segment" (Freeman chain code; RAG literature). Mobahi-Rao-Yang-Ma "Segmentation by Texture and Boundary Compression" (arXiv:1006.3679) is the MDL precedent. The v8 cure (encode each shared curve once) is exactly right. |
| C4 | **d_seg factorizes over pairwise tie-loci on the RAG** | **CONFIRMED (measured)** | OUR P-A n600: destination matrix = the RAG; every class flips ONLY at its Road separatrix, **zero interior flips** (Road hub 43.7%). MEASURED, reproduces the 0.000910 floor to 15 digits. |
| C5 | **Grid-bulk + INR-annulus hybrid carrier (#308)** | **CONFIRMED + sharpened** | Kim & Fridovich-Keil, "Grids Often Outperform INRs at Compressing Dense Signals" (arXiv:2506.11139, NeurIPS 2025): regularized grids beat INRs on dense signals **except** binary signals / shape contours, where INRs win. This is EXACTLY the bulk(grid)/boundary(INR) split. Strong external validation of the single most consequential v8 architecture choice. |
| C6 | **Curvelet = optimal sparse basis for the curved codim-1 singularity = minimal rate** | **REFINED (partly fragile)** | The L²-approximation half is solid (Candès-Donoho 2004 CPAM; shearlet cartoon N⁻² vs wavelet N⁻¹). **Three caveats** (§3): (a) optimality is L²-n-term, NOT archive-bytes/entropy; (b) our target is piecewise-CONSTANT with possibly non-C² edges (lane *dashes*) → α-scaling says parabolic (α=½) may be the wrong anisotropy (Grohs et al. α-curvelets, arXiv:1404.1043); (c) grids beat a pure curvelet/INR on the bulk (C5). "Curvelet=rate" is decorative as stated; the correct load-bearing form is C5 + a directional annulus basis. |
| C7 | **Distortion = Fisher metric on the separatrix; margin ↔ Fisher, Pearson 0.978** | **SURVIVES (ours-measured; lit-consistent direction)** | 0.978 is OUR measurement, not lit. Direction confirmed: steganographic Fisher information = reciprocal local std-dev = detectability = classifier-margin sensitivity (UNIWARD/Holub-Fridrich; square-root law/Ker; steg-Fisher↔KL, arXiv:2111.04960). Load-bearing for margin-band satisficing (Force-2) + UNIWARD cost. Keep labeled "ours-measured, lit-supported." |
| C8 | **Curriculum = coarse-to-fine = annealing = τ→0 tropical limit (τ=ε=ħ, L75)** | **CORE SURVIVES; chain partly decorative** | The τ→0 = Maslov-dequantization = tropical limit is RIGOROUS: `lim_{τ→0} τ·LSE(z/τ)=max(z)`, a semiring homomorphism onto (ℝ,max,+) (Maslov; "Transformer as Tropical Polynomial Circuit" arXiv:2601.09775; "Hamilton-Jacobi Theory of Deep Learning" arXiv:2605.28983). The `=persistence-order=curvelet-scale` links are analogies, not identities — keep them as intuition, not as load-bearing equalities. |
| C9 | **Chan-Vese area-Lagrange = missing area term of the level-set energy** | **CONFIRMED** | Chan-Vese + deep active contours are standard (DACN MICCAI 2020; end-to-end DCAC arXiv:1909.13359; Deep Level Set arXiv:2112.03451). OUR counterforce_insufficiency memo correctly derives it is the exact dual of the measured mass-conservation identity, and correctly bounds it to ~96% of the Road deficit with the placement residual ORTHOGONAL. Honest and sound. |
| F1 | **Paint problem (evaluator-inverse): generated frame1 → frozen SegNet → intended argmax** | **FRAGILE / UNDER-VALIDATED** | This is the true risk and the literature is thin here: essentially all VCM/coding-for-machines work (§4) trains the codec end-to-end WITH the task net; our frozen-scorer + generated-paint + rule-118-counted-payload setting is much harder and largely unstudied. OUR oracle floor (P-A: 0.00091 with REAL texture) is an UPPER bound; the procedural-fill floor is UNMEASURED. P-C is correctly the decisive $0 probe — do NOT design the paint stage before it runs. |
| F2 | **Pose: store-nothing-ξ carrier hits the target** | **FRAGILE (the blocker; lit-confirmed under-parameterized)** | §4. Independent of the seg work; binds v7.5 and v8 identically. |
| F3 | **20–50 KB Road/Undriv bulk-boundary band** | **CONSERVATIVE-leaning-confirmed but UNMEASURED** | OUR P-A refuted the "interiors are the hard part" risk (Road/Undriv within-class flip 0.17%/0.03%, the LOWEST). But the exact byte figure is increment-1's to measure; the oracle uses real-frame texture (upper bound). Correctly labeled DERIVED. |

---

## 3. RE-DERIVED FLOOR + WHICH EQUIVALENCES SURVIVE

### 3a. The task-RD floor (re-derived against the current indirect-RD + sufficient-statistic literature)

Score: `S = 100·d_seg + √(10·d_pose) + 25·B/37,545,489`. Rate cost per KB = 25·1024/37,545,489 ≈
**6.82e-4 S/KB** (40 KB → 0.0273; 114 KB → 0.0759; 255 KB → 0.170). [DERIVED, arithmetic]

The correct modern framing is the **single-terminal INDIRECT (remote) rate-distortion problem**, NOT the
multi-terminal CEO problem (Berger 1996) — we have one encoder observing the source and one decoder
reconstructing for a frozen predictor. Two literature pillars sharpen our floor:

1. **Sufficient statistic (Dubois et al. 2021, "Lossy Compression for Lossless Prediction," arXiv:2106.10800,
   NeurIPS).** To preserve a task, you need only the bits of the task's **minimal sufficient statistic**.
   Dubois's setting preserves ALL tasks invariant under a transform set (hence their 1000× ImageNet savings);
   **our case is strictly EASIER** — a SINGLE frozen scorer, so the sufficient statistic is just
   {argmax-partition sequence, 6-pose sequence}. This is precisely the quotient codec (#155) and it says the
   floor = coding cost of that statistic, NOT of the video. Confirms the direction of FEED-af's
   exact-partition store.
2. **f-separable indirect RD (arXiv:1505.04875 and the f-separable indirect-RD work).** Our distortion is
   **argmax 0-1 (d_seg) + a √ pose term** — NON-quadratic. Gaussian-CEO / quadratic intuitions do NOT
   transfer; the floor must be the actual argmax-0-1 indirect-RD curve. This is a real caveat against any
   MSE-proxy floor.

**Re-derivation of S_floor ≈ 0.118.** Bracket it with our two MEASURED endpoints:
- **d_seg=0 endpoint:** exact partition store = 255 KB (FEED-af, MEASURED context-arith) → rate 0.170 →
  `S = 0 + 0.170 + pose`. Lossless-partition alone already exceeds sub-0.15 on rate.
- **Oracle-lossy endpoint:** P-A MEASURED the through-R achievable d_seg = **0.00091** (composite, n600). At
  that d_seg, `100·d_seg = 0.091`. If the carrier lands near its DERIVED 40 KB (rate 0.0273):
  `S ≈ 0.091 + 0.0273 + pose ≈ 0.118 + pose`.

**Verdict on the floor:** S_floor ≈ 0.118 is still the right floor **but it is a SEG+RATE floor that assumes
pose ≈ 0.** Re-derived, it decomposes cleanly as **oracle-d_seg (0.091) + ~40 KB rate (0.027)**. The
literature (indirect-RD + sufficient-statistic) validates the *method* of computing it. The live term the
floor hides is pose: with the ANCESTOR pose 0.018 → 0.136 (sub-0.15 reachable); with the MEASURED witness
pose √(10·1.79)≈4.24 → hopeless. **The floor is honest; the pose term is where the campaign actually lives.**

### 3b. Equivalence survival summary
- **SURVIVE rigorously:** tropical-argmax (C1/C2), edge=RAG (C3/C4), grid-bulk/INR-annulus (C5),
  τ→0=Maslov=tropical (C8 core), Chan-Vese=area-Lagrange (C9).
- **SURVIVE refined:** curvelet-optimality → "directional multiscale annulus + grid bulk" (C6); margin↔Fisher
  as ours-measured/lit-supported direction (C7).
- **DECORATIVE (demote from load-bearing):** the `persistence-order = curvelet-scale = annealing` identity
  chain (keep as intuition); "curvelet = rate" as a standalone equality (rate is entropy/bytes, not L²
  n-term — a metric mismatch).

---

## 4. THE POSE-HARD VERDICT AGAINST THE DESIGNS (does it reshape v8?)

**MEASURED (ours):** run-1 d_pose ≈ 1.79 flat ⇒ √(10·1.79) ≈ 4.24 of S from pose alone. The store-nothing-ξ
carrier (single-keyframe homography + rank-6 twist) is H-target capped ~2.5 by construction. The ancestor
3.4e-5 is full-RGB photometric, NON-transferable (L68).

**Lit confirms the under-parameterization.** Reproducing two-frame flow needs a **dense SE(3) motion field**
(per-pixel rigid transforms) or a **depth + pose** decomposition (DeMoN; EMR-MSF, arXiv:2309.01296;
SfMLearner lineage). A single rigid twist reproduces the flow of a *planar* scene or *pure rotation* only;
with translation + scene-depth parallax it CANNOT. So the H-cap is not an implementation limit — it is the
geometry, and the literature says so.

**Does this reshape v8 (which treats frame0 as pure-pose territory)?**
- **No structural damage to v8's SEG architecture** — pose lives on frame0 / luma; the seg decomposition is
  frame1 / chroma-first. The channel-split (SegNet reads `x[:,-1]` only; PoseNet reads YUV6×2, luma-dominated)
  is a *genuinely good* match to the frozen scorer and is confirmed by the modules.py asymmetry. v8 correctly
  **isolates** pose rather than solving it.
- **But v8 inherits the SAME open pose risk** — "frame0 carries pose" only works if a low-byte frame0 +
  painted-frame1 pair can hit the 6 PoseNet scalars. That is unvalidated for BOTH vehicles.

**Does the literature offer a joint seg+pose representation that DOMINATES the split-vehicle?** — **No, not
for our constraint.** The joint multi-task coding-for-machines SOTA (All-in-One / Multi-Path Aggregation,
NeurIPS 2024, arXiv:2409.19660; latent-space scalability, arXiv:2205.01874) uses a **shared trainable latent
with task-specific paths** — it assumes you TRAIN the codec jointly with the task nets. We have a **frozen**
scorer and a payload-only archive; the shared-latent trick does not apply. The channel/frame split we use is
the constraint-matched analogue and is arguably better-posed than forcing a joint latent.

**Recommendation (reshapes the pose carrier, not v8's seg):** the pose carrier is under-parameterized by the
same geometry the lit describes. The break-even MUST be re-derived with MEASURED d_pose, and the carrier
likely needs to store MORE than nothing — either the **6 target scalars directly** (Quantizr stored-target
sidecar, ~7 KB raw / 1-2 KB coded — the CLAUDE.md-canonical pose solution, distinct from store-nothing-ξ) OR
a **low-rank depth/flow sidecar**. Note the crucial reframe: d_pose is MSE on 6 PoseNet *outputs*, NOT on the
flow field — so this is an **evaluator-inverse (adversarial) pose problem**, not a flow-reproduction problem;
the render must FOOL PoseNet into emitting the target 6 scalars. That is genuinely open and unstudied in the
lit, and is the campaign's real frontier. verdict_scope: **FORMULATION** (the store-nothing-ξ formulation is
capped; pose-as-a-paradigm via stored-target + FiLM is NOT killed — it is the untested alternative).

---

## 5. TOP-3 EXTERNAL TECHNIQUES TO DRAW FROM (ranked by measured-EV to the pointer)

1. **Overfitted-INR entropy coding — COOL-CHIC 5.0 / LANCE (arXiv:2605.02726, arXiv:2605.20672) + COIN
   (arXiv:2103.03123).** **EV: HIGH (rate axis; MEASURED headroom 0.049 S / ~74 KB).** Our witness weights
   ARE an overfitted INR; the rate half of sub-0.15 is the 114→40 KB drop. COOL-CHIC's autoregressive latent
   entropy model + low-complexity coordinate decode is production-proven and byte-close-compatible, and it
   composes with the grid-bulk hybrid (their latents ARE a grid). This is the single most directly
   actionable, highest-measured-EV borrow. Draw: the AR latent entropy model for the bulk-boundary carrier's
   byte-close.
2. **Grid-bulk + INR-annulus split — Kim & Fridovich-Keil 2025 (arXiv:2506.11139).** **EV: HIGH (BOTH
   d_seg-placement and rate; directly gates increment-1).** Already our #308; the paper's precise result
   ("INRs win ONLY on binary/contour signals, grids win on dense") is the theoretical license for the exact
   increment-1 carrier. Draw: regularized-grid interior (fast, stable, error-bounded) + a small INR/directional
   head on the annulus ONLY, with the paper's grid-regularization recipe.
3. **f-separable indirect-RD loss + feature-preserving RDO (arXiv:2504.02216 / 2408.07028; f-separable
   indirect RD arXiv:1505.04875; Dubois sufficient-statistic arXiv:2106.10800).** **EV: MEDIUM-HIGH
   (loss-design + floor correctness; guards against MSE-proxy drift).** Reframes the training objective to
   optimize the ACTUAL argmax-0-1 (f-separable) indirect distortion — the "17% bitrate savings for equal task
   accuracy vs SSE-RDO" anchor is the measured evidence the reframe pays. Draw: the sufficient-statistic view
   to keep the quotient codec honest + the f-separable distortion to stop optimizing a quadratic proxy of a
   0-1 target.

**Honorable mentions:** (a) **α-curvelet anisotropy tuning** (Grohs et al., arXiv:1404.1043) — if the lane
annulus edges are non-C² (dashes), the optimal directional anisotropy is α≠½; a cheap sweep. (b) **End-to-end
deep active contours / Deep Level Set** (arXiv:1909.13359 / 2112.03451) — the modern learned form of the
Chan-Vese machinery v7.5 already uses, useful if the boundary field is trained.

---

## 6. CANDIDATE CANONICAL EQUATIONS (council-FLAGGED, NOT registered — anchors owed per triality discipline)

1. **`softmax_tau_maslov_tropical_limit_v1` (candidate).** `lim_{τ→0} τ·LSE(z/τ) = max_c z_c`; the SegNet
   argmax partition is the tropical hypersurface of the logit fields; the witness partition `argmax_c(φ_c+b_c)`
   is the τ→0 Maslov-dequantization limit of the softmax the curriculum anneals. **Unifies** L75's τ=ε=ħ with
   the v8 tropical framing. Anchor owed: our τ-curriculum d_seg rows + increment-1 tie-locus rows. Lit basis:
   Maslov dequantization; arXiv:2601.09775; Zhang 2018.
2. **`cartoon_directional_annulus_grid_bulk_rate_law_v1` (candidate; supersedes the naive "curvelet=rate").**
   For a piecewise-constant partition with piecewise-C² separatrix, the archive-rate-optimal carrier is
   **grid-bulk (dense interior, near-free through-R) + directional-multiscale annulus (boundary, where the
   bytes go)** — NOT a single INR and NOT a pure curvelet. Anchor owed: increment-1 byte-close. Lit basis:
   Candès-Donoho CPAM 2004 (N⁻² vs N⁻¹); Kim-Fridovich-Keil arXiv:2506.11139; OUR P-A (100% residual on the
   4.7%-area annulus, L66).
3. **`indirect_task_rd_sufficient_statistic_floor_v1` (candidate).** Witness floor = coding cost of the
   frozen scorer's minimal sufficient statistic {argmax partition seq, 6-pose seq} under f-separable
   (argmax-0-1) indirect distortion; `S_floor = min_B [100·d_seg*(B) + 25·B/B0] + √(10·d_pose)`, bracketed by
   the MEASURED endpoints 255 KB@d_seg=0 (rate 0.170) and ~40 KB@oracle-d_seg 0.00091 (S_seg+rate ≈ 0.118).
   Anchor owed: increment-1 rate + a MEASURED witness d_pose. Lit basis: Dubois arXiv:2106.10800; f-separable
   indirect RD arXiv:1505.04875; Berger CEO 1996.

(The design's own two flags — `tropical_perclass_reconciliation_v1`, `perclass_rate_waterfill_v1` — remain
council-flagged; this research adds no anchor for them beyond P-A's RAG destination matrix, which is already
credited.)

---

## 7. OURS-vs-BORROWED ACCOUNTING (NO-FAKE #7)

- **OURS (measured/derived, not in the lit):** the n600 per-class flip attribution + RAG destination matrix
  (P-A); the mass-conservation theft identity (0.1189≈0.1191); the area-theft-vs-placement d_seg
  decomposition; the through-R oracle floor 0.00091; the 0.978 margin↔Fisher measurement; the frozen-scorer
  channel-split (frame0-pose/frame1-seg, luma/chroma) reconciliation; the store-nothing-ξ H-cap measurement.
- **BORROWED (external, cited):** tropical DNN geometry; Maslov dequantization; curvelet/shearlet
  cartoon-approximation optimality; grids-vs-INR; COIN/COOL-CHIC INR entropy coding; Dubois sufficient
  statistic; indirect/CEO + f-separable RD; RAG/Freeman boundary coding; deep active contours; SE(3)
  ego-motion geometry.
- **The genuinely novel intersection (originality bank):** applying the edge-centric tropical decomposition +
  grid-bulk/INR-annulus carrier to an **evaluator-inverse paint problem against a FROZEN scorer with a
  payload-only rule-118 archive and an argmax-0-1 + √-pose distortion** — this combination is not in the
  surveyed literature. Real only when a byte-closed `upstream/evaluate.py` n600 row beats 0.19110.

## 8. FINAL STATE
Design/research only; $0; n600 evidence where cited; run-1 (pid 63069) + all run dirs UNTOUCHED; NO launch.
**Pointer 0.19110 UNMOVED — MEANS.** Triality legs: DAG FEED (owed on next append) · 3 candidate equations
(council-flagged, anchors owed) · no DSL change (research only). Sources listed inline; primary arXiv IDs:
2002.08838 · 2403.11871 · 2601.09775 · 2605.28983 · 2106.10800 · 1505.04875 · 2504.02216 · 2408.07028 ·
2409.19660 · 2205.01874 · 2506.11139 · 2103.03123 · 2605.02726 · 2605.20672 · 1006.3679 · 1909.13359 ·
2112.03451 · 2309.01296 · 1404.1043 · 2111.04960 · Candès-Donoho CPAM 2004 · Berger CEO 1996.
