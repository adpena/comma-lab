# Novel Contributions and Originality Accounting

*Draft section, 2026-07-10. This is the NO-FAKE-#7 "borrowed-substrate accounting" applied to the
information-geometric foundations (`information_geometric_foundations.md`) and the task-space witness
architecture: an itemized separation of what we **apply** (classical mathematics and prior substrate) from
what is genuinely **ours**, with every ours-original claim tagged **[MEASURED]** (an n600 through-R number we
produced), **[DERIVED]** (a framing/identification we argue), or **[BUILT]** (an artifact in the repository).*

**Status — this is a publication deliverable, not a score-mover.** The frontier pointer is **0.19108282
[contest-CPU]** and this section does not move it. The originality below is the originality of a **design +
framing + measurement trail**; per THE GOAL, a novel design that has not yet lowered the exact score is
MEANS, not ends. The information geometry is classical; our contribution is the *application* to a
frozen-scorer indirect-rate–distortion codec plus the measured correspondence trail that confirms it. None of
this is an achievement until a byte-closed exact row below 0.19108282 proves the witness composition works —
where a claim below is not yet defensible as ours-and-measured/built, it is downgraded here rather than
padded.

---

## Ledger 1 — CLASSICAL / APPLIED (NOT ours: we apply, cite, and do not claim)

Each row is prior art or a competitor's substrate. We use it; we did not invent it. Naming it precisely is
the point of this ledger.

### 1a. The information-geometry corpus (classical mathematics)

| Result (source) | What it is | How we USE it (we do not invent it) |
|---|---|---|
| **Amari dually-flat spaces + generalized Pythagorean theorem** (Thm 6.12; *Information Geometry and Its Applications*, 2016) | A manifold with a pair of flat torsion-free (e/m) connections; canonical Bregman divergence splits `D[P:Q]=D[P:R]+D[R:Q]` when the legs meet dual-orthogonally. | We identify the frozen SegNet softmax as such a space (θ=logits, `F=logsumexp`, η=probs) and invoke the theorem to *justify* reverse-waterfill additivity and Dykstra convergence — and to *name the error* (the cross-term) when components are not dual-orthogonal. |
| **Chentsov uniqueness theorem** (*Statistical Decision Rules*, 1982) | The Fisher metric is the unique Riemannian metric (up to scale) invariant under Markov morphisms / sufficient-statistic reductions. | We invoke it to argue the Fisher/margin metric is *forced* (not an engineering preference) for a sufficient-statistic codec; the argmax is such a reduction, RGB-L² is not invariant under it. |
| **Nielsen, curved (representational) Bregman divergences** (arXiv 2504.05654) | Bregman divergence of a Legendre generator restricted to a non-affine `k<m` subspace; barycenter = right Bregman projection onto that subspace (his Thm 1). | We name the witness's `d_seg` distortion a curved Bregman divergence of `logsumexp`, and our head-offset/feasibility solves as its Bregman projection. Applied, not derived by us. |
| **Nielsen, Fisher = log-likelihood curvature at the argmax** (Entropy 2020); **Many Faces of Information Geometry** (Notices AMS 2022) | Fisher information = Hessian `∇²F` of the log-partition = curvature of the log-posterior peak. | The classical fact our whole distortion picture rests on; we *measure* the correspondence (Ledger 2), we do not prove the theorem. |
| **Nielsen, information-radius / JS centroid** (arXiv 2102.09728); **Fisher-Rao distance bounds** (2403.10089); **SPD-cone / Hilbert-projective distances** (2307.10644); **Bregman chord divergence** (Nielsen–Nock 1810.09113) | Divergence-correct centroids, tight Fisher-Rao bounds when the metric is Hessian, cone metrics on PSD matrices, gradient-free Bregman variants. | Held as **gated design principles** (correct codebook centroid for a future VQ-witness; distance-to-flip refinement; gradient-free finisher distortion). Cited as available tools; none yet built or claimed. |
| **Plus-Gourdon & Nielsen, Hessian-preconditioned Legendre–Fenchel** (arXiv 2606.09077) | Affine-deform to the canonical paraboloid, solve the well-conditioned residual, undo — the standard fix for anisotropic-Hessian Newton. | A **gated** solver candidate for our anisotropic annulus. The method is theirs; our contribution (if it lands) is only the measured check that it beats our current damped-Newton. Not yet built → not claimed. |
| **Goldman, geometric (G,X)-structures** (*Geometric Structures on Manifolds*, 2021) | Locally-homogeneous atlas with developing map + holonomy homomorphism; affine geometry as a flat (G,X)-structure. | We use it to name Amari's dually-flat space as an affine (G,X)-structure and the ground-frame construction as a developing map with `G=SE(3)`; the language is Goldman's. |
| **Li, transport information geometry** (2021) | The Wasserstein information metric is distinct from Fisher. | We invoke it to *distinguish* our two geometries (Fisher for `d_seg`, Wasserstein for the warp/generator) — a caveat we honor, not a result we own. |

### 1b. Substrate and domain-prior lineage (competitors' and upstream work)

| Source | What it is | How we USE it |
|---|---|---|
| **PR95 / HNeRV substrate lineage** (comma challenge PR95/100/101/103; HNeRV) | The winning HNeRV-family decoder + 8-stage curriculum + FP4/Brotli archive grammar + score-aware training that established the 0.19-band. | Studied as the incumbent; the witness is an explicit **pivot off** it (§ CLAUDE.md WITNESS CAPSTONE), not a reskin. Any element we reuse is credited; a PR95-curriculum-on-HNeRV run bolted with one lever would be a NO-FAKE-#7 fake of our own capstone, and is forbidden. |
| **openpilot / comma world-model geometric priors** | Ego-motion, lane-polynomial, and homography priors from the production driving stack; deterministic and free under rule-118. | Used as the free generic-algorithm prior for the screw-warp and lane band; the priors are theirs, the repurposing-as-codec is ours (Ledger 2). |
| **Quantizr PR55 `JointFrameGenerator` paradigm** | FiLM-conditioned depthwise-separable renderer, half-frame mask economy, KL-T=2 distill, FP4 packing. | Historical revelation-of-unknown-unknowns for the whole task-aware-rendering direction; explicitly acknowledged (see `EXTERNAL_SOURCE_ATTRIBUTION_C067.md`). Not part of the witness's counted payload. |
| **UNIWARD / inverse-steganalysis framing** (Fridrich–Yousfi lineage) | Cost of an embedding change is lowest in textured regions; the challenge is "inverse steganalysis." | An **on-ramp framing only**: it motivated the margin/texture reading of where flips hide. The load-bearing metric is our measured Fisher/margin geometry, not UNIWARD cost; we do not claim the steganalysis result. |
| **v2 codec borrowed primitives** (full list: `project_v2_novel_contribution_originality_accounting_20260629.md`) | Flow-matching generators (LieFlow, GNVC-VD, OT-NFM); conditional-VQ-VAE Wyner–Ziv binning (Whang; Özyılkan–Ballé); the warp-coords+residual INR skeleton (**INVC, arXiv 2112.11312** — our literal skeleton); the canonicalization framework (Kaba; Dym; frame-averaging); SDF-survives-bilinear (Valve/Green; msdfgen); difference-imaging residual (Alard–Lupton; ZOGY); factorized-field lit (TensoRF, K-Planes, Tensor4D, DeepSDF, Nerfies SE(3)-field); ID estimators (TwoNN, MLE-ID). | Every individual primitive of the witness codec exists in prior art and is credited in the full ledger. The composition is ours (Ledger 2); the parts are not. |

---

## Ledger 2 — OURS-ORIGINAL (the contribution; MEASURED / DERIVED / BUILT)

Each row is defensible as ours. The tag states *how*: **[MEASURED]** = an n600 through-R number we produced;
**[DERIVED]** = a framing/identification we argue from primary artifacts; **[BUILT]** = an artifact in the
repository. Borrowed numbers are never presented here as ours.

### 2a. The architecture and its objective

| Contribution | Tag | Evidence / anchor |
|---|---|---|
| **The task-space witness: amortize the frozen SegNet argmax partition as a non-RGB codec** — code the task-sufficient statistic (argmax partition + ego-pose), not full RGB. | **[BUILT]** | The witness trainer + level-set flow; `src/tac/boundary_math/*`, `experiments/train_levelset_witness_realized_through_R_mlx.py`. The *design* is ours; components (Ledger 1b) are borrowed. |
| **Indirect rate–distortion task-floor framing for THIS problem** — the comma scorer as a CEO/remote-source (Berger–Yeung) indirect-RD instance with a measured task-floor `S_floor≈0.118`. | **[DERIVED]** | The indirect-RD reduction and floor derivation are ours for this scorer; the indirect-RD theory is classical (cited). |
| **The stratified screw-warped level-set factorization (S²WL)** and the MDL-canonicalization gauge (canonicalize for *rate* over a task-equivalence class, not for downstream accuracy). | **[DERIVED] / [BUILT]** | `project_v2_novel_contribution_originality_accounting_20260629.md` (the 5 ours-elements; gauge-for-rate "appears genuinely new" vs the equivariant-ML canonicalization literature). Design-original, UNVALIDATED until measured. |

### 2b. The measured correspondence trail — the frozen-scorer distortion IS this geometry

This is the empirical heart of the contribution: not the geometry (classical), but the *recognition and
measurement* that the contest's frozen-scorer distortion realizes it. Every number is ours, produced n600
through-R.

| Measured correspondence | Tag | Value |
|---|---|---|
| Fisher curvature of the SegNet head ↔ class margin (top-two logit gap) | **[MEASURED]** | Pearson **0.978** → the margin field is the first-order Fisher surrogate |
| Boundary logit landscape is quadratic (the Fisher = Hessian of `logsumexp`, realized) | **[MEASURED]** | Levenberg–Marquardt goodness-of-fit ρ ≈ **0.85** |
| `d_seg` error localizes to a small-Fisher boundary annulus; it is argmax *jitter*, not region misclassification | **[MEASURED]** | ~**97%** of `d_seg` in ~**4.7%** of frame area |
| The witness rides an intrinsically low-dimensional non-affine (curved-Bregman) submanifold | **[MEASURED]** | nonlinear intrinsic dim ≈ **8–9** (TwoNN 9.79 / MLE 9.27; linear PR overcounts) |
| One stored se(3) twist `ξ` warps the partition (`d_seg`) *and* reads out the pose (`d_pose`) — the dual-use screw is the chart's holonomy | **[MEASURED] + [DERIVED]** | measured dual-use; the identification "= holonomy of the affine (G,X)-structure" is our framing |
| Fisher-optimal ≠ Wasserstein-optimal on this manifold (the two-geometry caveat, empirically forced) | **[MEASURED]** | an OT area-mass-match move *hurt* `d_seg` → the objective is flip-weighted, not mass-matched |

The contribution in 2b is the act of *converting separately-discovered facts (Fisher=margin, the quadratic
head, the boundary annulus, the dual-use screw) into one object* — a curved submanifold of a dually-flat
affine (G,X)-structured statistical manifold — and cross-validating each against an independent classical
name. The naming is applied; the measurements and the unification are ours.

### 2c. The apparatus (built infrastructure, ours)

| Contribution | Tag | Anchor |
|---|---|---|
| The specific levers, custom MLX/Metal kernels (fused diff-R, AA-SDF raster, margin map, curvelet), and witness curriculum | **[BUILT]** | the levelset trainer + `src/tac/witness_dsl/*`; each kernel bit-identical to the numpy-fp32 authority |
| **The triality campaign apparatus** — DAG ↔ DSL ↔ canonical-equations as three consistent views of one campaign, with a per-leg drift detector | **[BUILT]** | `tac.witness_dsl`, `tac.canonical_equations`, `tools/triality_drift_detector.py`; `docs/triality_dag_dsl_equations_deepmath.md` |
| **The costate controller** — a marginal-ΔS (λ) sense-organ that ranks never-fired levers and surfaces a duty-to-measure queue | **[BUILT]** | `tools/costate_digest.py` + the activation ledger; advisory-only actuation boundary |

### Honest downgrades (claims that did NOT make Ledger 2)

- **The Hessian-preconditioned solver, the JS/information-radius codebook centroid, the Fisher-Rao
  distance-to-flip, the Bregman-power-diagram v8 generator** are *applications of others' methods*, each gated
  on a $0 measured check that it beats the current approach. Until built-and-measured they are prior art we
  *may* apply — not ours-original contributions.
- **The information-geometric recognition itself** is a **justification, not a result** (per
  `information_geometric_foundations.md` §9). It gives our measured geometry canonical names and a
  cross-validation trail; it does not by itself lower the score. Presenting it as an achievement would be
  narrating means as ends.
- **Every codec-composition element** (S²WL, the 5 ours-elements, the gauge-for-rate) is an **UNVALIDATED
  design-originality claim** until a byte-closed exact row < 0.19108282 demonstrates the composition works.

---

## Ledger 3 — The contribution statement (for the paper, verbatim)

> The information geometry we invoke is entirely classical. Amari's dually-flat spaces and generalized
> Pythagorean theorem, Chentsov's uniqueness of the Fisher metric, Nielsen's curved-Bregman divergences and
> Legendre–Fenchel machinery, and Goldman's (G,X)-structures are prior mathematics that we apply and cite; we
> claim none of them. Our substrate lineage — the HNeRV/PR95 family, the openpilot geometric priors, the
> Quantizr rendering paradigm, and the UNIWARD/inverse-steganalysis framing — is likewise prior work that we
> build against and credit. What is new is the *application*: we recognize that the comma challenge's
> frozen-scorer distortion is not a pixel-fidelity problem but an **indirect rate–distortion** problem whose
> natural distortion geometry is the scorer's own dually-flat statistical manifold, and we build a **task-space
> witness codec** that spends its bit budget on that scorer-relevant curved submanifold — the argmax partition
> and the ego-pose — rather than on RGB the scorer is invariant to.
>
> The evidence that this is the *correct* geometry, and not a post-hoc metaphor, is an original measurement
> trail produced entirely on our vehicle: the Fisher curvature of the SegNet head correlates with the class
> margin at Pearson 0.978; the boundary logit landscape is quadratic (LM ρ ≈ 0.85), realizing the
> Fisher-equals-Hessian-of-`logsumexp` identity; ~97% of the segmentation distortion localizes to a ~4.7%-area
> small-Fisher boundary annulus as argmax jitter; the witness rides an ≈8-dimensional non-affine (curved-Bregman)
> submanifold; and a single se(3) twist serves as the holonomy that jointly warps the partition and yields the
> pose. Converting these separately-discovered facts into one geometric object, and cross-validating each
> against an independent classical name, is the contribution — together with the built apparatus (the witness
> trainer and its kernels, the triality campaign instrument, the costate controller) and the framing of gauge
> canonicalization for minimum *rate* rather than downstream accuracy, which appears genuinely new.
>
> Positioned against prior art, the honest claim is a **novel composition, not a new primitive**. The closest
> single neighbor is Implicit Neural Video Compression (arXiv 2112.11312) — our literal warp-the-coordinates +
> small-residual skeleton — but it optimizes pixel fidelity with a learned warp and has no task-space objective
> and no scorer geometry. The broader coding-for-machines / VCM literature carries the right (task) objective
> but with black-box decoders and no geometric prior; the driving-scene-reconstruction-with-warp literature
> carries the physical ego-motion warp but reconstructs RGB for the human eye. The task-space witness occupies
> the unoccupied intersection: the *physical ego-screw* as the temporal codec of a *driving* scene's
> *frozen-scorer task manifold*, with distortion measured in the scorer's own Fisher geometry. The geometry is
> classical; the witness, the measured correspondence, and the task-space indirect-RD codec are ours — and they
> remain a means, unvalidated as a result, until the exact contest score falls below 0.19108282.

---

## Honest scope (binding)

This document is the originality *accounting*, not an originality *result*. The pointer is UNMOVED at
0.19108282 [contest-CPU]; nothing here is a score claim. The classical geometry is framing (never a number);
the exact through-R measurement remains the sole authority for any score claim; and the composition's novelty
is DEMONSTRATED only when a byte-closed exact row below 0.19108282 proves it. Update triggers: (a) the
lane-survival exact run (which of the ours-elements are validated or falsified by the measured row); (b) any
gated info-geometry solver that passes its $0 measured check and is built (it then moves from Ledger 1 "applied"
to Ledger 2 "built"). Cross-references: `docs/paper/information_geometric_foundations.md`;
`project_v2_novel_contribution_originality_accounting_20260629.md`;
`.omx/research/v2_originality_provenance_synergy_citations_20260629.md`;
`.omx/research/intake_curved_bregman_and_geometric_structures_cluster_20260710.md`; CLAUDE.md NO-FAKE #7 +
the honest-attribution discipline.
