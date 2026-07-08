# PAPERS-CHECKED: arXiv 2504.05654 — "Curved representational Bregman divergences and their applications" (Frank Nielsen)

**Verdict: REFERENCE (score axes: NOT-APPLICABLE at the instance/formulation level; information-geometry framing: one durable ledger line). NO LEVER. Pointer contest-CPU 0.19110 UNMOVED — this memo is research/means only, no score claim.**

STORES CONSULTED (proactive recall): MEMORY L10 (GR unified action, Fisher metric), L24 (calibrate
parametrization = frozen-scorer Fisher geometry), L75 (deepmath "Amortizing the Argmax" ch.1-6:
separatrix = Laguerre = tropical = caustic = curvelet; **τ=ε=ħ**; Maslov err ≤ τ·ln5), L1/L17 (d_seg
residual = separatrix PLACEMENT), papers-checked ledger L55, `pose_solve_output_space_inverse_20260708.md`,
`papers_checked_ttd_functional_tensor_train_20260708.md` (memo format + negative-verdict rule). Existing
in-tree Bregman/Fisher surfaces GREPPED (not re-derived): `src/tac/composition/frontier_primitives.py`
(`bregman_barycenter`, `BregmanDivergence` Literal), `src/tac/optimization/iglt.py` (Fisher-metric Langevin
preconditioner). Axis of every paper number below: **MEASURED(paper) = NONE** — the paper is theoretical.

---

## 1. What the paper actually is

- **Author / venue**: Frank Nielsen (single author). Subjects cs.IT + cs.LG. Submitted 8 Apr 2025;
  **v5 = 27 Mar 2026** (current), 33 pp / 11 figures. **No code repository** (checked abs + PDF).
- **Problem**: pure information geometry. Generalize Bregman divergences to a **curved** (non-affine
  level-set) family by composing a **representation/embedding function φ** with a flat convex generator F:
  `D(p‖q) = F(φ(p)) − F(φ(q)) − ⟨∇F(φ(q)), φ(p)−φ(q)⟩`. "Curved" because the level sets are non-affine.
- **Main theorems** (all PROVEN, none MEASURED):
  1. **Barycenter = right Bregman projection onto a non-affine subspace**: embed points → arithmetic
     mean `m̄ = (1/n)Σφ(pᵢ)` → project onto `φ(P)` → invert `φ⁻¹`.
  2. **α-divergences are representational curved Bregman divergences** under power-law α-embeddings
     `φ_α(p)=p^α`; the induced Riemannian metric **recovers the Fisher metric as α→1**.
  3. **Intersection of α-divergence spheres** (level sets `{q : D_α(p‖q)=r}`) reduces to intersecting
     curved submanifolds via the same embed→project→invert machinery. (Outlined; no explicit algorithm
     / complexity.)
  4. Illustrative applications: symmetrized/pointwise Bregman variants, KL for circular complex normals.
- **Measured content**: **NONE.** Figures 5–8 are illustrative 2D-Gaussian level-set/barycenter
  visualizations. No runtime benchmarks, no error curves, no baseline comparison. Author's own claims
  ("geometrically richer", "computational advantages") are qualitative and un-benchmarked. This is a
  clean, correct unification result — MEASURED-experiment count = 0.

---

## 2. Facet-by-facet mapping to our stack

| Facet | Does the paper's math/mechanism apply? | Exact surface it would touch | Verdict |
|---|---|---|---|
| **d_seg / separatrix placement** | The SegNet 5-class **argmax** is the **tropical (T→0) limit**; the separatrix is the polyhedral tie-locus. The paper's α-divergence geometry is the **smooth, finite-temperature** (T>0) object on the probability simplex. Our binding regime is the zero-temperature limit where the correct geometry is **Laguerre/tropical/caustic** — already in-tree. The paper adds no zero-T actuator. | `src/tac/boundary_math/{laguerre_logit_offset,contour_codec,partition}.py`; `deepmath_amortizing_argmax_laws_20260704.py` | **NOT-APPLICABLE** (wrong temperature regime; our Laguerre framing already occupies the correct T→0 limit) |
| **pose / output-space solve** | Pose is a deterministic 6-dim **MSE** in PoseNet output space (LM solve through the eval roundtrip). It is not a probabilistic divergence minimization on a statistical manifold; the paper's Gaussian/circular-complex-normal KL examples do not attach. | `pose_solve_output_space_inverse_20260708.md`; `xi_pose_coder.py` | **NOT-APPLICABLE** (MSE, not a divergence) |
| **rate / INR-weight-coding** | The paper is about barycenters/projections/sphere-intersection, not entropy coders or weight quantization. No R(D) or coding contribution. | (none) | **NOT-APPLICABLE** |
| **curriculum / annealing** | Loose analogy only: the α-family (α: 0→1) is a *geometry* interpolation; our curriculum is a *temperature* (τ) anneal (CE→tau_softplus). No measured mapping, and α→1→Fisher is not an actuator we can schedule against d_seg. | `witness_dsl/schedule_readback.py`; `curriculum_dsl.py` | **NOT-APPLICABLE** (analogy, not a lever) |
| **architecture / basis** | No representational-basis contribution to a coordinate-INR. | (none) | **NOT-APPLICABLE** |
| **apparatus / measurement (Fisher framing)** | The α→1→Fisher theorem **confirms** (does not extend) our MEASURED anchor: margin field ↔ Fisher curvature Pearson **0.978** (MEASURED ours). The barycenter=projection theorem **generalizes an existing helper** we already ship. | `src/tac/composition/frontier_primitives.py::bregman_barycenter` + `BregmanDivergence`; `src/tac/optimization/iglt.py` (Fisher preconditioner) | **REFERENCE** (durable framing; no new number) |

The only real touch-point is the last row: **we already have `bregman_barycenter`** (flat divergences:
squared-Euclidean etc.) as a deterministic Pareto/tensor-mixing helper, and a Fisher-metric Langevin
preconditioner in `iglt.py`. Nielsen's result is the *curved/α generalization* of that barycenter. But
that composition surface is a **means helper (Pareto mixing / frontier primitives)**, not a byte-closed
score actuator — generalizing it to curved divergences moves no evaluator cell.

---

## 3. Verdict + the one durable ledger line

**REFERENCE.** One idea worth the ledger, nothing more:

> α-embeddings `φ_α(p)=p^α` express α-divergences as *curved* representational Bregman divergences whose
> induced metric → Fisher as α→1, with barycenters computable as embed→mean→flat-Bregman-project→invert
> (Nielsen 2504.05654, theory-only, no code, no benchmarks). This is the **smooth-temperature complement**
> to our tropical (T→0) Laguerre argmax framing and the curved generalization of our existing
> `frontier_primitives.bregman_barycenter`. It CONFIRMS (does not extend) our MEASURED margin↔Fisher
> 0.978 anchor. Not a lever: our d_seg binding lives at the zero-temperature argmax limit.

**No LEVER → no $0 probe spec, no DSL Lever-factory obligation, no DAG FEED owed.** (Per the
negative/reference-verdict registration rule, `[no-triality]`.) If a future unit ever wants a
**divergence-based soft surrogate** for the SegNet simplex OR a **curved class-prototype barycenter** for
v8 per-class reconciliation (`SPEC_v8_perclass_decomposition`, merge→diff→correct), this memo is the
pointer to reach for — but both are speculative and dominated today by (a) our measured `tau_softplus`
surrogate (top-2 CE, bit-exact CE at τ=1) and (b) the already-designed v8 reconciliation. Neither is
opened by this paper.

---

## 4. Adversarial pass on my own read (what would make this wrong)

1. **"The separatrix IS an α-divergence sphere intersection — that's a closed-form lever for the contour
   codec."** Checked and rejected: the SegNet/witness decision boundary `{p_i = p_j}` is a **max-plus
   (tropical) tie-locus**, the T→0 limit. The paper's sphere-intersection machinery is the finite-T
   smooth object. Our `laguerre_logit_offset` + deepmath ch.1-6 (Maslov dequantization, τ=ε=ħ) already
   solve the boundary at the *correct* zero-temperature limit. Adopting the smooth α-machinery would be a
   step *away* from the argmax geometry, not toward it. Verdict survives.
2. **"α-divergence surrogate loss could beat tau_softplus near the boundary."** Possible in principle, but
   (i) the scorer is FROZEN — any divergence loss is another *proxy* to the discrete argmax, not the
   authority; (ii) `tau_softplus` is already the measured top-2 CE surrogate and is bit-exact CE at τ=1;
   (iii) no paper number predicts an effect size. This is at most a future *formulation* to try, not a
   lever this paper hands us. Scoped: the NOT-APPLICABLE verdict is at the **instance/formulation** level
   (this paper, as an actuator) — it does **not** kill "divergence surrogates" as a family; that family
   is simply not advanced by a theory paper with zero measured surrogate-loss experiments.
3. **Confound check vs our measured anchors**: none. α→1→Fisher is *consistent* with margin↔Fisher 0.978;
   the paper operates at smooth-T while our binding is tropical-T, so there is no contradiction and no
   over-claim to reconcile. The barycenter theorem generalizes `bregman_barycenter` without invalidating
   its current flat use.
4. **Wrong-ID risk**: confirmed the fetched ID 2504.05654 = Nielsen's curved-Bregman paper (abs +
   PDF agree; title, v5 2026-03-27, cs.IT/cs.LG, no repo). Not a mis-typed vision paper.

**Net**: a genuinely elegant information-geometry unification, but it is a REFERENCE for our
Fisher/Bregman *framing* surfaces, NOT a score-moving lever for d_seg, d_pose, or rate. The critical
path (v7.5 trunk / #205 endgame / pose output-space solve) is untouched. Pointer 0.19110 UNMOVED.
