# SANDBOX — "why is a curvature polynomial secretly a topological invariant?" and where the same music shows up in a driving-video codec

<!-- review_status: recovery-committed 2026-07-08 (hardening sweep): found untracked with its
     companion dashboard SANDBOX tab (tools/dashboard_server.py); author agent apparently
     credit-died before committing. Content complete + honestly tagged on sanity read;
     UNREVIEWED by fresh eyes beyond that read. -->

A playground doc: the context behind a public reply to
[@nihilunbounded's Pontryagin-classes post](https://x.com/nihilunbounded) —
*"there's no apriori reason some random polynomial [of the curvature] should be a
homotopy invariant. You're defining p_i(M) using the smooth structure on M!"*

Everything here is labeled **MEASURED / DERIVED / BUILT** or **ANALOGY**. The
resonance with Pontryagin classes is a genuine structural rhyme and an actual
characteristic class (Maslov) in our geometry — **not** a claim that we computed
Pontryagin classes or detected exotic spheres. (Honest-attribution discipline:
describe the measured/derived math, never the wished-for version.)

---

## 1. The setup — what he's marveling at

The Pontryagin class `p_i(M)` is built from a curvature polynomial — you pick a
Riemannian metric / connection, form a specific `Ad`-invariant polynomial in the
curvature 2-form, integrate. Every ingredient depends on the **smooth structure**.
And yet the answer is a **topological / homotopy invariant** — independent of the
metric you used. Milnor (1956) then used *integral* Pontryagin classes + the
Hirzebruch signature theorem to detect **exotic spheres** (smooth structures on
S⁷ that are homeomorphic but not diffeomorphic). The puzzle: *why should a
construction that looks like it depends on all the smooth detail be secretly
invariant?*

## 2. The Lie answer to "why a polynomial" (the part he's circling)

This is Lie theory hiding in plain sight — **Chern–Weil theory**. The invariant
polynomials aren't random: they're exactly the **`Ad`-invariant polynomials on
the Lie algebra 𝔤**, and the **Weil homomorphism**

```
Sym(𝔤*)^G  ──→  H*(BG; ℝ)
```

sends each such invariant polynomial to a characteristic class. You're not
evaluating "some polynomial" — you're evaluating an element of `Sym(𝔤*)^G`, the
`G`-invariants of the symmetric algebra, on the curvature. The smooth/metric
choices cancel **by construction** because the class is the image of a
`G`-invariant, i.e. an invariant of the *bundle*, not of the connection. That is
the whole miracle, and it is a statement about a **Lie group and its Lie
algebra**. (We did a group-theory deep-math review on exactly this territory:
`.omx/research/group_theory_deepmath_review_20260707.md`.)

## 3. The domain and the oracle (auth-eval scorer)

Our problem is the **comma.ai video-compression challenge** — task-aware
("coding-for-machines") compression of openpilot driving video. The score is
computed by a **frozen oracle**, and *only three things carry authority*:

- **SegNet** (comma10k EfficientNet-B2): per-pixel 5-class **argmax** on the last
  frame → **`d_seg`** = argmax disagreement rate vs the source.
- **PoseNet** (FastViT-T12): two-frame YUV6 → 6-vector; **`d_pose`** = MSE on the
  first 6 outputs.
- **archive bytes** → the rate term.

```
S = 100·d_seg + √(10·d_pose) + 25·|archive.zip| / 37_545_489
```

Human visual fidelity is non-authority. Only the argmax partition, the pose
6-vector, and the byte count matter. (Contest spec — the scoring *definition*,
verified in-tree against `upstream/evaluate.py`; not a measurement.)

## 4. Our vehicle — a task-space level-set witness

Instead of reconstructing RGB, we train a **coordinate-INR** (an implicit neural
field over pixel coordinates) that **amortizes the SegNet argmax partition
directly** — a scorer-only witness spending its bytes on the *scorer-relevant
manifold*, not on full RGB. Geometrically it is the **viscosity solution of a
variational level-set flow**: the object of interest is the **separatrix**, the
codim-1 boundary between argmax cells (the Morse–Smale separatrix / the SegNet
class boundary). BUILT: `experiments/train_levelset_witness_realized_through_R_mlx.py`,
`src/tac/boundary_math/`.

## 5. The rhyme — "parametrization-dependent-looking, secretly invariant"

Here is the same shape he's marveling at, in our codec (this half is
**structural / by-construction**, not wished-for):

**`d_seg` factors through the argmax partition.** It depends only on the argmax
**partition** of the frame, so it is *invariant under any change to the witness
weights θ that leaves the partition fixed*. Precisely — and this is where a
topologist looks hardest, so we state it cleanly — d_seg **factors through** the
argmax map (logits → partition) and is **constant on each fiber**, a relatively
open **polyhedral argmax cell** `{ f_winner(x) > f_others(x) }`. It is *not* a
group quotient (a partition is not a group acting on ℝᴺ); it is a level-set /
fiber structure:

```
d_seg : (image of the argmax map) → ℝ,   constant on each polyhedral cell
```

The score looks like it depends on all the smooth realization detail (the millions
of INR weights that draw the field) — but it is secretly a function of the
partition alone. Metric/parametrization-dependent-looking → secretly
invariant. That's the **same phenomenon** as the curvature polynomial being
secretly topological. (This is literally how the scorer is defined; it's why our
"task-space QUOTIENT codec" framing exists — code the orbit
`ℝᴺ/(argmax-polytope × pose-null)`, the task-sufficient statistic.)

## 6. The Lie spine runs through BOTH scored axes (this is the part the operator flagged)

The Lie thread isn't just in his math — it's load-bearing in ours:

- **Pose is se(3)-valued.** The ego-motion between two frames is a rigid screw:
  by **Chasles' theorem** every rigid displacement is a screw motion, one twist
  **ξ ∈ se(3)**, `exp(ξ) ∈ SE(3)`. The *same* ξ that transports the argmax
  partition (a `d_seg` prior) is the pose PoseNet measures (`d_pose`) — one
  6-vector serves both terms. Engine: **`src/tac/lie/se3.py`** (MLX + numpy),
  `src/tac/xray/posenet_se3_lie_algebra.py`, `tac.ego_xi_trajectory`. BUILT +
  MEASURED (advisory / through-R, macOS-CPU non-promotable — not exact-eval:
  openpilot-ego prior gives −94/−99% on the pose axis).
- **Conditioning via orthogonalized updates.** Our finishing optimizer (Muon)
  replaces the gradient with its **orthogonal factor** (Newton–Schulz) — steepest
  descent in the **spectral norm**. It orthogonalizes the *update*, not the
  weights (an orthogonal-group / Stiefel-flavored geometry; note the Stiefel
  manifold is generally not a group — only the square case is `O(n)`).
  (MEASURED, advisory/through-R: −32% total d_seg vs AdamW from a stage-4 fork.)
- **A Maslov class of the Lagrangian Grassmannian, identified with our boundary.**
  In our "Amortizing the Argmax" deep-math chapters we read the SegNet argmax as
  a **caustic** (Lagrangian singularity) of the softmax-as-ħ→0 limit, with the
  identification **τ = ε = ħ** and the sharp log-sum-exp bound
  `0 ≤ τ·logΣexp(f/τ) − max_k f_k ≤ τ·ln K` (K = 5 classes, so `≤ τ·ln 5`) — a
  real inequality, not a slogan. The **Maslov class is a characteristic class of
  the Lagrangian Grassmannian** `Λ(n) = U(n)/O(n)` — another Lie-theoretic class
  of a homogeneous space of Lie groups, but of a **different type**: the generator
  of `H¹(Λ(n); ℤ)`, a degree-1 class coming from `π₁` (via `det²: Λ(n) → S¹`),
  **not** a Chern–Weil invariant-polynomial class (those are even-degree). Our
  τ→0 caustic framework *identifies* this Grassmannian structure with the argmax
  boundary. DERIVED (theoretical framework):
  `.omx/research/deepmath_amortizing_argmax_paper_draft_20260704.md`,
  `.omx/research/deepmath_lens_microlocal_se3_code_20260704.md`,
  `src/tac/canonical_equations/deepmath_amortizing_argmax_laws_20260704.py`.
- **Tropical / Laguerre bonus.** In the τ→0 limit the witness *is* tropical — its
  cells are **Laguerre (power-diagram) cells**. Tropicalization is exactly
  "forget the smooth structure, keep the piecewise-linear / combinatorial
  skeleton" — the same flavor as extracting a topological invariant from smooth
  data. DERIVED (#284).

## 7. The honest boundary

- **Real:** the score-as-quotient-invariance (by construction), the se(3)/Chasles
  pose engine (BUILT + MEASURED), the Stiefel/Muon conditioning (MEASURED), and
  the Maslov/tropical framework (DERIVED theory).
- **Analogy, not identity:** we did **not** compute Pontryagin classes, do
  Chern–Weil on a tangent bundle, or detect exotic spheres. Our characteristic
  class is the **Maslov** class (Lagrangian), and our "invariance" is
  score-invariance-under-reparametrization. Same music, different theorem.

## 8. Links

**Codebase** (contest closed → IP open source): repo
[github.com/adpena/comma-lab](https://github.com/adpena/comma-lab). In-tree:
`src/tac/lie/se3.py` (se(3)/SE(3)), `src/tac/boundary_math/` (the witness),
`src/tac/canonical_equations/deepmath_amortizing_argmax_laws_20260704.py`,
`docs/triality_dag_dsl_equations_deepmath.md`. *(Exact blob URLs to confirm
against the public repo layout before posting.)*

## 9. SHOW-DON'T-TELL design spec (operator 2026-07-07: "communicate the gestalt to a genius with the same sense organs as us")

The tab must not be prose. It must *show*. Target audience: a mathematician who
thinks geometrically, likes **LaTeX**, **citations**, **WebGPU interactivity**,
and **manim**-grade animation — all bridged. The static section above is the v0
substrate (correct math + honest labels); the build turns each claim into a
manipulable/animated demonstration. Register: exhibit, not copy. Four interactive
panels, each = one visceral demonstration + its LaTeX + its citation + a link to
the exact in-tree module.

1. **The invariance demo (the centerpiece — his exact puzzle, made physical).**
   A live field over a grid: three class "logit" fields `f₀,f₁,f₂`, colored by
   `argmax`, with the separatrix drawn. A slider injects a big smooth per-pixel
   scalar `c(x)` added to **all** logits equally. LaTeX:
   `argmax_k [f_k(x) + c(x)] = argmax_k f_k(x)`. Watch: the field surface heaves
   dramatically (the "smooth structure" changes), the partition and the live
   **d_seg = 0.0000 (INVARIANT)** readout do **not** move. The direct physical
   analog of "change the metric, the Pontryagin number is fixed." WebGPU for the
   field surface; a second knob does a genuinely partition-changing perturbation
   so the contrast is felt. Cite: Chern–Weil / Weil homomorphism; Dubois (task
   quotient). Bridge: `src/tac/boundary_math/`.
2. **τ→0 caustic / Maslov.** Animate the softmax temperature τ = ε = ħ → 0;
   smooth logits sharpen into the caustic (separatrix); overlay the Maslov index
   counting boundary crossings; error bar `≤ τ·ln 5`. manim-style. Cite: Arnold
   (Maslov index); Guillemin–Sternberg (geometric asymptotics). Bridge:
   `deepmath_amortizing_argmax_laws_20260704.py`.
3. **The se(3) screw, dual-use.** Interactive ego-ξ twist: drag the 6 twist
   components, watch the SAME `exp(ξ) ∈ SE(3)` both warp the partition (d_seg
   prior) and move the pose 6-vector (d_pose). LaTeX: the twist/Chasles form.
   Cite: Chasles; Murray–Li–Sastry. Bridge: `src/tac/lie/se3.py`.
4. **Tropical / Laguerre.** As τ→0 the cells become a Laguerre (power-diagram)
   tessellation; drag the weights, watch the piecewise-linear skeleton. LaTeX:
   the tropical/max-plus form. Cite: **Aurenhammer, *Power diagrams* (1987)** +
   Maclagan–Sturmfels, *Tropical Geometry* (2015). (NOT curvelets — those belong
   to the RATE/representation story, §6, not the tessellation panel.) Bridge:
   the τ→0 witness.

### Design-review pivot (2026-07-07, applied): homotopy centerpiece + ship 1+3

The reviewer's decisive upgrade: the additive `c(x)` shift is *freshman-obvious*
invariance and under-sells the rhyme. The **star demo is a homotopy of
realizations** — morph between two genuinely DIFFERENT smooth fields θ₀→θ₁ that
induce the **same** argmax partition; the surface reorganizes nontrivially, the
boundary + `d_seg = 0.0000` stay pinned the whole way. This is isomorphic to the
Chern–Weil proof itself (any two connections are joined by a path; the class is
constant along it; the difference is an exact transgression form
`P(Ω₁) − P(Ω₀) = d·TP(A₀,A₁)`). **The demo IS the proof structure, not an
illustration of it.** Three knobs: A = additive `c(x)` warm-up; B = the homotopy
(the star); C = a partition-CHANGING perturbation so `d_seg` visibly jumps off 0
(invariance is only felt against a break). **Ship panels 1 (invariance) + 3
(se(3) screw dual-use) FIRST**; 2 (Maslov) + 4 (tropical) are the scroll-reward.
Card title: *"Change the metric, the number stays. Change the field, the
partition stays."* Hero image: the invariance panel mid-heave, boundary unmoved,
`d_seg = 0.0000`. Opening 3 s: the demo autoplaying (field breathing, boundary
pinned) BEFORE any prose. No logo, no sub-0.15, no CTA — lead with the pure-math
demo; the codec is "where I met this in the wild," not the headline. Panel-1
canonical cite is **Chern–Simons, *Characteristic forms and geometric
invariants*, Ann. Math. 99 (1974)** (the transgression reference).

**Tech under CSP-free inline:** MathML (native, zero-dependency LaTeX) or inlined
KaTeX for prettier; inline `<script>` + WebGPU (already the WHY/HOW pattern);
manim animations pre-rendered to `<video>` (data-URI or served asset). Everything
self-contained. **Build gate: the recursive adversarial review of the math +
reply must pass BEFORE the heavy WebGPU/manim engineering** (per the operator's
review-before-each-step discipline).

## 10. Resources / prior art

**Resources / prior art**: Milnor, *On manifolds homeomorphic to the 7-sphere*
(1956); Chern–Weil theory / the Weil homomorphism; Arnold, the Maslov index;
Candès–Donoho, curvelets (the optimal sparse basis for a curved codim-1
singularity); Dubois et al., *Lossy Compression for Lossless Prediction*
(NeurIPS 2021, the task-space sufficient-statistic codec);
[comma.ai video compression challenge](https://github.com/commaai/comma_video_compression_challenge).
