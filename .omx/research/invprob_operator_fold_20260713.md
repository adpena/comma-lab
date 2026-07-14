# Inverse-problem operator fold onto the P0 backward attack and witness compiler

**Date:** 2026-07-13

**Role:** SOL xhigh deep-math reader/synthesizer

**Status:** COMPLETE DESIGN / MEANS; `$0`; no training, launch, scorer replay, live-run mutation, or live P0-arm edit

**Primary downstream:** `p0_sparse_adjoint` / `FEED-p0-backward-wave-20260713`

**Score authority:** `false`; pointer moved: `false`
**Verdict scope:** operator-theoretic grounding and construction only. Empirical low-rank, descent, timing, and score verdicts remain with real frozen-SegNet probes and eventual byte-closed exact evaluation.

## Executive verdict

1. **SegNet-adjoint PDO-like / low-rank:** **CONDITIONALLY PLAUSIBLE, NOT PROVED.** **SOURCED:** Ying proves the useful data-sparse structure for operators already known to be PDOs/FIOs. **DERIVED for our exact graph:** the frozen `tu-efficientnet_b2` U-Net linearization is a composition of local/pointwise maps plus 23 squeeze-excitation global-pooling corrections; their reduction bottlenecks sum to 626. This is real structural support for a local-plus-low-rank ansatz. It is not a theorem that the full input Jacobian is a PDO or that all wavelet off-diagonal blocks have bounded rank. The decisive spectra and held-out costate errors are deferred to `p0_sparse_adjoint`.
2. **Top transfer — BCR-style cheap adjoint:** build a **linear, state-local, classwise 2-D non-standard wavelet operator** for `J_Seg(x0)^T`, retain near-field blocks explicitly, factor admissible far blocks by randomized exact JVP/VJP probes, apply it to the current exact output covector, and fail closed outside a content-bound state/descent trust region. **DERIVED apply cost:**

   \[
   C_{\rm apply}=c_W(C_i+C_o)N+
   \sum_{\ell}N_\ell(c_s s_\ell r_\ell+c_r r_\ell^2).
   \]

   For a 2-D dyadic pyramid, `sum_l N_l < 4N/3`; if stencil widths and admitted ranks are bounded, apply and storage are `Theta(N)`. This is a constant-factor replacement of the full network reverse graph, **not** an `O(N^2)`-to-`O(N)` claim: exact CNN reverse mode is already `O(N)` for fixed architecture. Build cost and refresh cadence are charged by `T_apply + T_build/K`.
3. **Witness-as-inverse-map / n=1:** **NO-GO as a wholesale BCR/Switch replacement for the coordinate INR on present evidence; WORTH-KEEPING as an optimizer/preconditioner or compact field-to-field correction head.** Ying's applications use operator-correct architectures but still train on roughly 8K–16K paired fields. Deep Ray's generative priors explicitly require a sample set. Structure lowers sample complexity; it does not conjure operator identification from one video. Our one video does contain many correlated pixel/pair constraints, but the only admissible cure is a fixed analytic skeleton plus a small learned residual that wins exact archive bytes and evaluator cells.
4. **FIO = pose/temporal:** **THE GEOMETRIC IDENTIFICATION IS VALID; SWITCH-NET REPLACEMENT IS A SPEED NO-GO.** A visible-region depth-aware `SE(3)` image pullback is an FIO whose canonical relation is the cotangent lift of the warp. The ego-screw `xi` parameterizes that canonical transformation; it is **not itself** the homogeneous phase `Phi`. The current analytic bilinear warp applies in `Theta(N)` and is more faithful than an approximate quasi-linear butterfly for this local map. Switch/butterfly remains a scoped residual option only for multi-depth, occlusion, or genuinely nonlocal phase transport.
5. **Overall:** **PROCEED-TO-MEASURE for the state-local BCR adjoint; REFUTE the unconditional convergence claim.** The literature grounds the representation *conditional on operator-block ranks*. It does not transfer those ranks to a frozen CNN by analogy, and PDO pseudo-locality concerns singular support, not ordinary spatial support. Pointer movement remains possible only through a byte-closed exact row.

## Evidence language

- **MEASURED:** direct empirical value from a custody-bearing local artifact. No new measurement was run here.
- **DERIVED:** algebra or static source-graph result from named inputs.
- **INFERRED:** plausible transfer requiring empirical admission.
- **ASSUMED:** explicit condition used to expose a consequence; never promoted as fact.
- **SOURCED:** statement supported by the staged literature.

## Source extraction and custody

All four staged PDFs extracted cleanly with `pdftotext -layout`; no OCR or package install was needed. Page counts and hashes below refer to the operator-staged source bytes. Extracted text was inspection scratch only and is not a durable evidence path.

| Source | Extraction | Source-byte custody | Relevant content |
|---|---|---|---|
| Lexing Ying, *Solving Inverse Problems with Deep Learning* (ICM 2022), `lexing_icm.pdf` | **EXTRACTED CLEAN**, 22 PDF pages | 2,809,408 bytes; SHA-256 `1479352b198e890b3f4b0896de03d60785bcc28b009cee014aa565f776e65614` | PDF pp. 2–7: structure-guided inverse maps; PDO non-standard wavelet form; FIO complementary low rank, Hamiltonian flow, butterfly/Switch. PDF pp. 8–17: filtered-backprojection architectures and 8K–16K training-pair examples. |
| Philipp Holl, *Solving Forward and Inverse Problems with Differentiable Physics and Deep Learning* (TUM dissertation, 2024), `tum_diss.pdf` | **EXTRACTED CLEAN**, 158 PDF pages | 12,909,127 bytes; SHA-256 `7a50c1367b03798d25a80b5620d4e9834e4d07acb21e0bdd918a587c1167f909` | PDF p. 15: adjoint = reverse-mode differentiation. PDF pp. 36–43: joint inverse problems, ill-conditioning, physics inversion, half-inverse gradients. PDF pp. 73–82 and 117–120: scale-invariant physics and Jacobian half-inversion. |
| Deep Ray, *Deep Learning Approaches for Inverse Problems* (ICTS-TIFR talk, 2021), `deepray_invprob.pdf` | **EXTRACTED CLEAN**, 78 PDF pages | 5,562,327 bytes; SHA-256 `f320ed815bae2792099ed02938a2b95b0eacf2b13241f3d611896e93d40c78d2` | PDF pp. 33–45: ill-posed inverse maps, likelihood/prior/posterior, high-dimensional MCMC problem. PDF pp. 50–65: GAN prior/posterior constructions explicitly beginning from sample sets. |
| Elena Amosova and Kirill Kuznetsov, *Solution of Direct and Inverse Problems of Mathematical Physics Using a New Projection Method Physics Informed Neural Networks (PINN)* (PRIP 2025 B3.1), `prip_b31.pdf` | **EXTRACTED CLEAN**, 5 PDF pages | 167,249 bytes; SHA-256 `3376d0b10e060996b56815dd8810181da38c36dc9c672fa83d305bdbfed919b1` | PINN/EPINN for viscous heat-conducting gas control and retrospective nonlinear heat conduction; adaptive residual weights and anisotropic Fourier features. Relevant only as a limited n=1/PDE-residual comparison; not an adjoint-compression result. |

## Ranked fold table

| Rank | Source technique | Pact surface | Concrete transfer | Verdict |
|---:|---|---|---|---|
| 1 | Ying: PDO non-standard wavelet form with `O(n)` significant interactions; BCR-Net as its neural parameterization | Frozen-SegNet input costate `lambda = J_F(x)^T q`; diagnostic backward share | Linear state-local BCR operator for the real `5N -> 3N` adjoint; exact near field, factorized far blocks, charged refresh and exact fallback | **PROCEED-TO-MEASURE, CONDITIONAL.** Highest EV. Admission requires hierarchical block ranks and downstream descent, not vector sparsity alone. |
| 2 | Exact scorer graph: local convolution/upsampling plus squeeze-excitation global pooling | Why a CNN adjoint might be compressible at all | Use a local-plus-finite-rank prior: local multiscale kernels plus rank budget for SE global corrections; do not fit an unconstrained dense student | **DERIVED structural support.** Stronger than “CNN resembles a PDO,” but still not a global rank theorem. |
| 3 | Ying: filtered backprojection `K^*(K K^* + eps I)^{-1}` / ` (K^*K+eps I)^{-1}K^*` implemented by Switch+BCR | Ill-conditioned renderer-to-evaluator inverse step | Treat BCR as a training-time costate preconditioner or corrector after exact/approved adjoint, not automatically as a replacement | **DEFER.** Can improve direction conditioning; does not by itself remove the expensive teacher VJP. |
| 4 | Holl/TUM: physics inversion and half-inverse Jacobian gradients | Costate/renderer gradient conditioning | After an admitted cheap adjoint, precondition the evaluator-space or renderer-space update by a truncated half-inverse in the same multiscale basis | **DEFER / SECONDARY.** Potential convergence lever, not a P0 backward-cost lever unless its inversion is also cheap and exact descent is retained. |
| 5 | Ying: FIO Hamiltonian transport and butterfly/Switch factorization | `SE(3)` ego-screw temporal/pose warp; per-class/depth carriers | Keep analytic cotangent-lift warp; consider a sum of small phase-residual branches for occlusion/depth layers only | **NO-GO for replacing analytic warp; DEFER for residual nonlocal transport.** |
| 6 | Ying: operator structure as data-efficiency prior | Witness-as-inverse-map under one-video starvation | Hybrid fixed wavelet/butterfly skeleton plus tiny receiver-closed residual, only if it maps an actual compact input field to RGB/evaluator correction | **NEEDS-MORE.** Does not yet beat coordinate INR on archive bytes, receiver survival, or n=1 evidence. |
| 7 | Deep Ray: learned GAN prior/posterior; low-dimensional latent inference | n=1 witness prior | A pre-existing external prior could restrict witness solutions, but training a prior/posterior from this video is underidentified | **NO-GO for current n=1 lane.** The cited method assumes a sample set and often forward-model access. |
| 8 | PRIP B3.1: PINN/EPINN with Fourier features and adaptive residual weights | Coordinate INR and one-instance inverse optimization | No direct P0 transfer. At most, it independently supports anisotropic Fourier coordinates and balancing residual terms | **NO-GO as a new arm.** We lack a governing PDE residual equivalent to the frozen evaluator; the paper itself flags instability, resource cost, and missing nonlinear convergence theory. |

## 1. Does Ying's PDO result ground the sparse-adjoint arm?

### 1.1 Separate the three mathematical objects

Let `x in R^(3N)` be the last-frame SegNet input after the exact resize surface, `N = 384*512 = 196,608`, and let

\[
F(x)\in\mathbb{R}^{5N}
\]

be frozen SegNet logits. The evaluator uses

\[
d_{seg}(x,x_{gt})=N^{-1}\sum_p
1\{\arg\max_c F_c(x)_p\ne\arg\max_c F_c(x_{gt})_p\}.
\]

This exact term is discontinuous on logit-tie surfaces and has no ordinary VJP. Training uses a smooth, stage-dependent relaxation `L_seg(F(x), y)` such as CE/tau-family losses. Its exact output covector and input costate are

\[
q(x)=\nabla_F L_{seg}(F(x),y),\qquad
\lambda(x)=J_F(x)^T q(x).
\]

For ordinary weighted CE, `q_{c,p}` is proportional to `w_p(softmax(F_p)_c - 1[c=y_p])`; it is generally dense for finite logits. It can be *small* in confident interiors and large near low-margin pixels, but it is not mathematically supported only on the boundary.

**DERIVED correction to the proposed convergence:** the separatrix singularity belongs to `argmax o F`, not to the smooth logit map `F` itself. A frozen CNN Jacobian does not become singular merely because two output logits tie. Boundary concentration can ground a task-restricted covector family; it does not establish that `J_F` is a PDO.

### 1.2 What Ying actually supplies

**SOURCED, Ying PDF pp. 4–5:** a PDO has a smooth symbol away from zero frequency; in a local basis, off-diagonal matrix blocks are numerically low rank. The non-standard redundant wavelet form retains only `O(n)` significant interactions, and its forward/inverse wavelet transforms are linear-complexity. BCR-Net enriches this structured linear form with intermediate layers/nonlinearities for nonlinear operator learning.

**SOURCED, crucial nuance:** PDOs are pseudo-local: the singular support of `Kf` is contained in the singular support of `f`. This is not a statement that the ordinary value support of `Kf` is contained in the value support of `f`. Elliptic inverses can have global smooth tails. Therefore:

- boundary singularity preservation does **not** imply a spatially sparse input costate;
- low-rank off-diagonal operator blocks do **not** imply that a single output vector/image has low matrix rank;
- a compressible costate vector can still occur and can be useful, but it is a different empirical proposition.

### 1.3 Exact structure of our SegNet linearization

**DERIVED from `upstream/modules.py` and installed source introspection; no weights or forward replay:**

- `SegNet` is `smp.Unet('tu-efficientnet_b2', classes=5, activation=None)`.
- It reads only the last pair frame and bilinearly resizes to `384 x 512`.
- Static model size is 9,543,831 parameters; encoder feature channels are `[3,16,24,48,120,352]`; decoder channels are `[256,128,64,32,16]`.
- Convolutions, eval-mode batch normalization, SiLU/ReLU derivatives, pointwise channel mixing, nearest/bilinear interpolation, concatenation, and skip addition are local or pointwise at a fixed state.
- The EfficientNet-B2 encoder contains 23 squeeze-excitation (SE) modules. Their reduction bottlenecks are

  `[8,4,4,6,6,6,12,12,12,22,22,22,22,30,30,30,30,52,52,52,52,52,88]`,

  summing to 626.

For one SE block with `P` spatial sites,

\[
y_{c,p}=x_{c,p}g_c(s),\qquad
s_a=P^{-1}\sum_q x_{a,q}.
\]

At fixed `x`, its Jacobian is

\[
\frac{\partial y_{c,p}}{\partial x_{a,q}}
=\delta_{ca}\delta_{pq}g_c(s)
+x_{c,p}\frac{\partial g_c}{\partial s_a}\frac1P.
\]

The first term is diagonal/local. The second factors through the SE reduction bottleneck, so its global spatial correction has rank at most that bottleneck width. Composition and skip summation preserve a representation of the form

\[
J_F(x)=L_x+U_xV_x^T,
\]

where `L_x` is the composition/sum of local graph maps and the SE-induced global correction has the conservative architecture-level rank budget `rank(U_xV_x^T) <= 626`.

**Adversarial limit:** `L_x` is not necessarily a small-stencil operator at input resolution. Downsampling and many local layers can create a near-global effective receptive field, and input-dependent activation diagonals can be rough at semantic boundaries. The rank-626 result bounds one source of nonlocality; it does not prove low epsilon-rank for every off-diagonal block of the full `J_F(x)`.

### 1.4 Scoped verdict

The unconditional claim

> frozen SegNet is PDO-like, therefore its adjoint has low-rank off-diagonal blocks

is **REFUTED AS A THEOREM**.

The weaker claim

> the exact graph is local plus explicitly low-rank global couplings, while the task covectors are plausibly boundary concentrated; therefore a state-local wavelet/H-matrix approximation is a high-EV empirical arm

is **DERIVED + INFERRED and WORTH MEASURING**.

If the real hierarchical block ranks grow materially with block size or if held-out task-covector error/descent fails, the verdict is:

**NO-GO — verdict_scope:** state-local non-standard-wavelet BCR compression at the tested rank, state radius, loss regime, and hardware. It is not a verdict against all sparse/masked adjoints, exact graph surgery, alternate bases, or future content-conditioned operators.

## 2. Concrete BCR-style cheap-adjoint construction for our SegNet

### 2.1 Design target

Do not use the nonlinear BCR-Net of the generic learning setting for the actual VJP. At fixed `x0`, `J_F(x0)^T` is linear in `q`; adding ReLUs would violate linearity/oddness and create unnecessary n=1 parameters. Use the **linear non-standard wavelet form that motivates BCR-Net**.

Define orthonormal or biorthogonal 2-D transforms `W_o` classwise on the five logit-covector channels and `W_i` channelwise on the three RGB input-costate channels. The transformed adjoint is

\[
A_0=W_iJ_F(x_0)^TW_o^T.
\]

The provider stores a sparse/factorized approximation `A_BCR(x0)` and returns

\[
\widetilde\lambda(x;q)=W_i^T A_{BCR}(x_0)W_oq.
\]

The exact resize/R adjoint and renderer adjoint remain outside this provider and are applied normally. The construction replaces only the frozen-SegNet backward slice.

### 2.2 Build procedure

1. **Bind the state and objective.** Hash scorer bytes, preprocessing/resize definition, pair/frame, `x0` bytes, GT target, loss name and all stage parameters. A CE-built operator is not valid for a changed tau/focal objective without a new output-covector context.
2. **Choose the dyadic geometry.** Use a 2-D wavelet pyramid on `384 x 512`; partition each scale into spatial boxes. Mark receptive-near interactions as explicit sparse/locally connected blocks. Candidate far blocks are pairs separated relative to their scale.
3. **Probe the real operator matrix-free.** For each scale/block family, draw deterministic seeded Rademacher or Gaussian wavelet probes with oversampling. Apply exact `J_F(x0)^T` by VJP to obtain the left range; apply exact `J_F(x0)` by JVP to recover the compressed right factor. Reuse multiscale colored probes across compatible blocks. These exact probes are charged build work, not free labels.
4. **Fit rank adaptively, never by guessed constant.** For each admissible block, choose the smallest rank meeting a preregistered held-out operator-error target. Preserve the near field exactly/explicitly. The 23-SE / 626-bottleneck derivation is a rank-budget prior, not the selected rank.
5. **Emit NumPy-fp32 authority.** Serialize transforms, block partition, factors, state/objective hashes, ranks, residuals, seed, source hashes, and build timing. The portable reference applies the same ordered float32 operations. Torch/MLX implementations owe parity at or above the repository threshold and exact content binding.
6. **Run-time use.** Compute the exact current SegNet forward and current `q`; apply `A_BCR(x0)` without retaining the teacher backward graph; inject `stopgrad(lambda_tilde)` through the existing exact costate-injection identity into the renderer/witness backward.
7. **Trust and fallback.** Admit only when state radius, held-out random-probe residual, global and annulus costate metrics, renderer-gradient direction, and an exact-teacher shadow descent check pass. Any failure takes the full exact VJP. State radius alone is insufficient without a valid Jacobian-drift bound.
8. **Refresh/persist.** A future implementation must checkpoint the complete operator at each stage boundary, preserve prior stages, write atomically, and register under the canonical resume surface. The build is invalid if factors cannot be reloaded byte-close with their objective/state custody.

### 2.3 Derived cost

Let `N=HW`, `N_l` be the number of coefficients at scale `l`, `s_l` the explicit near-field width, and `r_l` the admitted far rank. A BCR-style three-layer block apply has the conservative work form

\[
C_{BCR}
=c_W(C_i+C_o)N
+\sum_{\ell=0}^{L}N_\ell
\left(c_s s_\ell r_\ell+c_r r_\ell^2\right).
\]

For a 2-D dyadic pyramid,

\[
\sum_{\ell=0}^{L}N_\ell
\le N\sum_{\ell\ge0}4^{-\ell}
=\frac43N.
\]

Thus, **ASSUMING** `sup_l s_l <= s_*` and `sup_l r_l <= r_*` independent of resolution,

\[
C_{BCR}=O\!\left(N[(C_i+C_o)+s_*r_*+r_*^2]\right)=O(N),
\]

with the same order for stored factors. If ranks grow with block size, this conclusion fails exactly where the live rank probe should fail it.

**Critical asymptotic correction:** reverse-mode autograd does not form the dense `3N x 5N` Jacobian. A fixed-depth convolutional U-Net VJP is also `O(N)` as resolution scales. The BCR value proposition is a much smaller network-depth/channel coefficient and removal of the teacher activation graph, not a change from quadratic to linear complexity.

Let `T_E` be exact frozen-SegNet backward time, `T_A` BCR apply time, `T_B` factor build time, and `K` the valid reuse horizon. The charged backward slice is

\[
\bar T_{BCR}(K)=T_A+T_B/K,
\]

and a strict timing win requires

\[
K>\frac{T_B}{T_E-T_A},\qquad T_A<T_E.
\]

The exact SegNet forward remains charged. On the existing diagnostic harness only:

- **MEASURED inputs:** 537 ms/pair forward and 3009 ms/pair forward+backward;
- **DERIVED:** `T_E = 2472 ms/pair`, or about 82.15% of that teacher slice;
- **DERIVED diagnostic ceiling:** eliminating backward at zero build/apply cost gives `3009/537 = 5.60x` teacher-slice speedup;
- **UNKNOWN:** in-loop split and whole-epoch speedup. The harness is about 12x heavier in absolute terms than the n600 in-loop accounting, so the 82% ratio remains unverified until the bounded D-A timer is run.

For a real BCR provider the diagnostic-slice expression is

\[
S_{teacher}(K)=
\frac{3009}{537+T_A+T_B/K},
\]

using diagnostic milliseconds only. No number may be substituted for `T_A`, `T_B`, or valid `K` without measurement.

### 2.4 Error and descent certificate

Suppose the fitted state-local operator satisfies

\[
\|J_F(x_0)^T-\widetilde J_F(x_0)^T\|_{op}\le\varepsilon
\]

and a content-bound state ball satisfies

\[
\|J_F(x)-J_F(x_0)\|_{op}\le L_J\|x-x_0\|,
\qquad \|x-x_0\|\le\rho.
\]

Then

\[
\|\lambda(x)-\widetilde\lambda(x;q)\|
\le(\varepsilon+L_J\rho)\|q\|.
\]

For renderer `x(theta)` with `B=J_{x(theta)}`,

\[
\|g_\theta-\widetilde g_\theta\|
\le\|B\|(\varepsilon+L_J\rho)\|q\|.
\]

A sufficient first-order descent condition is

\[
\|g_\theta-\widetilde g_\theta\|<\|g_\theta\|,
\]

because then `inner_product(g_theta, g_tilde_theta)>0`. This is a sufficient condition, not the only admissible empirical gate. In practice the arm should measure the actual renderer-gradient cosine and a fresh exact-teacher one-step delta; a high input-costate cosine alone can hide errors in sensitive renderer directions.

### 2.5 What `p0_sparse_adjoint` must distinguish

The live arm owns the empirical verdict. Four different ranks must not be conflated:

1. **Single-costate image rank:** SVD rank after reshaping `3 x H x W`. Coordinate dependent; can ground a low-rank image approximation only.
2. **Wavelet sparsity/compressibility:** sorted coefficient energy of one `lambda`. Can ground direct coefficient thresholding/masking.
3. **Ensemble rank:** covariance/SVD across real states or covectors. Can ground a task-subspace student.
4. **Hierarchical operator-block epsilon-rank:** rank of `P_I W_i J_F(x)^T W_o^T P_J` over held-out probes. This is the property that actually grounds Ying/BCR.

The current arm's planned concentration, spectra, and masked-adjoint curves are necessary. If they do not include randomized operator-block probes, the result can validate direct costate compression but cannot validate the PDO/BCR inference. Minimum downstream rows are:

- early / boundary / late real states and objective contexts;
- global and #333-annulus relative L2, cosine, sign, and tail mass;
- held-out task covectors plus structure-diagnostic random covectors;
- renderer-gradient cosine/norm and exact CE/tau one-step descent;
- exact realized-through-R d_seg non-worsening as a behavioral shadow, never as a differentiability claim;
- `T_build`, `T_apply`, exact fallback rate, valid `K`, and paired wall time.

## 3. Witness as inverse problem and the n=1 question

### 3.1 The valid correspondence

The contest compiler solves a constrained inverse problem:

\[
\min_{a,G}\quad
100d_{seg}(E(G(a)),E(v))
+\sqrt{10d_{pose}(E(G(a)),E(v))}
+25|a|/B_0,
\]

subject to legal archive/inflate behavior and deterministic receiver closure. Here `a` is counted payload and `G` is the legal decoder/generator. The inverse is highly nonunique: any witness inside the same evaluator cells is admissible.

Ying's design principle transfers cleanly: derive an architecture from the operator geometry instead of asking a generic network to learn it. This supports edge carriers, multiscale fields, analytic warps, and structured correctors.

### 3.2 Why BCR/Switch does not automatically cure n=1

**SOURCED:** Ying's showcased inverse-map networks use about 8K, 10K, or 16K paired examples depending on the application. The operator structure makes the networks compact and data-efficient relative to generic designs; it does not demonstrate learning a new inverse map from one instance.

**SOURCED:** Deep Ray's GAN-as-prior begins from a sample set `{x_i}` and GAN-as-posterior from paired `{(x_i,y_i)}`. It reduces latent dimension and Bayesian sampling cost only after that prior/posterior has been learned.

**SOURCED/DERIVED from TUM:** joint inverse optimization gains from patterns shared across several inverse problems; synthetic examples can supplement absent recordings. Pact's #434 adoption rule still requires real-input walk-forward evidence, so synthetic costates can initialize but not authorize.

Our `n=1` is not literally one scalar observation: the video provides 600 pairs and millions of spatial constraints. They are nevertheless strongly correlated, all from one scene/trajectory, and do not identify an arbitrary field-to-field operator family.

### 3.3 Architecture verdict against coordinate INR

A coordinate INR maps `(x,y,t,latent)` to RGB. A BCR/Switch module is most natural when an input **field** is mapped to an output field. Replacing the INR would therefore require storing or procedurally generating a compact input such as:

- edge/SDF/tropical carrier fields;
- pose/depth phase fields;
- evaluator residual/costate coefficients.

Then a fixed multiscale skeleton plus a small learned correction could be sensible. But every video-derived factor/weight is counted payload, and a field-to-field operator can easily cost more bytes than the existing compact coordinate representation.

**Verdict:**

- **NO-GO now:** “BCR-Net beats coordinate INR because operator structure cures n=1.” No exact archive-byte, receiver, or evaluator evidence supports that claim.
- **WORTH-KEEPING:** a fixed BCR preconditioner for training, or a tiny receiver-closed wavelet corrector from already-available compact edge/phase carriers to RGB residuals.
- **Admission:** exact archive bytes, parse-back survival, Seg/Pose deltas including the nonlinear pose term, and score units per byte. A lower training loss or fewer nominal parameters is not enough.

## 4. FIO, Hamiltonian flow, and the pose/temporal operator

### 4.1 Exact geometric identification

For a visible-region image diffeomorphism `kappa`, the pullback warp

\[
(T_\kappa f)(x)=a(x)f(\kappa^{-1}(x))
\]

has distribution kernel

\[
K(x,y)=a(x)\delta(y-\kappa^{-1}(x))
=a(x)\int e^{2\pi i(\kappa^{-1}(x)-y)\cdot\eta}\,d\eta.
\]

The phase is homogeneous of degree one in `eta`, so this is an FIO. Its canonical relation transports a wavefront covector by the cotangent lift

\[
(y,\eta)\mapsto
(x=\kappa(y),\;D\kappa(y)^{-T}\eta).
\]

For depth-aware ego motion,

\[
\kappa_{\xi,Z}(u)
=\pi\!\left(\exp(\widehat\xi)\,\pi^{-1}(u,Z(u))\right).
\]

Thus the Pact ego-screw is a low-dimensional parameterization of the canonical transformation. **DERIVED correction:** `xi` is not the FIO phase itself; a phase function generates/represents the cotangent relation induced by `xi` and depth.

Occlusions, disocclusions, class births, and depth discontinuities break the single-diffeomorphism model. They are naturally represented by a sum of visible branches plus birth/death residuals—consistent with per-class/depth carriers and the v8 edge-centric split.

### 4.2 Butterfly/Switch verdict

**SOURCED, Ying PDF pp. 5–7:** FIO matrices satisfy complementary low rank; butterfly factorization compresses these blocks, and Switch-Net is the nonlinear architecture inspired by that factorization.

For a generic dense oscillatory FIO, a multilevel butterfly can replace dense application with quasi-linear structured work. Our present image warp is not dense in the implementation: analytic projection plus bilinear sampling is already `Theta(N)`, has no learned video-specific operator weights, and directly preserves the `SE(3)` geometry.

Therefore:

- **NO-GO:** replace the analytic `SE(3)`/homography warp with Switch-Net for speed. It is asymptotically no better here and introduces approximation, n=1 fitting, and payload risks.
- **NO-GO:** claim that butterfly is more faithful merely because FIO theory applies. The exact geometric warp is the faithful canonical map on visible regions.
- **DEFER / bounded opportunity:** a tiny Switch-like residual for genuinely nonlocal phase mixing, multi-depth branch interaction, or occlusion completion, only after the analytic warp. It must beat a simpler layered local warp on exact PoseNet output, d_seg, bytes, and receiver survival.

## 5. What the other two sources add

### 5.1 TUM dissertation

The dissertation supplies three useful constraints:

1. **SOURCED:** adjoint computation and reverse-mode differentiation are the same core operation. This supports treating P0 as an operator-application problem.
2. **SOURCED:** ill-conditioned physical maps can make first-order gradients poorly scaled; physics inversion and half-inverse gradients can improve directions without inverting the full network-parameter Hessian.
3. **TRANSFER:** after—not before—an admitted cheap adjoint, a wavelet-domain half-inverse or regularized normal-operator preconditioner may improve progress per step.

It does **not** show that half-inversion makes the teacher VJP cheap. Computing/fitting the inverse can be more expensive than the gradient it replaces. This is a convergence follow-on, not the P0 construction.

### 5.2 PRIP 2025 B3.1 topic and relevance

The paper is about PINN/EPINN solution of direct and inverse mathematical-physics problems: viscous heat-conducting gas optimal control and retrospective nonlinear heat conduction. It uses adaptive residual weights and anisotropic Fourier-feature encoding.

**SOURCED caveats from the paper:** PINNs are sensitive to loss balance, architecture, and optimizer; can be resource-intensive/unstable; scale poorly with dimension/time; and lack general nonlinear convergence results. Its retrospective example reports restored/final distributions differing by no more than 20%, which is not remotely an evaluator-equivalence or exact-authority guarantee.

**Verdict:** **NO-GO for P0 and NO-GO as a new witness architecture claim.** A PINN avoids paired labels only because the governing differential residual supplies supervision. There is no known PDE residual whose solution set equals the frozen SegNet/PoseNet/archive evaluator cell. The Fourier-feature observation weakly supports the existing coordinate-INR family, especially anisotropic spatial/temporal scales, but is not a novel fold.

## 6. Canonical equation, DAG FEED, and triality

### 6.1 Canonical-equation proposal

New standalone proposal:

`src/tac/canonical_equations/segnet_state_local_bcr_adjoint_20260713.py`

Equation ID: `segnet_state_local_bcr_adjoint_v1`.

It records the conditional rank, apply-cost, state-drift error, renderer-gradient error, and amortization laws with NumPy-fp32 bound arithmetic. It explicitly excludes the false `O(N^2)` baseline and the inference from single-vector rank to operator rank.

**Registry status:** **HELD, NOT APPENDED.** `.omx/state/canonical_equations_registry.jsonl` was already modified by a live sibling; this unit obeys new-files-only and collision discipline. The proposal exposes an explicit `populate_*` main-review surface after empirical rank evidence and shared-file review.

### 6.2 Proposed DAG FEED — not appended

```text
## FEED-invprob-operator-fold-20260713

MEANS / research-only; pointer UNMOVED. Lexing Ying's PDO/FIO structure does not prove that the
frozen SegNet CNN Jacobian is a PDO. Exact source-graph derivation gives a stronger scoped prior:
the tu-efficientnet_b2 U-Net linearization is local/pointwise plus 23 squeeze-excitation global
pooling corrections (sum reduction bottleneck 626). Therefore a LINEAR state-local non-standard
wavelet adjoint is WORTH-MEASURING: preserve near interactions, randomized-fit far blocks from exact
JVP/VJP probes, apply lambda~=Wi^T A_BCR(x0) Wo q, and fail closed on rank/drift/parity/descent/timing.
Derived apply C=cW(Ci+Co)N+sum_l N_l(cs s_l r_l+cr r_l^2)=O(N) only for bounded ranks; exact CNN VJP
is already O(N), so the claimed win is constant-factor and charged by Tapply+Tbuild/K. p0_sparse_adjoint
owns real vector-compressibility AND hierarchical operator-block-rank verdicts. Witness BCR replacement
is NO-GO on current n=1/byte evidence; keep only hybrid corrector/preconditioner. SE(3) visible-region
warp is an FIO cotangent lift, but analytic bilinear warp is already O(N): Switch replacement NO-GO,
residual multi-depth/occlusion phase DEFER. verdict_scope: tested basis/rank/state/loss/hardware only.
Equation proposal: segnet_state_local_bcr_adjoint_v1. DSL leg N/A until a provider is empirically admitted.
```

### 6.3 Triality

- **Equation leg:** conditional BCR rank/cost/error/amortization law plus the explicit FIO phase/cotangent-lift derivation above.
- **DAG leg:** FEED block above, intentionally not appended to the hot historical DAG.
- **DSL leg:** **N/A now, explicitly.** This is theory/design and must not invent trainer flags. A future empirically admitted provider would need a typed scorer-gradient policy mode, exact fallback, resume persistence, and legacy-compatible default-off compilation.
- **Durable artifact leg:** this memo plus the standalone equation proposal.

## 7. Admission ladder and stop rules

1. **Structure gate:** hierarchical block ranks bounded at target error across real early/boundary/late states. Failure: formulation-scoped BCR NO-GO.
2. **Provider gate:** global and annulus costate errors, renderer-gradient direction, and strict exact-teacher one-step descent pass on held-out states. Failure: exact VJP fallback; no trainer integration.
3. **Economics gate:** charged `T_build`, `T_apply`, validation, fallback rate, and valid `K` beat exact backward with paired timing. Big-O or uncharged calibration cannot pass.
4. **In-loop gate:** D-A component timer confirms that backward is actually the dominant slice. If not, re-rank the campaign without erasing this scoped operator result.
5. **Vehicle gate:** resumable stage-bound provider integrated through typed DSL with NumPy-fp32/Torch/MLX parity and no scorer/objective drift.
6. **Authority gate:** exact receiver-closed archive evaluated through upstream CPU/CUDA axes. Only this can move the pointer.

## 8. STORES CONSULTED and pointer honesty

**Contracts / authority:** `CLAUDE.md`; `AGENTS.md`; `docs/operating_manual_craft_handoff.md`; v7.5 §8 operating contract; v8 edge-centric decomposition spec.

**Current P0 state:** `.omx/research/per_epoch_detailed_accounting_20260713.md`; `.omx/research/GO_PACKET_inloop_component_timer_20260713.md`; `FEED-p0-backward-wave-20260713` in the historical DAG; current lane/subagent ownership for `p0_sparse_adjoint`; latest frozen-SegNet exact-forward and on-policy surrogate Codex findings.

**Exact scorer graph:** `upstream/modules.py`; `upstream/frame_utils.py`; installed `segmentation_models_pytorch` U-Net decoder source; installed `timm` EfficientNet/SE source; static model introspection only.

**Prior settled laws checked:** exact costate-injection identity; costate trust-region and Jacobian-drift equations; prior projected pointwise-adjoint NO-GO; v7.5 already-settled separatrix and score-authority rules.

**Literature:** all four staged PDFs listed in the extraction table.

**Pointer delta:** none. This memo improves the means map: it turns a loose PDO analogy into a falsifiable local-plus-low-rank construction, exposes the true cost law, and routes the required operator-rank measurements to the live P0 arm. It creates no score, promotion, launch, or live-run authority.
