# Calin book lessons fold: inverse problems, adjoints, variational solvers, and structured dynamics

**Date:** 2026-07-13  
**Lane:** `calin_book_fold` · L0 theory-routing pass  
**Status:** `research_only=true` · `GROUNDING/MEANS` · uncommitted for main review  
**verdict_scope:** technique-to-Pact transfer design; no implementation, launch, scorer run,
or empirical arm verdict  
**Cost:** **$0**; no paid dispatch and no long local job  
**Pointer delta:** **NONE.** The submittable pointer remains `0.1910828242 [contest-CPU
Linux x86_64]`; the separate `0.1880443979880752` PR128-derived defensive bank is not
promoted here. This pass produced no byte-closed archive and no exact `n600` row.

## Executive route

### Top three highest-EV folds

1. **Certified discrete/inexact adjoint -> all three live P0-backward arms — `WORTH-ARM`.**
   **[DERIVED]** The current frozen-SegNet VJP already *is* the exact discrete adjoint of the
   implemented scorer graph. Therefore a generic “apply the adjoint method” rewrite cannot remove
   the backward. The productive transfer is an admission law for *approximating* that adjoint:
   measure the omitted cotangent, stale-costate, or recomputation error and admit it only while the
   resulting direction remains a certified descent direction. This feeds
   `p0_sparse_adjoint` with an exact mask-error identity, `p0_checkpoint_backward` with the
   discrete-vs-continuous exactness boundary, and `p0_costate_reuse` with an event-triggered
   refresh law. The current “82% P0” is **[DERIVED, DIAGNOSTIC-SCOPED]**, not an in-loop
   measurement: the cited diagnostic measured `537 ms/pair` forward and `3009 ms/pair`
   forward+backward, hence `2472 ms/pair` and `82%` by subtraction, while the same memo records
   that the diagnostic path is about `12x` heavier than the in-loop path. The arm family remains
   high-EV, but its in-loop share must be measured by the already-built component timer.

2. **Quotient-aware, discrepancy-constrained inverse solve -> #483/#157/#391/#249 —
   `WORTH-ARM`.** **[DERIVED]** The witness inverse is intentionally non-unique: many generators
   lie in the same exact receiver/scorer cell. Classical regularization adds the missing selection
   principle—choose the minimum-byte, minimum-complexity representative of that cell—while a
   Morozov-style discrepancy principle says to stop fitting once the deterministic through-`R`
   cell/tolerance is met. This turns Bousfield localization from an existence/quotient grounding
   into a constructive constrained program, composes with reverse-waterfill KKT, uses the exact
   resize adjoint, and reopens only the *regularized receiver-cell formulation* of frame-0 inverse
   solve. It does not erase the recorded negative for the prior #249 formulation.

3. **Adjoint-weighted weak/Deep-Ritz residual allocation -> #316/#318/#320 — `WORTH-ARM`.**
   **[DERIVED]** Our variational level-set flow is closer to Deep Ritz than to a raw strong-form
   PINN. The high-value PINN lesson is not “add more PDE residuals”; it is to avoid gradient
   stiffness, treat sharp interfaces/boundaries specially, preserve temporal causality, and spend
   collocation/compute where a residual can change the receiver objective. For local residual
   `r_i`, costate `z_i`, and compute cost `c_i`, prioritize `|z_i r_i|/c_i`, frozen at stage
   boundaries. This is the goal-oriented dual of #320's stability allocation and directly targets
   scorer-relevant eikonal/Hamilton–Jacobi error rather than globally small PDE residual.

### Sourcing caveat — binding

- **[MEASURED—BOOK METADATA ONLY]** The legitimate indexed record exposes the book title,
  publisher, DOI, dates, and chapter titles. The accessible titles include direct/inverse
  equation solving, classical mechanics, Lagrangian/Hamiltonian systems, conservation laws,
  Hamilton–Jacobi theory, free-boundary problems, classical PDEs, inverse mechanics, parameter
  estimation, equation discovery, inverse classical PDEs, and special structures.
- **[INFERRED]** I did **not** have licensed full-text access. Any statement about what a chapter
  likely teaches beyond its title is marked `INFERRED-FROM-TOC/SCOPE`. The World Scientific page
  returned an access-denied response in this environment; I did not bypass it. Search results
  offering an unauthorized full-book upload were excluded and not opened or used.
- **[MEASURED—COMPANION METADATA ONLY]** The official Springer page for Calin's 2020 companion
  exposes its description and contents, including activation/cost functions, minima,
  approximation/exact learning, information capacity, output manifolds, and neuromanifolds. No
  claim below depends on inaccessible companion prose.
- **[MEASURED—PRIMARY LITERATURE]** Exact properties of Neural ODE adjoints, checkpointed
  adjoints, PINN failure modes, Deep Ritz, HNN/LNN, and NCDEs come from the cited primary papers,
  not from invented book contents.
- **[DERIVED]** Every technique-to-Pact mapping and every equation below is this pass's derivation
  from those sources plus inspected repository artifacts. Every arm remains **[ASSUMED]** until a
  controlled real-input measurement lands.

## 1. Ranked technique -> surface -> transfer -> scoped verdict

| rank | book technique and evidence status | Pact surface | concrete transfer | scoped verdict |
|---:|---|---|---|---|
| **1** | **Adjoint-state / reverse accumulation.** `INFERRED-FROM-INVERSE/PDE-SCOPE`; exact properties `MEASURED-BY-PRIMARY-LITERATURE`. | P0 costate-VJP; all three live arms. | Treat frozen-SegNet reverse mode as the exact **discrete** adjoint. Share one inexact-gradient admission law across sparse support, checkpointing, and reuse. Measure in-loop before attributing the diagnostic 82%. | **`WORTH-ARM`; `FEED-p0_sparse_adjoint`; `FEED-p0_checkpoint_backward`; `FEED-p0_costate_reuse`.** Scope: the current discrete scorer/training graph. |
| **2** | **Inexact/stale adjoints and optimal-control costates.** `INFERRED-FROM-INVERSE/MECHANICS-SCOPE`; transfer `DERIVED`. | `p0_costate_reuse`; #247/#426 Pontryagin/costate organ. | Reuse the input costate only within a measured piecewise-smooth trust tube; refresh on stage, topology, margin-support, resize/quantization, or validation events rather than a guessed fixed period. | **`WORTH-ARM`; `FEED-p0_costate_reuse`; `FEED-247`; `FEED-426`.** Scope: smooth within-regime segments only. |
| **3** | **Regularization + discrepancy principle + well-posedness.** `INFERRED-FROM-INVERSE-CHAPTER-TITLES`; exact inverse-problem principles `MEASURED-BY-STANDARD-LITERATURE`. | Witness inverse; #483 Bousfield; #157 KKT/reverse-waterfill; #391 resize adjoint; #249 frame-0 inverse solve. | Optimize minimum archive/geometry complexity over the exact receiver-cell feasible set. Use scorer-aware regularization and a deterministic through-`R` discrepancy tube, not RGB fidelity. Add deterministic tie-breaking because uniqueness is neither available nor required. | **`WORTH-ARM`; `FEED-483/157/391/249`.** Scope: a new quotient/discrepancy formulation; prior #249 negative stays valid for its tested formulation. |
| **4** | **PDE-constrained optimization / goal-oriented adjoints.** `INFERRED-FROM-INVERSE-PDE-SCOPE`; adjoint equations `DERIVED/STANDARD`. | Witness-as-PDE-solver; variational level-set flow. | Separate state equation, receiver objective, and control. Use the adjoint twice: parameter gradient and dual-weighted residual/error allocation. This prevents spending solve accuracy where it cannot change score. | **`WORTH-ARM`; `FEED-316/318/320`.** Scope: weak/variational receiver-goal allocation, not a replacement scorer. |
| **5** | **Deep Ritz / variational neural solve.** `INFERRED-FROM-SOLVING-EQUATIONS/PDE-SCOPE`; method property `MEASURED-BY-PRIMARY-LITERATURE`. | GR unified action; level-set/Morse–Smale energy. | Keep the energy as the primary object; use weak/energy residuals and admissible parameterizations. Do not bolt on an equally weighted strong-form PDE loss. | **`GROUNDING-only` for current formulation; `WORTH-ARM` only as the dual-weighted allocator in rank 4.** |
| **6** | **PINN loss stiffness and gradient imbalance.** `INFERRED-FROM-SCOPE`; failure mode `MEASURED-BY-PRIMARY-LITERATURE`. | #316/#318/#320 eikonal/viscous-HJ curriculum. | Normalize/allocate by receiver influence and enforce stage-level continuation. Freeze weights within a stage per the operating contract; never install per-step adaptive loss weights. | **`FEED-316/318/320`.** Scope: curriculum and residual allocation, not evidence that PINNs beat the current solver. |
| **7** | **PINN causality / sequential time marching.** `INFERRED-FROM-PDE-SCOPE`; method property `MEASURED-BY-PRIMARY-LITERATURE`. | Deep-unroll witness; #344 trajectory observer. | Treat earlier time/stage slabs as prerequisites; prevent future residuals from dominating before earlier dynamics are resolved. For logged trajectories, preserve chronological folds and event resets. | **`FEED-344`; `GROUNDING-only` now.** Existing #344 data gate (`>=10` logged trajectories and held-out-by-run win over persistence+slope) remains binding. |
| **8** | **Sharp-gradient, boundary-layer, and hard-boundary handling.** `INFERRED-FROM-FREE-BOUNDARY/HJ/PDE-TITLES`; standard PINN result `MEASURED-BY-PRIMARY-LITERATURE`. | Multiphase SDF interfaces; eikonal annulus; exact receiver. | Use SDF/distance-function parameterizations for hard geometry where possible; concentrate weak residual samples near moving interfaces, but select them by receiver costate rather than curvature alone. | **`FEED-316/318/320`; `GROUNDING-only` for hard constraints already embodied.** |
| **9** | **Adaptive checkpoint adjoints.** `MEASURED-BY-PRIMARY-LITERATURE`. | `p0_checkpoint_backward`. | First determine whether the P0 is activation-memory/bandwidth bound. Checkpointing trades memory for recomputation; it can improve wall time only through a measured hardware effect, not by theory alone. Preserve exact discrete gradients and stage checkpoints. | **`WORTH-ARM`; `FEED-p0_checkpoint_backward`.** Scope: exact discrete graph partitions only. |
| **10** | **Neural-ODE constant-memory continuous adjoint.** `MEASURED-BY-CHEN-ET-AL`; limitations `MEASURED-BY-ANODE/ACA`. | Frozen EfficientNet-B2 teacher backward. | The advertised O(1)-memory continuous adjoint is not an exact backward for the fixed discrete B2 graph. Reverse integration can be unstable and optimize-then-discretize can disagree with the implemented discrete gradient. | **`NO-GO`.** `verdict_scope`: drop-in exact replacement for the present frozen discrete teacher only; Neural ODEs remain open for genuinely continuous new models. |
| **11** | **NCDE / Neural-ODE trajectory models.** `MEASURED-BY-PRIMARY-LITERATURE`; book placement `INFERRED-FROM-SPECIAL-STRUCTURES`. | #344 NCDE training-trajectory model. | Use control interpolation for irregularly sampled, partially observed run telemetry and checkpoint at observation/event times. It is an observer/predictor, not the teacher costate and not score authority. | **`FEED-344`.** Scope: after the existing corpus gate; no new arm before it. |
| **12** | **Hamiltonian and Lagrangian neural parameterizations.** `INFERRED-FROM-NAMED-CHAPTERS`; properties `MEASURED-BY-HNN/LNN-PAPERS`. | GR unified action; #247/#426. | Use variational structure to audit equations and, on conservative subsystems, test state-costate pairing. Do not force a conservative Hamiltonian onto the dissipative level-set descent. | **`GROUNDING-only` for #247/#426; `NO-GO` as the primary level-set-flow parameterization.** Scope: unforced conservative HNN/LNN; port/contact/dissipative extensions remain untested. |
| **13** | **Symplectic adjoint integration.** `MEASURED-BY-SANZ-SERNA`; relation to book `INFERRED-FROM-HAMILTONIAN-SCOPE`. | Smooth shadow-controller segments; adjoint correctness audit. | On a genuinely Hamiltonian/symplectic submodel, reverse accumulation should preserve the state-costate pairing. Measure pairing defect as an audit; reset across hybrid events. | **`GROUNDING-only`; `FEED-247/426`.** Scope: diagnostic shadow model, not a scorer-backward accelerator. |
| **14** | **Hamilton–Jacobi / viscosity / free-boundary theory.** chapter titles `MEASURED`; specific numerical lessons `INFERRED` and grounded in standard PDE literature. | #316/#318/#320. | Continue viscosity/eikonal homotopy from smooth to sharp while respecting two-sided stability; evaluate receiver-goal error near shocks/interfaces rather than global residual alone. | **`FEED-316/318/320`.** Scope: mathematical grounding for existing stability family, not a new empirical win. |
| **15** | **Conservation-law identification / equation discovery.** chapter titles `MEASURED`; transfer `ASSUMED`. | Canonical equations; trajectory organ. | Use only as a falsifier: test whether proposed invariants or dissipations hold on recorded trajectories. Never “discover” a law from one run and register it as authority. | **`GROUNDING-only`.** Scope: diagnostics until multiple independent trajectories and held-out residuals exist. |
| **16** | **Companion: exact learning, information capacity, output/neuromanifolds.** companion TOC `MEASURED`; transfer `DERIVED`. | Argmax-cell quotient; #157 rate allocation; margin/Fisher surfaces. | View the witness solution set as a receiver-induced output manifold stratified by argmax margins. Allocate bits to transverse directions that cross a cell boundary; quotient/null directions are candidates for rate removal. | **`GROUNDING-only`; `FEED-157`.** Much of this is already embodied by margin/Fisher/reverse-waterfill work. |

## 2. The adjoint fold: the exact law and what each P0 arm can actually win

### 2.1 Discrete receiver adjoint

Let the decoded generator and exact receiver be

\[
  y_\theta = G(\theta),\qquad x_\theta=R(y_\theta),\qquad
  h_{k+1}=f_k(h_k),\qquad z=h_L=F_{\rm seg}(x_\theta),
\]

and let `ell_tau(z)` be the differentiable training surrogate used for the SegNet term. Define

\[
  q=\nabla_z\ell_\tau,\qquad
  \lambda_L=q,\qquad
  \lambda_k=(D f_k(h_k))^\top\lambda_{k+1}.
\]

Then

\[
  \lambda_x=(D F_{\rm seg}(x_\theta))^\top q,
  \qquad
  g_\theta=(D(R\circ G)(\theta))^\top\lambda_x.
  \tag{1}
\]

**[DERIVED/EXACT for the implemented differentiable graph]** Equation (1) is reverse-mode AD.
Calling it an adjoint changes the decomposition and exposes approximation opportunities, but does
not make its arithmetic disappear. The main new leverage is that the expensive teacher costate
`lambda_x` can sometimes be compressed, refreshed sparsely in time, or generated from a reduced
output cotangent—provided the resulting generator direction is still valid.

### 2.2 One admission law for every approximate backward

Let `g` be the exact current generator gradient and `g_hat=g+e` an approximate direction produced
by masking, reuse, checkpoint/recompute approximation, or a surrogate. If

\[
  \|e\|_2 \le \rho\,\|g\|_2,\qquad 0\le\rho<1,
  \tag{2}
\]

then

\[
  \langle g,-\widehat g\rangle
  =-\|g\|_2^2-\langle g,e\rangle
  \le -(1-\rho)\|g\|_2^2<0.
  \tag{3}
\]

**[DERIVED/EXACT]** Thus `-g_hat` is a descent direction for the same local surrogate. This is a
mathematical admission gate, not an empirical speed or exact-score guarantee. In practice, main
must estimate the error from preregistered exact-VJP spot checks and positive/negative controls;
no guessed `rho` threshold is introduced here. Exact through-`R`/`n600` validation remains the
downstream authority.

### 2.3 Feed to `p0_sparse_adjoint`

For an output-support mask `M`, let `q_M=Mq`. The exact generator-gradient error is

\[
  g-g_M=(D(R\circ G))^\top(D F_{\rm seg})^\top(I-M)q.
  \tag{4}
\]

**[DERIVED]** This tells the arm what to measure: not only energy retained in `q`, and not only
energy retained in the cached input costate, but the induced error after contraction with the
current renderer/generator Jacobian. A narrow output annulus can produce a spatially broad input
costate because B2 receptive fields mix support. Also, sending zeros into an ordinary dense VJP
does not imply fewer FLOPs or lower wall time; the arm wins only if a structured partial backward,
feature cut, or sparse kernel actually avoids work. A measured low-rank input costate may still be
useful for storage/reuse without making the frozen teacher VJP cheaper.

### 2.4 Feed to `p0_checkpoint_backward`

**[MEASURED-BY-LITERATURE]** Neural ODEs introduced a continuous adjoint that reconstructs a
trajectory by reverse integration for constant-memory training. ANODE showed two decisive limits:
backward integration of the forward ODE may be numerically unstable, and the continuous
optimize-then-discretize adjoint need not equal the gradient of the implemented discrete solver.
Adaptive checkpoint adjoint methods restore trajectory states and control gradient error with a
memory/recomputation trade.

**[DERIVED-Pact]** EfficientNet-B2 is a fixed discrete composition, not an ODE discretization we
are free to change. Exact reverse mode must retain, recompute, or reversibly reconstruct the
needed activations. Therefore:

- checkpointing is a **memory/bandwidth hypothesis**, not a theorem of wall-clock improvement;
- a continuous-adjoint replacement is `NO-GO` for exact scorer equivalence;
- an exact segment-checkpoint schedule remains `WORTH-ARM` only after the arm measures peak
  memory, memory pressure/bandwidth evidence, recompute work, and paired wall time at the real
  in-loop shape;
- stage checkpoints required by the operating contract are run-state custody and are distinct
  from autograd activation checkpoints inside one iteration.

### 2.5 Feed to `p0_costate_reuse`

Write `A_t = D(R o G)(theta_t)` and `lambda_t = lambda_x(x_t)`, where `o` denotes
composition, so
`g_t=A_t^T lambda_t`. If the arm recomputes the cheap current `A_t` action but reuses a costate
from refresh time `r`, then

\[
  \|g_t-A_t^\top\lambda_r\|
  \le \|A_t\|\,\|\lambda_t-\lambda_r\|.
  \tag{5}
\]

If it also reuses/approximates `A_r`, then

\[
  \|A_t^\top\lambda_t-A_r^\top\lambda_r\|
  \le \|A_t-A_r\|\,\|\lambda_t\|+\|A_r\|\,\|\lambda_t-\lambda_r\|.
  \tag{6}
\]

**[DERIVED]** Equations (5)-(6) make “costate smoothness” falsifiable. Smoothness is expected only
inside a fixed stage/active-set regime; it is not assumed across tau changes, curriculum changes,
topology birth/death, margin-support changes, validation/rollback events, or discrete receiver
events. Refresh should therefore be event/trust-residual driven, with exact VJP spot checks, not a
fixed guessed `k`. The diagnostic-only forward/backward numbers imply a gradient-free crossover of
about `2472/537 = 4.60` teacher forwards per backward **on that heavier diagnostic path only**;
the accounting memo explicitly blocks transferring this ratio to the real in-loop path until the
component timer measures it.

## 3. Inverse-problem fold: what regularization and well-posedness add

### 3.1 The correct unknown is an equivalence class, then a minimum-rate representative

Let `Q` collect exact parse-back, through-`R` SegNet-cell behavior, and PoseNet behavior relevant
to the frozen evaluator. Define

\[
  \theta\sim\theta'\quad\Longleftrightarrow\quad Q(G(\theta))=Q(G(\theta')).
  \tag{7}
\]

**[DERIVED]** The evaluator deliberately makes the inverse non-identifiable in RGB space. Hence
Hadamard uniqueness of `theta` is the wrong objective. The useful well-posedness target is:

1. **existence** of at least one legal, receiver-closed representative;
2. **stability of the scorer cell/quotient** under allowed numerical perturbations;
3. **deterministic selection** of a minimal-complexity representative.

A constructive program is

\[
\begin{aligned}
 \min_{\theta,c}\quad & B(c)+\beta\Omega_{\rm geom}(\theta)\\
 \text{s.t.}\quad & d_{\rm seg}(Q_{\rm seg}(R(G_\theta,c)),Q_{\rm seg}^{*})\le\delta_{\rm seg},\\
 & d_{\rm pose}(Q_{\rm pose}(R(G_\theta,c)),Q_{\rm pose}^{*})\le\delta_{\rm pose},\\
 & \text{archive parse-back, deterministic decode, and runtime compliance pass.}
\end{aligned}
\tag{8}
\]

Here `B(c)` is exact archive bytes and `Omega_geom` regularizes generator complexity in the lawful
witness chart. **[DERIVED]** RGB smoothness is generally the wrong prior because it can spend bits
along receiver-null directions and erase scorer-legible texture. Suitable regularizers are
forward/operator aware: generator description length, interface complexity, receiver-weighted
Sobolev/TV terms, codebook sparsity, and per-class carrier complexity.

### 3.2 Discrepancy principle without pretending receiver quantization is noise

**[MEASURED-BY-STANDARD-LITERATURE]** Tikhonov regularization selects stable solutions by trading
data fit against a prior; Morozov's discrepancy principle stops when residual matches an allowed
data-error level. **[DERIVED-ANALOGY]** Our evaluator is deterministic, so `delta` in (8) is not a
stochastic noise estimate. It is an exact, preregistered receiver-cell/rounding/score tolerance.
Once the candidate lies safely inside that tube, further surrogate-CE reduction has no authority
unless it buys cell robustness or fewer bytes. This is the inverse-problem version of floor-first.

### 3.3 Composition with existing surfaces

- **#483 Bousfield:** localization supplies the right quotient language but no constructive
  section/minimal representative. Equation (8) is the missing optimizer; it does not claim the
  localized homotopy machinery itself solves the witness.
- **#157 reverse-waterfill/KKT:** once (8) supplies local receiver sensitivities and active
  constraints, #157 remains the bit allocator. No reopening of already-settled KKT arithmetic.
- **#391 resize adjoint:** the constraint gradient must use the exact transpose of the actual
  resize/round chain where differentiable or its explicitly declared surrogate. Never optimize a
  pre-resize field and transfer the claim.
- **#249 frame-0 inverse solve:** reopen only the quotient/discrepancy/goal-regularized formulation.
  `verdict_scope`: the recorded negative continues to rule out the tested frame-0 formulation; it
  does not kill all regularized inverse solvers.

## 4. PINN / Deep Ritz fold: use the failure literature, not the slogan

### 4.1 Why Deep Ritz is the closer analogy

**[MEASURED-BY-PRIMARY-LITERATURE]** Deep Ritz optimizes a variational energy rather than forcing
a pointwise strong-form residual. PINN analyses document numerical stiffness and unbalanced
backpropagated gradients among soft constraints; later work shows causal curricula/sequential
training can repair some time-dependent failures, and distance-function constructions can impose
boundary conditions exactly.

**[DERIVED-Pact]** The multiphase level-set witness is already an energy/gradient-flow object.
Adding raw PDE residuals with independently tuned per-step weights would duplicate the action,
violate the stage-boundary-only weight rule, and likely amplify stiffness near interfaces. The
transfer is instead:

- encode hard geometric constraints in the chart/parameterization where possible;
- use weak/energy residuals for viscous HJ/eikonal terms;
- continue smooth-to-sharp only at governed stage boundaries;
- advance causal time/stage slabs only when earlier residual and receiver-margin gates pass;
- allocate samples/solver effort by receiver impact.

### 4.2 Goal-oriented dual-weighted residual law

For a discretized state equation `A(u,theta)=0` and receiver objective `J(u,theta)`, define the
adjoint `z` by

\[
  (D_uA)^\top z=(D_uJ)^\top.
  \tag{9}
\]

For a state approximation with local residuals `r_i`, first-order objective error is

\[
  \delta J\approx-\langle z,r\rangle=-\sum_i z_i r_i.
  \tag{10}
\]

**[DERIVED/FIRST-ORDER GOAL-ORIENTED FORM]** With per-item compute cost `c_i`, a local
benefit-per-cost proxy is

\[
  a_i=\frac{|z_i r_i|}{c_i}.
  \tag{11}
\]

**[ASSUMED-Pact allocator]** Equation (11) becomes the greedy allocation rule only under the local
approximation that residual corrections are independently purchasable at costs `c_i`; coupled
updates and dual-approximation error can break that ordering. It is nevertheless a sharper probe
than sampling the largest eikonal residual: a large residual in a receiver-null region is low
first-order value, while a moderate residual on a class boundary with large SegNet costate is high
first-order value. Main must freeze the allocation for a stage, record it in the typed DSL, and
compare it against uniform and residual-only controls through full `n600` exact receiver closure.

## 5. Hamiltonian/Lagrangian and Neural-ODE folds

### 5.1 The conservative/dissipative boundary

HNNs learn a scalar Hamiltonian and generate dynamics through its symplectic gradient; LNNs learn
a Lagrangian and enforce Euler–Lagrange structure. **[MEASURED-BY-PRIMARY-LITERATURE]** These
parameterizations improve conservation/structure on compatible mechanical systems.

For our variational gradient flow with positive mobility `M`, however,

\[
  \dot\phi=-M\,\frac{\delta E}{\delta\phi}
  \quad\Longrightarrow\quad
  \frac{dE}{dt}=-\left\langle\frac{\delta E}{\delta\phi},
  M\frac{\delta E}{\delta\phi}\right\rangle\le0.
  \tag{12}
\]

**[DERIVED]** A primary conservative HNN is therefore the wrong inductive bias for the dissipative
witness flow. `NO-GO` is scoped to an unforced conservative HNN/LNN replacement. A port/contact
Hamiltonian or forced Lagrangian model could represent dissipation, but no source or measurement
reviewed here shows it beats the current direct action, so that broader family remains open but
unarmed.

### 5.2 What symplectic theory still contributes

**[MEASURED-BY-SANZ-SERNA]** Reverse accumulation for certain Runge–Kutta discretizations has a
hidden symplectic partitioned structure and preserves a state-adjoint pairing. **[DERIVED-Pact]**
This is useful as an audit on any genuinely smooth conservative shadow subsystem in #247/#426:

\[
  \Delta_{\rm pair,k}
  =\langle\lambda_{k+1},\delta h_{k+1}\rangle
   -\langle\lambda_k,\delta h_k\rangle.
  \tag{13}
\]

Pairing defect can catch a mismatched discrete adjoint. It is not expected to remain zero across
dissipation or hybrid stage/topology/receiver events; those need explicit jump maps/resets. This
is a `GROUNDING-only` audit, not a claim that symplectic integration accelerates SegNet.

### 5.3 Neural ODE / NCDE boundary

- **Frozen B2:** discrete exactness controls; continuous adjoint drop-in is `NO-GO`.
- **Deep-unroll new witness model:** a Neural ODE is admissible only if the decoder/generator is
  actually defined as that ODE, with deterministic solver/tolerances and event maps in the counted
  artifact contract. Quantization, `R`, stage changes, and topology changes make it a hybrid
  system; the adjoint needs jump Jacobians.
- **#344 NCDE observer:** NCDEs are well matched to irregular, partially observed trajectory
  telemetry. Preserve the existing `>=10` trajectory and held-out-by-run gate; beat persistence
  plus slope before any organ status changes. Its adjoint trains the observer—it does not supply
  the frozen scorer costate by identity.

## 6. WORTH-ARM dispatch specifications for main

### Arm A — receiver-cotangent structured sparse adjoint

**[ASSUMED; `FEED-p0_sparse_adjoint`]** At preserved real stage checkpoints, compute the exact
output cotangent `q`, exact input costate, and exact generator gradient. Rank output support by a
receiver-aware statistic; replay nested masks and measure (i) retained `q` mass, (ii) input-costate
error, (iii) final generator-gradient error/cosine through the current renderer Jacobian, and (iv)
actual backward wall time. Include a random mask with identical cardinality and a dense-zeroed-VJP
control. The arm passes only if a *structured* implementation avoids work and its exact spot-check
directions satisfy the descent gate; a concentrated costate with unchanged dense VJP time is a
representation result, not a P0 speed win. Validate the best surviving treatment at full real
`n600` through `R`; do not infer archive or score improvement from gradient similarity.

### Arm B — exact discrete segment checkpointing

**[ASSUMED; `FEED-p0_checkpoint_backward`]** On the exact in-loop batch/shape and frozen scorer,
pair an uncheckpointed control with several graph-native segmentations chosen from real activation
sizes. Record forward, backward, recompute, peak memory, memory pressure/bandwidth evidence, and
total epoch wall time under identical seed/threading. Verify the resulting gradient/update against
the exact discrete control; do not substitute a continuous adjoint or a new scorer architecture.
The falsifier is simple: if the path is compute-bound or recomputation dominates, checkpointing is
`NO-GO` for wall-clock on this hardware even if peak memory falls. This activation experiment does
not alter the separate mandatory end-of-stage crash-resume checkpoints.

### Arm C — event-triggered costate reuse with exact refresh sentinels

**[ASSUMED; `FEED-p0_costate_reuse`, `FEED-247/426`]** Cache the exact input costate only at a
refresh checkpoint. At subsequent steps, combine that costate with the current renderer/generator
Jacobian, log state/margin/active-support drift, and periodically compute a hidden exact VJP
sentinel. Fit no global smoothness claim: estimate within-stage autocorrelation and the empirical
right sides of (5)-(6), with forced refresh at stage/tau/topology/receiver events. Compare fixed-`k`,
event-triggered, persistence, linear extrapolation, and always-exact controls using paired wall
time and exact-gradient error. Adopt a refresh policy only where exact sentinels preserve descent
and the same real `n600` treatment does not regress receiver-cell debt. Gradient-free ES/SPSA is a
separate comparator whose evaluation count must be costed from the real in-loop timer, not the
diagnostic `4.60` ratio.

### Arm D — receiver-discrepancy inverse continuation

**[ASSUMED; `FEED-483/157/391/249`]** Start from the existing differentiable receiver surrogate and
optimize a regularized generator inside the exact `R` loop. Continue temperature/viscosity only at
stage boundaries; once exact parse-back enters the preregistered Seg/Pose receiver tube, freeze
fidelity pressure and use #157 to remove bytes/complexity until a constraint becomes active. Use
#391's exact resize transpose, a scorer-aware geometry/MDL regularizer, and deterministic
tie-breaking among byte-equal representatives. Controls are the previous #249 formulation, RGB
Tikhonov/TV, and no regularizer. Verdicts remain formulation-scoped; only a byte-closed exact
`n600` row can upgrade this arm.

### Arm E — adjoint-weighted weak-residual allocator

**[ASSUMED; `FEED-316/318/320`]** At a stage boundary, compute local weak eikonal/HJ residual,
receiver costate, and per-sample compute cost; freeze the next stage's allocation by (11). Compare
uniform, residual-only, costate-only, and product-per-cost allocations with identical total work
and in-run controls. Track all class interfaces, island births, `d_pose`, and rate—not a headline
residual. Kill the arm if it reduces PDE residual while failing to improve receiver-cell debt, if
the allocation changes within a stage, or if sharp-region sampling destabilizes the two-sided CFL
gate. A full real `n600` through-`R` row is required for any score claim.

## 7. Accessible chapter coverage audit — no silent omissions

The left column below is **[MEASURED]** chapter-title metadata only. Routes are
`INFERRED-FROM-TITLE` unless grounded above by primary literature.

| accessible title | Pact route | disposition |
|---|---|---|
| Review of Feed-forward Neural Networks | Frozen scorer/generator composition; companion approximation/manifold grounding. | `GROUNDING-only` |
| Deep Learning for Solving Equations | PINN/Deep Ritz/direct-vs-inverse solve. | ranks 4–8 |
| Introduction to Classical Mechanics with Neural Nets | Dynamical-system state, controls, trajectories. | `FEED-247/344/426` |
| Lagrangian Systems | Unified action and Euler–Lagrange audit. | `GROUNDING-only`; dissipative caveat |
| Hamiltonian Systems | HNN, symplectic state-costate structure. | `GROUNDING-only`; primary-flow `NO-GO` |
| Conservation Laws | Invariant/dissipation falsifiers. | `GROUNDING-only` |
| Hamilton–Jacobi Theory | Eikonal/viscous-HJ continuation. | `FEED-316/318/320` |
| Free-boundary Value Problems | Multiphase level-set interfaces and topology events. | `FEED-316/318/320` |
| Classical PDEs of Mathematical Physics | Weak/variational residuals and adjoint state. | ranks 4–6 |
| Introduction to Inverse Problems in Mechanics | Witness quotient, well-posedness, regularization. | rank 3 |
| Identifying Conservation Laws | Recorded-trajectory falsification only. | `GROUNDING-only` |
| Parameter Estimation | Frame-0/control/code inference with exact receiver. | `FEED-249`; formulation-scoped |
| Equations Discovery | Candidate-law generation only; multi-trajectory validation required. | `GROUNDING-only` |
| Inverse Problems for Classical PDEs | PDE-constrained optimization and goal adjoint. | ranks 1, 3, 4 |
| Special Structures | Neural ODE/HNN/LNN/NCDE are plausible from the stated book scope, but chapter detail was inaccessible. | `INFERRED`; ranks 9–13 |
| Further Applications | No accessible subtopic list was treated as authority. | no transfer claimed |
| Miscellaneous Exercises | No accessible exercise content. | no transfer claimed |

## 8. Canonical equation candidate and DAG FEED — held, not silently orphaned

### Canonical equation candidate

```yaml
equation_id: receiver_discrete_adjoint_inexact_descent_v1
status: FORMALIZATION_PENDING
claim_label: DERIVED
equations: [1, 2, 3, 4, 5, 6]
statement: >
  The frozen receiver VJP is the exact discrete adjoint. An approximate masked,
  checkpointed, reused, or surrogate generator gradient is locally admissible only
  while its exact-gradient error is smaller than the exact gradient norm; under that
  condition its negative is a strict descent direction for the same surrogate.
empirical_anchor: null
verdict_scope: local differentiable surrogate within one smooth active-set regime
future_consumers:
  - p0_sparse_adjoint
  - p0_checkpoint_backward
  - p0_costate_reuse
  - task_247
  - task_426
registration_disposition: >
  DEFER TO MAIN. This mission permits new files only and no empirical anchor was
  produced. Registering into the shared canonical-equation registry here would both
  violate scope and overstate empirical authority.
```

Equation (11), `adjoint_weighted_receiver_residual_priority_v1`, is a second clean
**[DERIVED]** candidate but should remain a child of the receiver-adjoint law until Arm E supplies
a real comparison against uniform and residual-only allocation.

### DAG FEED proposed for main append

```yaml
feed_id: FEED-calin-book-fold-20260713
status: PROPOSED_NOT_APPENDED
research_only: true
parents:
  - per_epoch_detailed_accounting_20260713
  - task_483_bousfield_localization
  - task_157_reverse_waterfill_kkt
  - task_391_resize_adjoint
  - task_249_frame0_inverse_solve
  - task_247_pontryagin_controller
  - task_426_costate_organ
  - task_316_eikonal
  - task_318_viscous_hj
  - task_320_two_sided_stability
  - task_344_ncde_trajectory
children:
  - p0_sparse_adjoint_receiver_cotangent_error
  - p0_checkpoint_exact_discrete_adjoint
  - p0_costate_reuse_event_refresh
  - receiver_discrepancy_inverse_continuation
  - adjoint_weighted_weak_residual_allocator
gates:
  - in_loop_component_timing_before_P0_share_claim
  - exact_gradient_sentinel_before_inexact_adjoint_admission
  - typed_DSL_before_execution
  - full_real_n600_through_R_before_score_verdict
  - byte_closed_archive_before_pointer_delta
verdict_scope: theory routing only
append_disposition: DEFER_TO_MAIN_NEW_FILES_ONLY
```

## 9. Triality and apparatus wire-in

- **DSL leg — [ASSUMED future work]:** each selected arm needs a typed program node for its
  support rule, checkpoint partition, refresh/event policy, or residual allocation. This memo does
  not invent CLI flags. Loss/residual allocations are compiled and frozen at stage boundaries.
- **DAG leg — [DERIVED]:** the proposed FEED above joins every transfer to the existing task
  surface and records the profiler/exact-gradient/`n600` gates. Shared append is intentionally
  deferred to main under the new-files-only instruction.
- **Equations leg — [DERIVED]:** equations (1)–(6) give the receiver discrete-adjoint and
  approximation gate; (7)–(8) give the quotient inverse program; (9)–(11) give receiver-goal
  residual allocation; (12) separates dissipative from Hamiltonian dynamics; (13) is a pairing
  audit. Registry mutation is deferred.
- **Sensitivity-map hook:** exact/approximate costate error and `|z_i r_i|` become receiver-aware
  sensitivity signals.
- **Pareto hook:** every accepted arm must report Seg, Pose, rate/bytes, and wall time separately;
  no composite-only promotion.
- **Bit allocator hook:** inverse continuation hands active receiver constraints/marginals to
  existing #157 rather than creating a second allocator.
- **Cathedral/autopilot hook:** future dispatch consumes the profiler gate, exact-gradient
  sentinel, and receiver-closure status; theory alone cannot auto-promote.
- **Continual-learning hook:** exact arm outcomes become typed probe/canonical-equation anchors;
  this theory memo is not an empirical posterior update.
- **Probe disambiguators:** sparse `q` vs sparse `lambda_x`; memory-bound vs compute-bound
  checkpointing; fixed-`k` vs event refresh; residual-only vs costate-weighted allocation; prior
  #249 vs quotient/discrepancy inverse formulation.

## 10. Falsifiers and scoped negatives

1. **Adjoint family falsifier:** if the in-loop component timer shows teacher backward is not a
   dominant share, the “P0” priority is demoted even though the diagnostic ratio remains true in
   its own scope.
2. **Sparse-adjoint falsifier:** dense receptive-field costate plus no structured-kernel wall-time
   reduction kills the sparse-*compute* formulation, not costate compression/reuse.
3. **Checkpoint falsifier:** compute-bound behavior or recomputation regression kills wall-clock
   checkpointing on that hardware, not checkpointing's memory benefit.
4. **Reuse falsifier:** event-local autocorrelation that fails the exact descent sentinel kills
   reuse for that regime, not the existence of costates or #426's other roles.
5. **Inverse falsifier:** if scorer-aware regularization enters the same receiver cell with more
   bytes or worse stability than controls, kill that regularizer/formulation, not inverse problems.
6. **PINN/Ritz falsifier:** lower PDE residual without lower exact receiver-cell debt kills that
   residual allocation, not the eikonal/HJ family.
7. **HNN negative:** conservative HNN as primary dissipative level-set flow is `NO-GO`; this does
   not rule out a port/contact/dissipative structured model if separately derived and measured.
8. **Neural-ODE negative:** continuous adjoint as an exact frozen-B2 replacement is `NO-GO`; this
   does not rule out Neural ODE/NCDE models whose forward definition is genuinely continuous.

## 11. Legitimate sources used

### Book and companion metadata

- Ovidiu Calin, *Deep Learning Methods of Mathematical Physics, Vol. I: Direct and Inverse
  Problems*, World Scientific, DOI `10.1142/14702`: [publisher landing page](https://www.worldscientific.com/worldscibooks/10.1142/14702)
  (access denied here; not bypassed) and [ScienceOpen bibliographic/chapter record](https://www.scienceopen.com/book?vid=40d45c95-44a2-4bd9-a2e3-ccc7c0fb956c).
- Ovidiu Calin, *Deep Learning Architectures: A Mathematical Approach*, Springer, 2020:
  [official description and TOC](https://link.springer.com/book/10.1007/978-3-030-36721-3).

### Adjoint, optimal control, Neural ODE, and NCDE

- Chen et al. (2018), [Neural Ordinary Differential Equations](https://papers.neurips.cc/paper_files/paper/2018/hash/69386f6bb1dfed68692a24c8686939b9-Abstract.html).
- Gholami, Keutzer, and Biros (2019), [ANODE: Unconditionally Accurate Memory-Efficient Gradients for Neural ODEs](https://www.ijcai.org/proceedings/2019/103).
- Zhuang et al. (2020), [Adaptive Checkpoint Adjoint Method for Gradient Estimation in Neural ODE](https://proceedings.mlr.press/v119/zhuang20a.html).
- Sanz-Serna (2016), [Symplectic Runge–Kutta schemes for adjoint equations, automatic differentiation, optimal control, and more](https://epubs.siam.org/doi/10.1137/151002769).
- Kidger et al. (2020), [Neural Controlled Differential Equations for Irregular Time Series](https://proceedings.neurips.cc/paper/2020/hash/4a5876b450b45371f6cfe5047ac8cd45-Abstract.html).
- MIT OpenCourseWare, [adjoint-state and discrete-recurrence notes index](https://ocw.mit.edu/courses/18-s096-matrix-calculus-for-machine-learning-and-beyond-january-iap-2023/pages/lecture-notes/) (open educational corroboration; primary-paper claims above carry the technical load).

### PINNs, Deep Ritz, and structured mechanics

- Raissi, Perdikaris, and Karniadakis (2019), [Physics-informed neural networks](https://doi.org/10.1016/j.jcp.2018.10.045).
- E and Yu (2018), [The Deep Ritz Method](https://arxiv.org/abs/1710.00211).
- Wang, Teng, and Perdikaris (2020), [Understanding and mitigating gradient pathologies in PINNs](https://arxiv.org/abs/2001.04536).
- Krishnapriyan et al. (2021), [Characterizing possible failure modes in physics-informed neural networks](https://proceedings.neurips.cc/paper_files/paper/2021/hash/df438e5206f31600e6ae4af72f2725f1-Abstract.html).
- Wang, Sankaran, and Perdikaris (2022), [Respecting causality is all you need for training PINNs](https://arxiv.org/abs/2203.07404).
- Sukumar and Srivastava (2021), [Exact imposition of boundary conditions with distance functions in PINNs](https://arxiv.org/abs/2104.08426).
- Becker and Rannacher (2001), [An optimal control approach to a posteriori error estimation in finite element methods](https://doi.org/10.1017/S0962492901000010) (dual-weighted, goal-oriented residual control).
- Nochetto, Veeser, and Verani (2009), [A safeguarded dual weighted residual method](https://academic.oup.com/imajna/article/29/1/126/680830) (why an unsafeguarded approximate dual can understate goal error).
- Greydanus, Dzamba, and Yosinski (2019), [Hamiltonian Neural Networks](https://papers.neurips.cc/paper_files/paper/2019/hash/26cd8ecadce0d4efd6cc8a8725cbd1f8-Abstract.html).
- Cranmer et al. (2020), [Lagrangian Neural Networks](https://arxiv.org/abs/2003.04630).

### Inverse problems and regularization

- Hansen (2010), [Tikhonov regularization chapter in *Discrete Inverse Problems*](https://epubs.siam.org/doi/10.1137/1.9781611972344.ch5).
- Ito and Jin (2010), [A new approach to nonlinear constrained Tikhonov regularization](https://doi.org/10.1088/0266-5611/26/2/025001) (Morozov/discrepancy context).
- Mukherjee et al. (2020), [Learned convex regularizers for inverse problems](https://arxiv.org/abs/2006.10869) (forward/operator-aware learned regularization context).

**Copyright exclusion:** no pirated PDF, full-book repost, scraped chapter text, or unauthorized
upload contributed to this memo.

## STORES CONSULTED

**[MEASURED—read in this pass]:** `CLAUDE.md`; `AGENTS.md`;
`docs/operating_manual_craft_handoff.md`;
`.omx/research/t5_crucible/SPEC_v75_optimal_single_trunk_20260708.md` §8;
`reports/latest.md`; `.omx/state/lane_registry.json`;
`.omx/state/subagent_progress.jsonl`; current Codex summary and latest T3/design routing memos;
`.omx/research/per_epoch_detailed_accounting_20260713.md`;
`.omx/tmp/codex_prompts/p0_sparse_adjoint_20260713.txt`;
`.omx/tmp/codex_prompts/p0_checkpoint_backward_20260713.txt`;
`.omx/tmp/codex_prompts/p0_costate_reuse_gradfree_20260713.txt`;
`.omx/research/DRAFT_derived_optimal_next_run_ml_strategy_20260703T164754Z.md` (#344);
the repository DAG records for #157/#247/#249/#316/#318/#320/#391/#426/#483;
`.omx/research/bousfield_deep_read_20260713.md`; canonical-equation source/registry patterns; and
the legitimate external sources listed above.

## Final pointer-delta honesty

**[MEASURED]** This pass routed theory into five falsifiable arms and one held canonical law. It
did not change a trainer, costate consumer, live P0-arm file, run directory, archive, or evaluator
row. **GROUNDING/MEANS only: the pointer moves only through a receiver-closed, byte-closed exact
row. Pointer delta = 0.**
