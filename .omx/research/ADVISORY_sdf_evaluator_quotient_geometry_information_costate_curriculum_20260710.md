# ADVISORY — evaluator-quotient geometry, information field theory, and costate curricula for the SDF witness line — 2026-07-10

```yaml
artifact_kind: advisory
research_only: true
authority: non_promotional_design_and_falsifier_surface
vehicle_scope: [v7.5.2, v7.5.3, v8, HNeRV_control]
measurement_axes: [contest-CPU_pointer_snapshot, macOS-CPU_advisory, predicted]
owned_file: true
training_authorized: false
dispatch_authorized: false
pointer_move_authorized: false
process_signals_authorized: false
```

## 0. Answer first

The shortest path past the current frontier is not “a better neural video codec” and not a larger
collection of SDF penalties. It is a **receiver-closed compiler for the shortest legal representative
of an evaluator-equivalence class**, with four coupled mathematical layers:

1. a stratified quotient geometry for the five-class argmax partition;
2. a gauge/Hodge-safe field representation whose decoded margins are globally integrable;
3. an individual-sequence task rate-distortion/MDL compiler operating on exact archive bytes; and
4. a hybrid optimal-control curriculum whose true costate is propagated through optimizer and event
   dynamics, rather than inferred only from a trailing score slope.

The strongest new proposals in this pass are:

- **EQM** — Evaluator Quotient Manifold: optimize on the task-effective quotient of archive,
  receiver, scorer, and argmax fibers.
- **RDEC** — Receiver Discrete Exterior Calculus: exact archive mutations form a cell complex on
  which score differences must be conservative; nonzero loop circulation detects apparatus drift.
- **RQTD** — Receiver-Quantized Topological Derivative: price the smallest island or handle that
  survives the real quantize/resize/parse-back receiver.
- **NLSH** — Nonlocal Shape Hessian: measure same-interface, adjacent-interface, and remote-interface
  interactions instead of assuming the scorer is local.
- **FGC-v8** — Flat Gauge Connection: encode gauge-fixed potentials or a spanning-tree connection,
  never unlabelled independent edge fields.
- **PCB** — Pose Connection Bundle: treat the pose carrier as a section over decoded geometry and
  measure its connection curvature/holonomy.
- **ZEQC** — Zero-Error Quotient Compiler: merge receiver/task-equivalent archive symbols and select
  the shortest deterministic representative after full context coding.
- **HAC** — Hybrid Adjoint Curriculum: a saltation-aware discrete adjoint and stage-switch advantage
  layered above the existing read-only costate observer.

These are **proposals**, not canonical laws. Each has a receiver-exact falsifier below. Nothing here
moves a pointer, launches a run, changes a config, or authorizes training.

## 1. Current authority snapshot

Snapshot read at `2026-07-10T17:06:27Z`:

| surface | current authority |
|---|---|
| local frontier | `[contest-CPU] S = 0.19108282419209976` |
| archive | 177,169 bytes |
| archive SHA-256 | `ad02b0124cbb3405c23d3480ac16f12b4e48cbf6f75878dd77a5e621bebd079c` |
| exact components | `d_seg=0.00055961`, `d_pose=0.00002942` |
| score decomposition | Seg `0.055961` (29.286%), Pose `0.0171522593` (8.976%), rate `0.1179695649` (61.737%) |
| corresponding CUDA | **unmeasured for this archive**; the older separate `[contest-CUDA]` pointer remains `0.20533002902019143` |
| public-frontier snapshot | stale/wrong-repository surface in the canonical pointer; not authority for PR128 |
| live v7.5.2 dry-start | governed launcher still alive; first pass checkpointed, resume pass live; no final `dry_start_report.json` at this snapshot |

Exact terminal score covector at this operating point:

\[
\nabla_y S
=
\left(
100,
\frac{5}{\sqrt{10d_{pose}}},
\frac{25}{37{,}545{,}489}
\right)
=
(100, 291.5067866, 6.6585895\times10^{-7}).
\]

The entries have different units and **must not be compared by magnitude alone**. The action-level
quantity is the pullback `J_yu^T grad_y S`, or a receiver-exact finite difference, divided by the real
cost of the action. The current rate term dominates the score level, but `d_seg` may still be the most
controllable training axis; that is an empirical action-Jacobian statement, not a consequence of
`100 > 6.66e-7`.

Reducing 177,169 bytes to 90,000 at fixed task errors would save exactly `0.0580422591 S`. That is
large enough that rate cannot be treated as a terminal cleanup axis. It should enter only after a
complete receiver establishes a nondegenerate task/bit slope, but then it must remain a first-class
stage-boundary dual.

## 2. Already-settled firewall

This advisory does **not** reopen the following settled results:

- the witness is the SDF/level-set task-space vehicle; HNeRV/PR128 is the matched control and a source
  of reusable mechanisms, not the mission;
- the exact score law and separate CPU/CUDA evidence axes;
- the Maslov softmax-to-argmax bound, categorical Fisher/margin identity, CE mirror-descent reading,
  `tau=epsilon=hbar`, multiphase Modica-Mortola limit, mean-curvature minority erasure, or the settled
  curvelet/shearlet upper-bound law;
- existing eikonal, area-Lagrange, temporal-screw, persistence/clDice, satisficing hinge, and
  critical-nucleus laws;
- the existing first Hadamard shape derivative and prior Morse/Conley/microlocal analyses;
- the measured self-orient result, the v7.5.3 registered-off ladder, the v8 increment-1a screen, or
  any prior NCA negative;
- standalone Morse-Smale coding as a rate-dominated full codec;
- any current live-run state, checkpoint, process, or owed16v2 custody.

The new work begins where those surfaces stop: exact quotient geometry, receiver-discrete calculus,
nonlocal second variation, topological insertion price, hybrid adjoints, and costate-controlled
curriculum integration.

## 3. The exact object: an evaluator-equivalence compiler

Let the legal chain be

\[
M \xrightarrow{C} A \xrightarrow{D} X
\xrightarrow{R} \widetilde X
\xrightarrow{E} Y
\xrightarrow{S} \mathbb R,
\]

where `M` is counted model/state, `A` the complete archive, `D` the deterministic receiver, `R` the
exact resize/uint8/raw-reload chain, and `E` the frozen SegNet/PoseNet evaluator. Define

\[
A\sim_E A'
\iff
E(R(D(A)))=E(R(D(A'))).
\]

The compiler objective is not visual fidelity. It is

\[
\min_{A\ \mathrm{legal}}
100d_{seg}(A)+\sqrt{10d_{pose}(A)}
+\frac{25|A|}{37{,}545{,}489}.
\]

Locally, on a smooth fixed-quantization/fixed-argmax stratum, a parameterization
`theta -> y(theta)` induces the degenerate task metric

\[
G_\theta=J_\theta^\top WJ_\theta,
\qquad
J_\theta=\frac{\partial y}{\partial\theta}.
\]

Directions in `ker(J)` are locally task-invisible. The effective manifold is the quotient by that
kernel. Globally, however, uint8 thresholds, archive grammar, argmax ties, and topology changes make
the object a **stratified quotient with discrete jumps**, not one smooth Riemannian manifold.

Consequences:

- SDF, HNeRV, analytic carriers, texture atoms, and pose codes are competing coordinate charts or
  upper bounds on the same task-description problem.
- A smooth Jacobian proposes directions only inside one stratum.
- A receiver-exact finite difference decides whether a direction survives the discrete quotient.
- A useful representation minimizes description length on the quotient, not reconstruction error in
  raw RGB space.

## 4. Partition quotient, braid geometry, and transversality

For `K=5` potentials `phi=(phi_1,...,phi_K)`, the partition `P(x)=argmax_c phi_c(x)` is invariant
under the partition-only gauge

\[
\phi(x)\mapsto a(x)\phi(x)+b(x)\mathbf1,
\qquad a(x)>0.
\]

With `C = I - 11^T/K`, define

\[
q(x)=\frac{C\phi(x)}{\|C\phi(x)\|}\in S^{K-2}=S^3.
\]

The quotient target is stratified by the type-`A_4` braid arrangement

\[
H_{ij}=\{q:(e_i-e_j)^\top q=0\}.
\]

The actual interface is the active top-two part of the hyperplane pullback:

\[
\Gamma_{ij}=q^{-1}\!\left(
H_{ij}\cap\{q_i=q_j>q_k,\ k\notin\{i,j\}\}
\right).
\]

Executable regularity receipts are

\[
\tau_1=\min_{\Gamma_{ij}}\|\nabla(\phi_i-\phi_j)\|,
\qquad
\tau_2=\min_{\Gamma_{ijk}}
\sigma_{\min}
\begin{bmatrix}
\nabla(\phi_i-\phi_j)^\top\\
\nabla(\phi_j-\phi_k)^\top
\end{bmatrix}.
\]

Positive `tau_1` makes pair interfaces regular curves; positive `tau_2` makes triple ties
transverse and isolated. A four-label `2x2` block in two dimensions is not automatically a
stable quadruple junction: it is more likely unresolved nearby triple events or a loss of
transversality.

**Caveat/falsifier:** the full receiver may consume raw potential magnitude for rendering. If
centering or radial normalization changes decoded RGB or exact score, that magnitude is not a dead
gauge of the complete receiver. The quotient is then a partition diagnostic, not a legal codec fold.

Arbitrary class potentials form a tropical upper envelope. They are a classical Laguerre/power
diagram only under the corresponding weighted-site parameterization; the distinction must stay
explicit.

## 5. Gauge, group theory, Hodge decomposition, and v8

Let `B in R^(KxE)` be an oriented incidence matrix of the class adjacency graph and `g in C^1(G)`
the edge-margin field. Global potentials exist when

\[
g=B^\top\phi,
\qquad
\oint_Cg=0\quad\text{for every cycle }C.
\]

For positive edge weights `W`, the best integrable projection is

\[
L_W=BWB^\top,
\qquad
\phi^\star=L_W^+BWg,
\qquad
h=g-B^\top\phi^\star.
\]

The residual `h` is cyclic/nonintegrable debt. The observed connected five-node, nine-edge RAG has

\[
\dim H^1(G)=E-K+1=9-5+1=5.
\]

Thus nine independent edge channels contain five cycle degrees beyond the four independent
translation-quotiented potentials. This proves algebraic overcompleteness, **not** a literal `5/9`
byte saving; support sparsity and entropy still decide bytes.

Safe first receivers:

1. encode centered `K` potentials and derive every edge after decode; or
2. encode `K-1` oriented values on a spanning tree, reconstruct potentials by exact path sums,
   then derive non-tree edges.

Both are flat by construction after quantization. A selected tree must charge its metadata and
measure path-stretch/quantization amplification. Independent cyclic edge state is admissible only
with an explicit deterministic global labeler and a matched-byte exact win.

### Root-compatible eikonal

At a triple junction,

\[
u_{ij}+u_{jk}+u_{ki}=0,
\qquad
\nabla u_{ij}+\nabla u_{jk}+\nabla u_{ki}=0.
\]

Forcing every pair margin to unit gradient privileges 120-degree geometry and is generally
incompatible with heterogeneous measured tensions. The proposed active-root term is

\[
E_{root}
=
\sum_{i<j}\int
\chi^{top2}_{ij}\delta_\epsilon(u_{ij})
\left(F_{ij}^{\!*}(\nabla u_{ij})-s_{ij}\right)^2dx,
\qquad
s_{ij}\propto\sigma_{ij}.
\]

This couples the existing anisotropy/tension surfaces at their missing junction constraint. It must
not be imposed blindly on pinned, data-forced, or unresolved compound junctions.

### Equivariance must commute with the receiver

For a proposed group action `rho_X(g)` in image/geometry space and `rho_Y(g)` in evaluator
space, measure

\[
\mathcal C_g=ER\rho_X(g)-\rho_Y(g)ER.
\]

SE(2), SE(3), scale, or time-shift equivariance is useful only where this commutator is small through
the exact receiver. Group theory is a compression tool only after the scorer/receiver respects the
orbit.

## 6. GMT, topology, and second-order shape calculus

### 6.1 Single-owned interfaces

Treat the partition as a finite-perimeter Caccioppoli partition:

\[
E_{GMT}=\sum_{i<j}\int_{\Gamma_{ij}}
\sigma_{ij}(x,n)\,d\mathcal H^1.
\]

Each shared interface is owned once. Summing independent class perimeters double-counts it. A
differentiable predicted varifold is

\[
V_\phi=\sum_{i<j}
\sigma_{ij}\chi^{top2}_{ij}\delta_\epsilon(u_{ij})
\|\nabla u_{ij}\|\,dx\otimes\delta_{[n_{ij}]}.
\]

Use a deterministic spatial-orientation kernel and a multi-direction Crofton estimator; derive the
quadrature weights from the chosen directions rather than hand-setting an axis-biased TV penalty.

### 6.2 Receiver-realized topology

Let source and decoded fresh-scorer margins be

\[
u_c^\star=z_c^\star-\max_{j\ne c}z_j^\star,
\qquad
\widehat u_c=z_c(R(D(A)))-\max_{j\ne c}z_j(R(D(A))).
\]

Compute cubical persistence on these margins, not only on generator `phi`. A superlevel interval
`(d,b)` is alive in the exact argmax mask when `d < 0 <= b`. The useful signature is

\[
\mathcal T(P)=
\left(
\{\operatorname{Dgm}_{0,1}(u_c)\}_c,
\operatorname{RAG}(P),
\{\text{junction labels and cyclic orders}\}
\right).
\]

If every potential changes by at most `epsilon`, then

\[
\|u_c-\widehat u_c\|_\infty\le2\varepsilon,
\qquad
d_B(\operatorname{Dgm}(u_c),\operatorname{Dgm}(\widehat u_c))\le2\varepsilon.
\]

This is a stability receipt, not a score theorem. Barcode agreement can coexist with threshold-mask
cell changes, so the decoded hard mask remains authority.

### 6.3 Receiver-quantized topological derivative

An ordinary shape derivative cannot create a missing component. Define the proposed exact insertion
price

\[
D_T^R(c,x)
=
\frac{
S(A\oplus I^R_{c,x})-S(A)
}{
|A\oplus I^R_{c,x}|-|A|
},
\]

where `I^R_(c,x)` is the smallest legal class-`c` island at `x` that survives archive compile,
decode, uint8, resize, raw reload, and the fresh scorer. This replaces the continuum
`epsilon -> 0` limit with the receiver's true discrete quantum.

It schedules births by exact score value per byte. It also makes “topology token” economics literal:
one repaired Seg cell pays for only

\[
\frac{100/(600\cdot384\cdot512)}{25/(8\cdot37{,}545{,}489)}
=10.1848657\ \text{bits}=1.2731082\ \text{bytes},
\]

before Pose harm, headers, entropy interactions, or collateral cells.

### 6.4 Nonlocal shape Hessian

The settled first variation is extended to

\[
\delta^2\widetilde D[V,W]
=
\iint K(s,s')V(s)W(s')\,ds\,ds',
\qquad
K\approx K_{local}+U\Lambda U^\top.
\]

The sparse-near plus low-rank-far form is a falsifiable hypothesis motivated by local boundaries plus
global scorer paths. Estimate same-interface, adjacent-interface, and remote-interface factorial
interactions through the full receiver.

Use the Sobolev shape metric

\[
G_\Gamma(V,W)=\int_\Gamma
(aVW+b\partial_sV\partial_sW)\,ds,
\qquad
V=-(a-b\partial_s^2)^{-1}g,
\]

with `b/a` derived from the measured interaction spectrum. An arbitrary smoothing length can erase
Lane dashes.

## 7. Signal, information, and integer receiver theory

### 7.1 Individual-sequence task RD

For counted bit length `L_zip = 8|A|`,

\[
J(M)=100d_s(g(M))+\sqrt{10d_p(g(M))}
+\lambda_bL_{zip}(M),
\qquad
\lambda_b=\frac{25}{8\cdot37{,}545{,}489}.
\]

Define

\[
R_{task}(D_s,D_p)=
\min_{\substack{M,g\ legal\\d_s\le D_s,\ d_p\le D_p}}
L_{zip}(M).
\]

This is an individual-sequence contest object. A numerical Shannon floor is not rigorous without a
declared source ensemble; Kolmogorov/absolute MDL lower bounds are uncomputable. Distributional Fano
or Gaussian bounds may be optimistic theory only, never pointer evidence.

For a complete receiver mutation `U`, the exact break-even value is

\[
V_{bits}(U)=
\frac{
100(d_s^0-d_s^U)+\sqrt{10d_p^0}-\sqrt{10d_p^U}
}{\lambda_b}.
\]

Admit the mutation only when its compiled joint bit increment is smaller than `V_bits`.

### 7.2 Boundary distortion, not coefficient distortion

For a stable `i/j` interface,

\[
|\Omega_{ij}\triangle\widehat\Omega_{ij}|
\approx
\int_{\Gamma_{ij}}
\frac{|\delta(\phi_i-\phi_j)|}
{|\nabla(\phi_i-\phi_j)|}\,ds.
\]

Thus geometry coding should rank atoms by expected contour-normal cell displacement—boundary `L1`,
not coefficient `L2`. Curvelets/shearlets remain a plausible smooth-edge basis; births, junctions, and tiny islands
require separate sparse topology symbols.

The proposed task-adaptive frame solves a generalized eigenproblem

\[
H_{task}v_i=\gamma_iH_{rate}v_i,
\]

where `H_task` is measured from receiver-exact or receiver-calibrated influence and `H_rate`
is a local description-length metric. This is a basis proposal; exact compiled archives decide.

### 7.3 Task-Slepian exact-D atoms

Let `P_D` project into a verified exact Pose-preprocessor kernel and `W_Gamma` select an SDF
annulus. Solve

\[
P_DW_\Gamma P_Dv=\rho v.
\]

Large-`rho` vectors maximize annulus concentration while staying in the linear Pose-null proposal
space. Construct at scorer grid, then solve the bounded integer camera preimage through the exact
polyphase receiver. Camera-grid “high frequency” alone is not a null theorem because resize aliases.

### 7.4 Integer-lattice preimage compiler

One Pose block maps 16 native camera pixels (=48) byte variables to four scorer RGB pixels and then
six YUV features. The local real-linear kernel therefore has dimension at least 42 before bounds,
rounding, and nonlinear downstream effects.

Use the dyadic/fixed-point idealization to generate integer candidates with Smith normal form or LLL:

\[
Aq=b,\qquad q\in\mathbb Z^{48},\qquad 0\le x_0+q\le255.
\]

For frame 1, optimize Seg benefit subject to exact block invariance; for frame 0, choose the
minimum-description preimage for the desired Pose coordinates. IEEE accumulation order, clamp,
rounding, and raw reload mean this lattice is only a proposal generator. Exhaustive source-exact
receiver verification is mandatory.

### 7.5 Conditional coding and zero-error quotient

Geometry, `xi`, and prior pairs are decoder side information. The operational conditional
information saving is

\[
\widehat I=L(C)-L(C\mid G,\xi,C_{<t}),
\]

measured on the actual deterministic coder. Merge symbols only when they are receiver/task-equivalent
under their full entropy context. Because adaptive contexts couple choices, shortest-representative
selection is a joint compiler problem, not a per-symbol nibble trick.

## 8. Neural/Jacobian apparatus

For smooth local proposals, use the scorer pullback/Gauss-Newton metric

\[
G=J^\top WJ.
\]

For parameter blocks `a,b`, measure normalized cross-block coupling

\[
\rho_{ab}=
\left\|G_{aa}^{-1/2}G_{ab}G_{bb}^{-1/2}\right\|_2.
\]

Low `rho_ab` supports independent stages; high `rho_ab` requires a joint arm or an explicit
orthogonalization/projection. This turns “geometry/texture/pose are separate” into a falsifiable
Jacobian statement.

Continuous gradients are proposal fields. The deciding influence matrix is the discrete receiver
finite difference

\[
I_{rj}=S_r(A\oplus\delta_j)-S_r(A),
\]

indexed by exact score component `r` and legal archive mutation `j`. Neural tangent/spectral-bias
analysis may choose candidate bases, but only `I` licenses score economics.

Quantization should use hard-forward, soft/proximal proposal updates and exact parse-back selection.
Straight-through gradients are not receiver authority.

### Exact Seg and Pose curvature caveats

For Pose residual `r(z)` with mean-square distortion `D = mean(||r||^2)`, the local
receiver Gauss-Newton matrix is

\[
G_D\approx\frac{2}{M}J^\top J.
\]

It is a Fisher matrix only after declaring a fixed-variance Gaussian likelihood. The outer square-root
score has curvature

\[
\nabla^2\sqrt{10D}
=
\frac{\sqrt{10}}{2\sqrt D}\nabla^2D
-\frac{\sqrt{10}}{4D^{3/2}}\nabla D\nabla D^\top.
\]

The negative rank-one term couples every pair through the aggregate `D`. Pair-local renders can
therefore be disjoint while pair-local **score gains are non-additive**.

For Seg, the categorical surrogate

\[
F=\operatorname{diag}(p)-pp^\top,
\qquad
G_{CE}=J_{logit}^\top FJ_{logit}
\]

is a legitimate training geometry. The exact Hamming argmax objective is flat almost everywhere and
discontinuous at ties. “Margin is the Fisher metric” is valid only in the declared categorical/local
surrogate domain, never as an exact evaluator identity.

### Exact-D is an intersection, not a frequency label

The useful exact-D space is

\[
\ker L_{preprocess}
\cap\operatorname{reachable}(\text{camera uint8 lattice})
\cap\operatorname{active\ receiver\ chart}
\cap\{\delta x:J_{Seg}\delta x\ne0\}.
\]

The six linear block atoms establish only the first factor. Rank reachable modes through

\[
Z^\top J_S^\top W_SJ_SZv
=\lambda(G_{rate}+\mu I)v,
\qquad LZ=0,
\]

then require exact lift, rounding, clamp, parse-back, and Pose invariance. The current pre-sigmoid,
soft-mask, both-frame texture path cannot inherit an exact-D claim without this intersection receipt.

### Isolation and implicit-gradient limits

Disjoint class heads are necessary but insufficient for v8 isolation. Shared codes move every field,
and CE/top-two margin losses on class `c` have gradients on the competing fields. A true isolation
receipt must include the whole optimizer step and declare one routing mode:

- `global_competition`;
- `owner_projected`; or
- `edge_pair`.

Implicit differentiation

\[
\frac{dL}{d\theta}=L_\theta-L_zH_{zz}^{-1}H_{z\theta}
\]

is valid only for a fixed active set, a locally unique regularized inner optimum, and nonsingular
Hessian. Topology births, quantizer crossings, and argmax chart changes violate those conditions.

Finally, parameter-space natural-gradient invariance is not archive-rate invariance: two
functionally equivalent parameterizations may serialize to different byte lengths. Every geometric
optimizer must still close on exact archive bytes.

### Pose as a fiber bundle

Let decoded geometry `G` be the base and the set of image pairs with equal Pose output be the fiber.
The current `xi` representation is one section, not a sufficiency theorem. A minimum-norm Jacobian
defines a proposed horizontal lift. Its curvature

\[
\mathcal F=d\mathcal A+\mathcal A\wedge\mathcal A
\]

measures path dependence: nonzero holonomy around a geometry loop identifies hard-pair innovations
that a global `xi` chart cannot derive. Official Pose remains the outer gate.

## 9. Receiver discrete exterior calculus

Let legal archive states be vertices of a mutation complex and legal one-symbol mutations be oriented
edges. The exact score is a 0-form; measured mutation deltas are the exact 1-form

\[
(dS)(A\to A')=S(A')-S(A).
\]

For every closed mutation loop `C`, deterministic exact evaluation requires

\[
\oint_CdS=0.
\]

Likewise, mixed second differences must commute:

\[
\Delta_i\Delta_jS=\Delta_j\Delta_iS.
\]

Nonzero circulation or asymmetric mixed differences indicate nondeterminism, stale scorer state,
archive/context mismatch, or custody drift—not exploitable score. Approximate proposal fields can be
Hodge-decomposed into conservative gradient, curl, and harmonic parts; only the conservative part is
trusted for greedy descent until exact loop closure passes.

Apply RDEC recursively at architecture, component, class edge, footprint, symbol, and bitplane scales.
It supplies a precise apparatus test for “recursively fractally optimized”: the same conservation law
must close at every scale.

## 10. Source audit of the existing curricula and costate controller

### 10.1 What is actually live in v7.5.2

The current launch script composes:

- unified `L_tau` via `--seg-form-unify-tau`;
- event-driven tau octaves via `--tau-advance-mode event`;
- Muon on `powerlaw_meat` with epoch 726 as backstop;
- lane band on `lane_nucleus` with epoch 500 as backstop;
- chroma and temporal screw on `annulus_plateau` with epoch 450 as backstop;
- island LADDER, area counter-force, birth completion/ramp, Polyak backstop, head/pose gates;
- checkpoint/resume state for the event controllers.

The run itself logs that the legacy `--curriculum-event-triggered` CE→tau→l7 dispatcher is **inert
under `--seg-form-unify-tau`** because unified `L_tau` dissolves those discrete form boundaries.
Therefore the effective curriculum is not the prose CE/tau/l7 graph. It is the unified loss plus tau
octaves and the separately wired treatment/optimizer/pose events.

The tau controller is mathematically respectable as a continuation scaffold: geometric tau values,
per-octave relaxation, loud caps, full resume state, and freeze-at-Muon. Its open risks are already
visible in source:

- the LR fraction is a step function `k/N`, so every octave event also jumps LR;
- the max-dwell cap is flat although critical slowing should increase late-rung relaxation time;
- `powerlaw_meat` reads one scalar EMA-shadow `d_seg` trajectory, not a hybrid action advantage;
- the event threshold is in `d_seg` units and does not include exact rate/Pose opportunity cost.

### 10.2 The current “costate gates” do not bind

Both live LADDER flags are zero:

```text
--ladder-movable-lambda-gate 0.0
--ladder-lane-lambda-gate 0.0
```

In `LadderArmSpec.gate_multiplier`, zero means always open. The current
`perclass_lambda_proxy = flip_share * d_seg_by_class` is a useful residual-priority statistic, but it
is dimensionless and is not `partial S / partial r_c`, `partial S / partial u_c`, or a dynamic
adjoint. Thus the advertised per-class-costate integration is structurally present but
**non-actuating at the live operating point**.

### 10.3 The shadow observer is valuable but is not yet a Pontryagin costate

The current controller deliberately has `actuation = NONE`. It computes:

- exact score partials with respect to aggregate observables;
- trailing same-stage OLS slopes and `dS/depoch`;
- transition jumps, rollback opportunity, stall alarms, duty-to-measure, priors, and ranked advice.

Those are useful SENSE surfaces. Mathematically, however, `grad_y S` is the **terminal score
covector**, not the costate of optimizer state. The latter must propagate backward through training
dynamics.

Source-level gaps to record rather than smooth over:

1. `transition_jump_costate` currently computes only the Seg jump although its prose mentions Pose.
2. `per_class_within_flip_costates` documents class-area weighting but returns the worst unweighted
   class slope times 100.
3. `rollback_gain` selects the “best” row by minimum `d_seg`, then compares full implied score; the
   full-score minimizer can be different.
4. channel uncertainty is combined under an independence approximation even though Seg, Pose, bytes,
   and epoch share a trajectory.
5. “binding term = d_seg because lambda 100 is largest” compares quantities with different units.
6. action cost is currently a coarse `1` versus `horizon_epochs`, not measured wall time, archive
   bytes, confidence, reversibility, or opportunity cost.
7. the live digest at this snapshot reports 66 owed registered levers, while visible rate candidates
   such as D18 are not registered in the duty queue; the controller therefore does not yet see the
   whole current score-level rate pressure.

These are apparatus findings, not permission to edit the live controller in this advisory turn.

### 10.4 v7.5.3 and v8 are not yet new compiled curricula

- The default v7.5.3 argv is intentionally byte-identical to the GO'd v7.5.2 OFF arm. Its texture,
  widened head, analytic lane training, and 12-rung queue are default-off decisions. It currently
  reuses v7 schedule governance unchanged.
- The compiled v8 surface is increment-1a, a paint-free mask-level measurement screen. It is not a
  trainer or curriculum. The richer v8 stage grammar remains a design surface until its receiver,
  carriers, matched control, and through-R gates exist.

## 11. Hybrid adjoint curriculum (HAC)

### 11.1 State and dynamics

At checkpoint boundary `k`, define a sufficient resume state

\[
x_k=(\theta_k,m_k,v_k,\theta_k^{EMA},q_k,\tau_k,
\mathcal T_k,a_k,f_k,\sigma_k,R_k,p_k^{pose},t_k),
\]

containing model/optimizer/EMA state, discrete stage/rung, topology and area receipts, per-class
flips, conditioning spectrum, exact section bytes, Pose state, and wall time. Between events,

\[
x_{k+1}=F_{q_k}(x_k,u_k,\omega_k).
\]

Controls `u_k` are typed stage-boundary choices only: continuation rung, one new force, optimizer
mode, module engagement, rate budget, or stop/rollback. They are not arbitrary per-step weight
mutations.

### 11.2 True discrete adjoint

For terminal exact objective `Phi(x_N)` and stage costs `ell_k`,

\[
p_N=\nabla_x\Phi(x_N),
\qquad
p_k=\nabla_x\ell_k+left(D_xF_{q_k}\right)^\top p_{k+1}.
\]

The Hamiltonian is

\[
H_k(x,u,p)=\ell_k(x,u)+p^\top F_{q_k}(x,u).
\]

A proposed switch `u -> u'` is admitted only when its robust switching advantage is negative:

\[
\mathcal A_k(u')=
H_k(x,u',p)-H_k(x,u,p)
+C_{switch}(u\to u')+C_{risk},
\]

with an upper confidence bound below zero and receiver-exact confirmation at the next preserved
checkpoint.

### 11.3 Event saltation

An event with guard `h_e(x,t)=0` and reset `x+ = R_e(x-)` has saltation matrix

\[
\Xi_e
=D R_e+
\frac{
(f^+-D R_ef^-)\,n^\top
}{
n^\top f^-+\partial_t h_e
},
\qquad n=\nabla_xh_e.
\]

The adjoint jumps as

\[
p^- = \Xi_e^\top p^+ + \nabla_xc_e.
\]

This is the missing math for tau fires, birth completion, Muon optimizer resets, LR jumps, head
solve, quantization activation, and pose engagement. Fitting one smooth OLS line across a jump loses
event-time sensitivity and reset dynamics.

### 11.4 Bilevel interpretation

The outer problem selects the stage grammar and boundary controls; the inner problem trains to a
checkpointed state:

\[
\min_{u_{0:N-1}}\Phi(x_N(u))
\quad\text{s.t.}\quad
x_{k+1}=F(x_k,u_k).
\]

Use exact unrolling only on short boundary windows, implicit differentiation around settled stages,
or matched finite-difference branches when derivatives are not trustworthy. The HAC observer must
begin read-only. Actuation is a later, separately governed gate.

### 11.5 Distinguish four different lambdas

The repository currently overloads “lambda.” Keep separate typed names for:

1. score covector `grad_y S`;
2. dynamic adjoint `p_k`;
3. constraint duals `mu_c, nu_h`;
4. continuation coordinates `tau, epsilon, beta, r`.

For per-home rate constraints `R_h <= b_h`, update only at exact checkpoint boundaries:

\[
\mathcal L=D_{task}+\lambda_bR_{zip}
+\sum_h\nu_h(R_h-b_h),
\qquad
\nu_h^{k+1}=[\nu_h^k+\eta_k(R_h-b_h)]_+.
\]

Active unconstrained homes should approach

\[
-\frac{\Delta D_{task}}{\Delta R_h}\approx\lambda_b
\]

on matched complete archives. Entropy loss is an instrument, never the dual authority.

### 11.6 Proposed typed records

These names specify missing contracts; none is built by this advisory:

| record | minimum payload |
|---|---|
| `ReceiverChartSpec` | archive/raw/receiver/scorer hashes, clamp mask, top-two map, topology hash |
| `StageDynamicsLinearization` | state/control Jacobians, fit interval, chart hash, uncertainty |
| `AdjointCostateTrace` | terminal covector, backward recursion, event jumps, evidence |
| `SwitchingSurfaceSpec` | source/target mode, Hamiltonian delta, reset cost, confidence, cap disposition |
| `ReceiverGGNTrustRegion` | declared likelihood, block, damping, rank, predicted and exact reduction |
| `TangentNormalProjector` | class edge, margin band, gauge, texture attachment, junction policy |
| `ExactDIntersectionProjector` | linear null, integer lift, active chart, Pose tolerance, Seg floor |
| `ReceiverNTKSpectrumProbe` | block spectra, residual capture, cross-coherence, quantizer survival |
| `TopologyEventTrustRegion` | allowed births, forbidden deaths/merges, clearance, exact recount |
| `ClassGradientRouting` | routing mode plus full optimizer-step isolation receipt |
| `QuantizedImplicitFinisher` | chart hash, hypergradient check, discrete neighbor set, exact ratchet |

## 12. Recommended v7.5.3 curriculum

This is a proposed **event graph**, not a config change:

### V0 — receiver and controllability census

- compile one complete receiver early;
- mutate/remove every counted section;
- require changed raw bytes and the intended evaluator surface;
- record RDEC loop closure, exact bytes, per-section effect, and fresh scorer identity.

No task/bit costate is admitted before this stage.

### V1 — form the coarse partition under topology/area protection

- structured initialization and the existing area/birth machinery;
- class-presence and receiver-realized persistence gates;
- one-sided protection only for missing/subcritical features;
- no texture or new rate pressure until the topology survives parse-back.

Exit when every required class has positive receiver-realized persistence clearance above the measured
quantization/resize envelope and the area dual is no longer compensating a missing class.

### V2 — quotient-normal and root-eikonal geometry

- train active pair margins, not five unrelated unit SDFs;
- measure `tau_1, tau_2`, junction closure, Crofton/varifold receipts;
- use one new geometry force per increment;
- update area/topology duals only at the checkpoint boundary.

### V3 — self-paced continuation

- retain geometric tau octaves;
- switch on a multi-sensor advantage: relaxed exact task marginal, topology clearance, no rising
  hard class, and favorable stage-switch advantage;
- treat the simultaneous tau/LR jump with an explicit rewarm/reset saltation receipt;
- derive late-rung dwell from measured relaxation eigenvalues after data exists, not a guessed law.

### V4 — engage texture after geometry exists

- T-engage is a receiver-realized island/separatrix-stability event;
- measure `rho_GT` before assuming geometry and texture are independent;
- route exact-D/chroma-first only after the integer receiver verifies Pose invariance;
- keep the v7.5.3 registered arms as isolated branches, not a blind composition.

### V5 — turn on rate duals

- require at least one full exact archive and two matched task/bit points;
- allocate by home/class/scale using complete ZIP deltas;
- admit topology tokens and exception bundles with `V_bits`;
- record superadditive interactions as hyperedges rather than forcing a greedy scalar ordering.

### V6 — conditioning-limited finish

- switch optimizer only when the effective quotient-conditioned spectrum and stage advantage support
  it, not because a clock reached 726;
- preserve the Muon backstop and banked rollback;
- run head solve, then exact MC ratchet, then re-evaluate geometry/topology/rate receipts.

### V7 — pose section and final Pareto compile

- engage pose only after separatrix stability and the hybrid conditioning gate;
- condition Pose innovations on decoded `G, xi`, and temporal history;
- rollback unless exact finite Pose improvement beats the banked carrier after bytes and Seg effects;
- compile/prune the complete Pareto envelope and exact-score only byte-closed candidates.

## 13. Recommended v8 curriculum

The first v8 receiver should retain global potentials even while loss/rate ownership is edge-centric.

### E0 — keep increment-1a honest

- paint-free decoupled versus matched-compute control;
- operative seed-spread floor;
- scorer-grid partition and tie-flicker receipts;
- no claim about through-R or rate.

### E1 — gauge-fixed global potentials

- centered `K` potentials; class-isolation training;
- every edge derived from potentials, so holonomy is identically zero;
- record active RAG, interface ownership, `tau_1/tau_2`, and topology.

### E2 — edge-owned geometry

- root-eikonal and single-count GMT energy per active edge;
- per-edge area/topology duals, with junction force balance;
- receiver-quantized birth insertions for missing islands;
- do not advance per-class continuation coordinates independently across a shared edge: both incident
  class switching functions and the junction/topology guard must agree;
- keep paint absent so geometry attribution is clean.

### E3 — receiver and quantization closure

- quantize/parse-back the potentials;
- require zero/charged holonomy, topology clearance, and stable global labels;
- compare `K`-potential receiver with a Hodge-safe spanning-tree codec at matched ZIP bytes;
- independent cyclic edges remain HOLD unless an explicit labeler wins exactly.

### E4 — merge, diff, correct, then texture

- merge class carriers into the global partition;
- measure the exact residual against the frozen scorer;
- correct with task-Slepian/polyphase atoms, chroma-first and luma-reserved;
- treat the geometry-to-paint transition as a receiver-chart reset with a fresh adjoint boundary;
- engage paint/texture only after P-C/generator coverage is established.

### E5 — graph/home rate dual

- allocate bytes over geometry, topology, texture, `xi`, hard-pair innovations, and headers;
- use exact complete-archive slopes and interaction hyperedges;
- select a tree/graph grammar only after metadata and path-stretch are charged.

### E6 — pose connection and terminal compile

- learn/measure the horizontal Pose section over decoded geometry;
- identify nonzero holonomy as a hard-pair innovation signal;
- run exact finisher, byte-close, CPU/CUDA separately, and preserve every rejected candidate receipt.

## 14. Falsifier matrix

| proposal | decisive receiver-exact falsifier |
|---|---|
| EQM quotient | centering/radial normalization changes raw output or score at matched bytes |
| RDEC | a supposedly exact deterministic mutation loop has nonzero circulation after custody is fixed |
| RQTD | predicted birth value does not survive compile/decode/fresh scorer or costs more bits than repaired-cell value |
| FGC-v8 | independent edges plus explicit labeler beat potentials/tree at matched bytes with stable junctions |
| root eikonal | gradient triangles fail closure, tensions violate feasibility, or exact per-class Seg worsens |
| GMT/varifold | geometric discrepancy improves without exact corrected cells |
| receiver persistence | generator topology agrees but fresh-scorer topology or score does not |
| NLSH | remote factorial interactions are noise and sparse-local Hessian predicts exact mutations better |
| task-adaptive frame | generalized-eigenvalue rank does not predict compiled exact value/bit |
| task-Slepian | integer lift changes Pose or fails to localize Seg benefit after exact receiver |
| integer preimage | source-exact float/round/clamp program rejects the SNF/LLL candidate |
| conditional Pose code | `L(C | G, xi, C_<t) >= L(C)` on the actual coder |
| HAC | saltation-aware boundary prediction is no better than matched finite differences or violates resume identity |
| rate dual | entropy/rate proxy falls while final ZIP does not, or ZIP falls while exact score worsens |
| v7.5.3 event graph | composed winner fails the matched lineage at exact full budget; rollback arm by arm |
| v8 graph curriculum | 1a partition gain fails through-R 1b or complete rate remains a frontier wash |

## 15. Highest-value next work, in order

All items below are advisory sequencing. No launch is authorized by this list.

1. Canonicalize the distinction `terminal_score_covector` versus `dynamic_adjoint` versus
   `constraint_dual` versus `continuation_parameter`.
2. Add a read-only source audit test demonstrating that both current LADDER lambda gates are open and
   label the current proxy honestly.
3. Repair the observer math contracts: full-score transition jump, area-weighted per-class aggregate,
   full-score rollback selection, correlated uncertainty.
4. Replace unit-magnitude “binding term” reasoning with action-Jacobian value per measured cost.
5. Register currently unregistered rate actions in the duty-to-measure surface.
6. Build RDEC loop-closure and mixed-difference receipts around exact archive mutations.
7. Add receiver census: every counted section must alter raw bytes and its intended evaluator surface.
8. Measure partition gauge/common-mode/radial energy and full-receiver dependence.
9. Add post-quantization Hodge/holonomy and global-label consistency receipts.
10. Compute `tau_1, tau_2` at generator, decoded field, and fresh-scorer margin surfaces.
11. Add receiver-realized cubical persistence/RAG/junction ledger.
12. Build the smallest RQTD insertion compiler for one class/site, fully parse-back verified.
13. Measure same/adjacent/remote central finite differences for the NLSH kernel.
14. Solve the shape-vs-rate generalized eigenframe from measured finite differences.
15. Implement a read-only hybrid boundary model with guard, reset, saltation, and confidence bands.
16. Backtest it on existing tau/Muon/birth transitions before any actuation discussion.
17. Estimate real stage costs: wall time, bytes, uncertainty, reversibility, and opportunity cost.
18. Add checkpoint-boundary matched micro-branches from one preserved state to identify control
    derivatives without cross-run confounding.
19. Measure `rho_GT, rho_GP, rho_TP` before declaring curriculum separability.
20. Replace simultaneous continuation changes near the discriminant with one-coordinate
    pseudo-arclength predictor/corrector proposals.
21. Add topology/transversality clearance to tau and texture engagement receipts.
22. Establish the first complete task/bit slope before enabling any new rate dual.
23. Compare `K`-potential and Hodge-safe tree v8 receivers with all metadata charged.
24. Build the task-Slepian scorer-grid basis and exact integer camera lift; keep it proposal-only until
    Pose invariance passes.
25. Measure conditional Pose-code savings against `G, xi`, and temporal history.
26. Preserve HNeRV/PR128 as a matched complete-artifact control on bytes, runtime, receiver, and axis.
27. Add bundle interaction tests; route persistent superadditivity into hyperedge candidates.
28. Only after the read-only HAC backtest is green, design a separately governed advisory-to-actuation
    gate with resume serialization and fail-closed rollback.

## 16. Literal dispositions

| item | disposition | reason |
|---|---|---|
| live v7.5.2 governed dry-start/resume | **OBSERVE / DO NOT INTERFERE** | live checkpointed process; final report absent at snapshot |
| v7.5.3 training or ladder actuation | **HOLD** | no training authority; default curriculum is still v7.5.2 governance; costate gates inert |
| v8 training | **HOLD** | compiled surface is a screen, not a receiver-closed trainer; integrability/labeler/through-R gaps |
| v8 increment-1a measurement | **DESIGN-GO, EXECUTION-HOLD** | falsifier is well specified, but this turn has no dispatch authority |
| RDEC / quotient / topology / Hodge audits | **GO FOR READ-ONLY/BUILD PLANNING** | cheapest structural falsifiers; no score claim |
| HNeRV/PR128 | **KEEP AS CONTROL / MECHANISM DONOR** | not the mission; exact complete-artifact comparison remains valuable |
| independent v8 edge payload | **HOLD** | five-dimensional cycle debt on the nine-edge RAG; no explicit global labeler win |
| numerical Shannon-floor claim | **REFUSE** | no declared source ensemble; individual-sequence contest |
| costate auto-actuation | **HOLD** | present observer is SENSE-only and lacks hybrid adjoint validation |
| pointer move / CUDA equivalence claim | **REFUSE** | advisory only; CUDA for the current CPU archive unmeasured |

## 17. Exact remaining blockers

1. The live dry-start has not produced its final machine-readable report; the resume pass is live and
   must not be signalled.
2. The current legacy curriculum event is inert under unified `L_tau`; the effective event graph
   must be documented from executable behavior, not inherited stage names.
3. The live LADDER costate thresholds are zero, and its proxy is not a score/action derivative.
4. The shadow controller has no dynamic optimizer adjoint or event saltation model; this is by design
   for Phase A, but it blocks optimal-control claims.
5. Several observer estimators do not yet match their stated full-score/per-class contracts.
6. Current duty-to-measure coverage does not include all visible rate actions, despite rate being
   61.737% of the current score level.
7. v7.5.3 has no measured winner for its default-off curriculum additions in this advisory turn.
8. v8 has no receiver-closed training implementation, global labeler proof for independent edges,
   measured carrier bytes, or through-R 1b result.
9. Task-Slepian, integer-preimage, nonlocal-Hessian, and pose-connection proposals are unbuilt and
   unmeasured.
10. Exact CPU/CUDA replay remains separate; no axis is inferred from the other.
11. Public-frontier canonical intake remains stale/wrong-repository and cannot certify PR128 state.
12. This turn authorizes advisory Markdown only: no training, dispatch, pointer move, run termination,
    or source/state edits.

## 18. Triality and future wire-in

This artifact is `research_only=true`, so it intentionally does not modify triality surfaces. A future
implementation must land all three legs together:

- **equations:** EQM/RDEC/RQTD/HAC typed equations with established/proposed status and empirical
  anchors;
- **DSL:** typed guards, state names, stage-boundary dual updates, and no invented flags;
- **DAG:** receiver census → quotient/Hodge/topology receipts → hybrid boundary backtest → matched
  branch → complete archive → exact eval.

Every empirical result must feed sensitivity, Pareto constraints, bit allocation, cathedral/autopilot,
continual learning, and a probe-disambiguator when two interpretations remain defensible.

## 19. Primary research anchors

The applications above are new to this vehicle; the mathematical ingredients are grounded in primary
sources:

- hybrid event linearization and saltation: [Kong et al., *Saltation Matrices: The Essential Tool for
  Linearizing Hybrid Dynamical Systems*](https://arxiv.org/abs/2306.06862);
- hybrid adjoint jump analysis: [Corner, Sandu, and Sandu, *Adjoint Sensitivity Analysis of Hybrid
  Multibody Dynamical Systems*](https://arxiv.org/abs/1802.07188);
- Pontryagin/deep-learning control: [Li et al., *Maximum Principle Based Algorithms for Deep
  Learning*](https://arxiv.org/abs/1710.09513) and [Li and Hao, *An Optimal Control Approach to Deep
  Learning*](https://arxiv.org/abs/1803.01299);
- continuation scheduling: [Lin et al., *Continuation Path Learning for Homotopy
  Optimization*](https://proceedings.mlr.press/v202/lin23n.html);
- dynamics-aware curriculum selection: [Zhou, Wang, and Bilmes, *Curriculum Learning by Optimizing
  Learning Dynamics*](https://proceedings.mlr.press/v130/zhou21a.html);
- differentiable topology: [Brüel-Gabrielsson et al., *A Topology Layer for Machine Learning*](https://arxiv.org/abs/1905.12200),
  [Hu et al., *Topology-Preserving Deep Image Segmentation*](https://arxiv.org/abs/1906.05404), and
  [Shit et al., *clDice*](https://openaccess.thecvf.com/content/CVPR2021/html/Shit_clDice_-_A_Novel_Topology-Preserving_Loss_Function_for_Tubular_Structure_CVPR_2021_paper.html);
- Hodge/edge signal processing: [Schaub and Segarra, *Flow Smoothing and Denoising*](https://arxiv.org/abs/1808.02111)
  and [Roddenberry et al., *Signal Processing on Cell Complexes*](https://arxiv.org/abs/2309.01632);
- geometric deep learning and symmetry: [Bronstein et al., *Geometric Deep Learning*](https://arxiv.org/abs/2104.13478)
  and [Finzi et al., *Generalizing Convolutional Neural Networks for Equivariance to Lie Groups*](https://proceedings.mlr.press/v119/finzi20a.html);
- task/indirect rate-distortion: [Harell, De Andrade, and Bajic, *Rate-Distortion in Image Coding for
  Machines*](https://arxiv.org/abs/2209.11694) and [Enttsel and Corlay, *Model-Aware Rate-Distortion
  Limits for Task-Oriented Source Coding*](https://arxiv.org/abs/2602.12866), with this advisory using
  exact individual-sequence MDL rather than claiming an ensemble floor;
- hard quantization: [Guo et al., *Soft then Hard*](https://proceedings.mlr.press/v139/guo21c.html)
  and [Huh et al., *Straightening Out the Straight-Through Estimator*](https://proceedings.mlr.press/v202/huh23a.html);
- multiscale edge representation: [Candès and Donoho, curvelets](https://curvelet.org/papers/Curve99.pdf)
  and [Kittipoom, Kutyniok, and Lim, compactly supported shearlets](https://arxiv.org/abs/1002.2661).

## STORES CONSULTED

`CLAUDE.md` · `AGENTS.md` · Claude Pact memory top-10 · canonical frontier pointer and exact equation
anchors · lane registry · subagent progress · modal ledger · latest findings/session/council/design
surfaces · `ADVISORY_RESTART_HANDOFF_v752_v753_v8_20260710.md` and the completed advisory family ·
`SPEC_v75_optimal_single_trunk_20260708.md` · `SYNTHESIS_v3_v752_20260709.md` ·
`SPEC_v8_perclass_decomposition_20260708.md` · `SPEC_v8.1_20260709.md` ·
`fullstack_fractal_optimal_synthesis_20260710.md` · canonical deep-math and curriculum equations ·
`witness_autoconfig.py` v7.5.2/v7.5.3/v8 builders · `event_wirings.py` · `tau_advance.py` ·
`ladder_homotopy.py` · `costate_estimator.py` · `shadow_controller.py` · current governed launch
scripts/logs/checkpoints · read-only `costate_digest.py --json` · primary sources linked above.
