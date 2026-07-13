# Conditional probability, homotopy, and Lie transport: the temporal rate law

- **Date:** 2026-07-13
- **Role:** SOL xhigh deep-math digger
- **Status:** DESIGN / ANALYSIS; uncommitted for main review
- **Scope:** `research_only=true`; no launch, scorer actuation, live-run mutation, archive mutation,
  or shared-registry edit
- **Authority:** mathematical derivation plus read-only source/code/artifact inspection
- **Pointer delta:** **UNMOVED**. Grounding is **MEANS**; only a receiver-closed exact byte row on a
  declared contest axis can move a pointer.

## Answer first

1. **Temporal-rate decomposition: DERIVED AFTER REPAIR; BROKEN AS THE PROPOSED THREE-TERM
   EQUALITY.** The exact object is a conditional chain rule for a *marked prediction-innovation
   process*. The displayed conjecture omits the rate of quantized `xi` unless `xi` is already side
   information, omits event/no-event mixture weights, under-types the event mark, omits event-branch
   geometry, and assumes without proof that phase plus residual are lossless coordinates.
2. **`Weyl strata = homotopy classes`: PARTIAL, NOT EQUAL.** Homotopy type is one coordinate of the
   Weyl regular-stratum label. Activation/clamp state, scorer argmax cell, orbit/stabilizer type,
   labeled adjacency/junction incidence, and receiver-lattice phase refine it. The correct object is
   a common refinement `Sigma_(kappa,omega,a,r)`, with a many-to-one projection onto homotopy type.
3. **`flip independent of class | margin,xi`: NOT ESTABLISHED; NO AS A UNIVERSAL LAW.** Scalar
   top-1/top-2 margin is the exact logit-space distance to the first argmax facet, but a crossing
   probability also depends on the class-pair-conditioned perturbation scale/direction, local
   scorer Jacobian, and receiver phase. Existing measurements do not estimate the required
   conditional mutual information.
4. **Highest-value actionable:** on the existing n600 cached boundary/phase/xi surface, run one
   cross-fitted codelength disambiguator with two nested contrasts:
   `q00(flip | margin,xi)` versus `q01(flip | margin,xi,class_pair)` for the prize question, then
   `q10(flip | margin,xi,phase)` versus `q11(flip | margin,xi,phase,class_pair)` for the actual
   coder. Report both held-out `Delta L` values, calibrated conditional mutual-information gains,
   model/table overhead, and
   exact conditional-contour bytes per surviving corrected flip against the registered
   `0.65 B/flip` GO bar. This single read-only measurement decides both class-blind waterfilling and
   whether a conditional arithmetic contour coder is worth building.

No claim above is a score result.

## 0. Custody, notation, and proactive recall

### Binding sources consumed

**READ IN FULL / SOURCE-INSPECTED:** `CLAUDE.md`; `AGENTS.md`;
`docs/operating_manual_craft_handoff.md`; `PROGRAM.md`; SPEC-v7.5 section 8 and its settled table;
SPEC-v8; the full-research directive; current canonical frontier/lane/subagent pointers; the latest
required Codex/council/design surfaces; and the named settled rungs below.

**SETTLED, CONSUMED WITHOUT RE-DERIVATION:**

- Weyl #464: the useful constructive invariance atlas is a state-dependent stratified groupoid;
  the phase zero-mode generates a transport symmetry and its Noether charge is the conjugate
  momentum, not the zero-mode coordinate itself.
- Infinite Descent #466: the semantic rate is the kernel-setoid law
  `R_sem = H(U(W))`; a non-fiber-complete constructive atlas pays exactly
  `H(q_G(W) | U(W)) >= 0`.
- Garrett #467: a literal typed semidirect product splits by definition; splitting does not imply
  statistical independence. Its exact chain rule keeps action/atlas, extension-class, and gluing
  debts conditional rather than adding independent marginal rates.
- L87: pair dependence is modeled through `(xi,R)` plus scene events/gauge; the measured covariance
  audit brackets smooth explained fraction rather than proving totality pointwise.
- L85: n600 GT stride-2 spike rate is **MEASURED 0.005318**; **MEASURED 97.7%** are neighbor-majority
  repairable; spikes are dominated by deterministic GT-side sub-pixel advection through the fixed
  sampling lattice, with a **MEASURED 0.00086** real-frame-content existence row below that floor.
- #323: per-class-lambda-gated island homotopy, movable dilation and lane tangent flow are already
  built; their score delta is owed. Uniform amplification and isotropic lane dilation are only
  formulation negatives.
- #307: **MEASURED n600** contour coding costs `0.8201 B/flip` on the witness-alone surface, with
  142,270 components and 44.6% singleton components; this fails the registered `0.65 B/flip` bar.
- #365/R1: `tac.lie` is the translation-first `xi=(rho,omega)` engine; the banked joint-descent dxi
  carrier is **MEASURED 7.2 KB** with `d_pose=0.001610`; the Morse-Smale parallax design remains a
  stratified depth-field route, not a scalar-depth theorem.

### Random variables and sigma-algebra

Let:

- `W_t` be the receiver-relevant labeled partition state at scored time `t`, including whatever
  geometry/appearance coordinates the declared codec needs. A partition-only `W_t` is not silently
  assumed sufficient for the frozen scorer.
- `C_t` be public/already-decoded context: past decoded states, frozen receiver grammar, declared
  apparatus `R`, the current atlas/chart identifiers, and any already-paid public state.
- `X_t = Q(xi_t)` be the **quantized** SE(3) transport datum actually available to the receiver.
  Continuous differential entropy is not substituted for counted bits.
- `kappa_t` be a declared topology signature. The minimal Betti pair is insufficient for a
  five-label partition; a useful signature includes labeled component/hole data plus RAG/junction
  incidence or the labeled Morse-Smale complex.
- `Phi_t` be receiver-lattice sub-pixel phase on the regular boundary worldtubes.
- `Delta_t` be the remaining within-chart innovation after applying the `X_t` transport and phase.
- `E_t` be a **marked prediction-break variable**. `E_t=0` means the next state stays in the current
  regular predictive chart. A nonzero value must carry the event family, class/edge, location/time,
  attachment/incidence, and any other information not receiver-derived; a binary bit is generally
  not a full event code.

Every entropy below is conditioned on the declared public receiver data even when that suffix is
suppressed typographically.

## 1. The exact temporal rate decomposition

### 1.1 The chain rule that actually closes

Assume a declared chart supplies a lossless measurable coordinate map

\[
  (W_{t+1},X_t)\longleftrightarrow
  \begin{cases}
    (X_t,E_t=0,\Phi_t,\Delta_t^0),&\text{regular branch},\\
    (X_t,E_t\ne0,\Phi_t,\Delta_t^1),&\text{marked-event branch}.
  \end{cases}
\]

Then the ideal joint temporal source rate is **DERIVED-EXACT**:

\[
\boxed{
\begin{aligned}
R_{\rm temporal}^{\rm ideal}
&:=H(X_t,W_{t+1}\mid C_t)\\
&=H(X_t\mid C_t)
 +H(E_t\mid X_t,C_t)
 +H(\Phi_t\mid E_t,X_t,C_t)
 +H(\Delta_t^{E_t}\mid\Phi_t,E_t,X_t,C_t)\\
&=H(X_t\mid C_t)+H(E_t\mid X_t,C_t)\\
&\quad+\sum_e\bar p_e\!\left[
 H(\Phi_t\mid E_t=e,X_t,C_t)
 +H(\Delta_t^e\mid\Phi_t,E_t=e,X_t,C_t)
 \right],
\end{aligned}}
\tag{1}
\]

where `bar p_e=P(E_t=e)` is the ensemble branch mass; equivalently, at fixed decoded context
`C_t=c`, use `p_e(c)=P(E_t=e|C_t=c)` and average over `c`. The first equality avoids any ambiguity
about how conditional branch masses are averaged. If the nonzero event mark already contains all
event geometry, its entropy is the second term. If `E_t` is only a binary indicator, add a marked
payload `J_t` and the event branch contains
`P(E=1) H(J_t | E=1,X_t,C_t)`.

If `X_t` has already been paid and is available as side information, the incremental witness rate
is simply

\[
R_{W\mid\xi}^{\rm ideal}=H(W_{t+1}\mid X_t,C_t),
\tag{2}
\]

which is equation (1) without `H(X_t|C_t)`. If `X_t` is deterministically recoverable from other
already-coded variables, that term is zero. Otherwise the Lie datum is not free.

### 1.2 Precisely where the proposed three-term formula breaks

The conjecture

\[
H(\text{homotopy events})+H(\xi\text{-residual}\mid\text{no event})
+H(\text{sub-pixel phase}\mid\cdots)
\]

is a valuable mnemonic, but not an equality for six independent reasons:

| break | status | repair |
|---|---|---|
| `xi` has no rate term | **DERIVED defect** | Add `H(Qxi_t|C_t)`, or explicitly declare it side information. |
| event entropy is unconditional | **DERIVED defect** | Use `H(E_t|Qxi_t,C_t)`; predictability is the whole point of temporal coding. |
| no event probabilities | **DERIVED defect** | Multiply branch entropies by `p_0` / `p_e`, or state that the quantity is per-branch rather than per time step. |
| “event” is only a topology-change bit | **DERIVED defect** | Code the full mark: kind, class-edge, spacetime address, attachment/incidence, and any nonderived geometry. |
| event branch has no residual/phase cost | **DERIVED defect** | Retain event-branch geometry. A birth flag does not specify the newborn component. |
| phase/residual are presumed coordinates | **ASSUMED in the conjecture** | Prove a receiver-valid bijective chart or label the expression an upper-bound/model codelength. |

An entropy is an ensemble lower bound on expected uniquely decodable length. It is not a pointwise
bound on one Brotli/ZIP archive. Exact archive bytes, receiver parse-back, and scorer survival remain
separate gates.

### 1.3 Noether structure prices sensitivity, not bytes by itself

On the transport subaction, Weyl #464 derived the phase-shift symmetry and momentum

\[
\pi_b=\partial\mathcal L/\partial(D_t^\xi\phi_b),\qquad
Q_b=\int_{\Gamma_b}\pi_b\,ds,
\qquad D_t^\xi Q_b=0
\]

between symmetry-breaking events. **DERIVED:** a conserved charge is receiver-derivable from initial
data only after the initial charge and transport law are available. It can then remove per-frame
repetition. **DERIVED correction:** `Q_b` is not a codelength and not the phase coordinate `c_b`; it
is conjugate momentum. The counted ideal price of the Lie datum is still
`H(Qxi_t|C_t)`, and the price of a broken phase symmetry is the conditional entropy of its
innovation/zero-mode selection. Noether supplies canonical dual coordinates and a rate-distortion
sensitivity metric; a quantizer and a source distribution are still required to produce bits.

**Verdict scope:** `FORMULATION`. This rejects “conserved charge automatically means zero archive
bytes,” not the use of the charge as a predictor or bit-allocation costate.

## 2. Event ontology: topology is one cause, not the event set

### 2.1 The continuous topology theorem

For a class pair, let

\[
f_{cd}(x,s)=\phi_c(x;\theta(s))-\phi_d(x;\theta(s)).
\]

On a compact spacetime slab, if zero remains a regular value of every active `f_cd`, all junction
maps remain transverse, no component crosses an undeclared domain boundary, and the label
incidence pattern remains fixed, the implicit-function flow supplies an isotopy. Therefore the
labeled partition topology is constant on that interval. A generic topology change requires entry
into the discriminant

\[
\mathcal D_{cd}=\{(x,s): f_{cd}(x,s)=0,\ \nabla_x f_{cd}(x,s)=0\},
\tag{3}
\]

or a higher-order junction rank failure. A nondegenerate Morse crossing changes the local handle
structure: extrema give component/hole births or deaths; saddles give merge/split or hole events.

This is **DERIVED** for the continuous labeled field under the stated regularity/transversality
conditions. It is not a theorem about a sampled uint8 argmax movie without a declared complex.

### 2.2 The prediction-break event is a disjoint refinement

Define the marked event family by priority:

\[
\boxed{
E_{\rm pred}
=E_{\rm top}\ \dot\cup\
(E_{\rm chart}\setminus E_{\rm top})\ \dot\cup\
(E_R\setminus(E_{\rm top}\cup E_{\rm chart})).
}
\tag{4}
\]

- `E_top`: latent continuous labeled-topology changes: component/hole birth/death, merge/split,
  junction incidence change.
- `E_chart`: the no-event predictive chart fails while latent topology remains fixed: stabilizer or
  admissible-arrow change, occlusion/disocclusion, nonrigid residual, clamp/ReLU/argmax-cell change,
  or a required atlas transition.
- `E_R`: the continuous state remains in the same chart, but the fixed resize/uint8/sampling lattice
  crosses a receiver cell and changes the observed label/phase symbol.

With a no-event chart that fixes topology, `E_top` is a subset of `E_pred`. The converse is false.
The v8 event law is therefore **DERIVED at the predictor-innovation scope**: code only what the
declared transport/chart cannot derive. It is **not** derived as “code exactly the homotopy
transitions.”

### 2.3 Reconciliation with L85 flicker spikes

L85 measured GT single-scored-frame spikes at **0.005318**, with **97.7%** neighbor-majority
repairability, and attributed the dominant mechanism to sub-pixel boundary advection through the
fixed no-AA sampling comb. That is not a contradiction:

- In the **latent continuous partition**, a boundary can slide across a pixel center and back with
  constant `pi_0`, `pi_1`, RAG, and junction incidence. The spike is `E_R` / phase innovation, not
  `E_top`.
- In a **digital cubical complex**, an isolated one-pixel label can create a sampled component, so
  some spikes are digital `pi_0` changes. That “topology” is gauge/scale dependent and must not be
  silently identified with scene topology.
- Many flips do not change any Betti number; some latent topology changes can remain sub-pixel and
  cause no scorer flip. Thus spike support and topology-transition support are neither equal nor
  mutually determining.
- The spikes are deterministic given the full source appearance and apparatus. Their conditional
  entropy can approach zero with sufficient phase/appearance state—the **MEASURED 0.00086**
  existence row proves the smooth-label floor is pierceable. “Irreducible event code” is always
  relative to `C_t`, not metaphysical randomness.

**Verdict:** `E_spike = E_top` is **NO-GO, FORMULATION x TOPOLOGY-SCALE**. Reformulate as equation
(4) and record whether topology is latent-continuous or receiver-digital.

### 2.4 Is the between-event entropy phase-only?

The exact no-event term is

\[
H(W_{t+1}|E=0,X_t,C_t)
=H(\Phi_t|E=0,X_t,C_t)
+H(\Delta_t^0|\Phi_t,E=0,X_t,C_t).
\tag{5}
\]

It is phase-only **iff** the second term is zero. That requires, at minimum:

1. `W_t` contains sufficient boundary geometry/appearance and the relevant depth/staticness state;
2. `X_t` determines the physical transport on the stratum, including parallax where needed;
3. no occlusion, nonrigid motion, semantic relabeling, or local shape deformation remains;
4. `R` and all quantization are deterministic receiver-known maps; and
5. phase plus the transported state uniquely reconstruct the next state.

These conditions are plausible on restricted ground/sky/static regular strata. They fail globally:
SE(3) camera motion alone does not determine depth parallax, movable motion, disocclusion, local
shape deformation, or scorer-response drift. The covariance audit's lower/upper explained bracket
and event/gauge residual taxonomy support the decomposition, but do not measure
`H(Delta^0|Phi,E=0,X,C)=0`.

**Verdict:** `between-event entropy = phase only` is **INFERRED/CONDITIONAL on restricted strata**,
not DERIVED globally. Defining every non-phase residual to be an “event” would make it true by
definition and therefore non-predictive; the event predicate must be fixed before measurement.

## 3. Are the Weyl strata exactly homotopy classes?

No. The identification is useful but only as one projection.

Weyl #464 explicitly used a regular stratum with fixed Seg labels, fixed ReLU/clamp active set, and
fixed boundary topology. Its constructive groupoid also changes isotropy rank when boundary
components appear/disappear. Therefore a robust stratum label has the form

\[
\boxed{
\sigma=(\kappa,\omega,a,r),
}
\tag{6}
\]

where:

- `kappa`: labeled embedded-isotopy/Morse-Smale signature, not merely unlabeled Betti numbers;
- `omega`: orbit/stabilizer/admissible-arrow type of the constructive groupoid;
- `a`: scorer activation, clamp, tie, and local statistic-fiber chart;
- `r`: receiver-lattice/uint8 phase cell.

There is a many-to-one map `pi_hom:sigma -> kappa`. Two translated disks have the same abstract
homotopy type but different receiver phases and possibly different scorer activation cells. Two
five-class partitions can share every per-class Betti number yet differ in RAG/junction incidence.
Two states with identical labeled topology can have different stabilizers or legal null arrows.

The clean unification is:

```text
Weyl regular groupoid restricted to Sigma_(kappa,omega,a,r)
                         |
                         +--> homotopy projection kappa
                         +--> orbit/stabilizer projection omega
                         +--> scorer chart projection a
                         +--> receiver phase projection r
```

Topology transitions force a boundary between `kappa` strata under the continuous regularity
conditions of section 2.1. Weyl-stratum changes also occur with `kappa` fixed. Hence the homotopy
stratification is a coarsening/factor of the Weyl stratification, not the same partition.

**Verdict:** **PARTIAL / NO exact identification.** `verdict_scope=FORMULATION`. The family remains
open through the common-refinement groupoid (6).

## 4. The conditional flip law and the sufficiency question

Let `F` be the next receiver-visible argmax flip, `C` the directed class pair (winner, runner-up),
`M` the top-1/top-2 margin, `X=Qxi`, and `Phi` the local receiver phase. The proposed simplification
is

\[
F\perp C\mid(M,X)
\quad\Longleftrightarrow\quad
I(F;C\mid M,X)=0.
\tag{7}
\]

### 4.1 What is already established

- **MEASURED n600:** top-1/top-2 margin is the exact logit-space distance to the first argmax
  facet over all 118M scored pixel instances; third and lower logits cannot be the first crossing.
- **MEASURED:** Fisher curvature aligns with negative margin at Pearson `0.978` on the established
  surface.
- **MEASURED n200 advisory:** raw margin predicts actual witness-render flips with AUC `0.7774`;
  the multi-class Chernoff surrogate is worse at `0.7303`.
- **MEASURED aggregate:** Lane carries about 19% flip mass at 0.59% area, while per-class flip
  density spans roughly 120x. This is unconditional and may partly reflect different
  `P(M|C)` distributions.

None of those rows estimates (7).

### 4.2 Why margin is not a probability-sufficient statistic by theorem

Locally, a flip occurs when a perturbation crosses the active facet:

\[
F=1\left\{n_C^T\delta z\le -M\right\}.
\tag{8}
\]

Even with fixed `M=m` and `X=x`, the probability is

\[
P(F=1|m,x,C=c)=P(n_c^T\delta z\le-m\mid x,c).
\tag{9}
\]

It depends on the class-pair normal and the conditional projected perturbation law. A simple
counterexample is class-dependent Gaussian scale:
`n_c^T delta z ~ N(mu_c(x),sigma_c(x)^2)`, which yields a class-dependent Gaussian tail at the
same margin. Therefore scalar distance cannot imply (7) unless the normalized perturbation law is
also class-invariant.

Relevant omitted variables include directed class pair, `xi`-projected normal velocity, receiver
phase, local margin-gradient norm/Fisher scale, boundary tangent frequency/curvature, and
occlusion/staticness. The smallest plausible near-sufficient statistic is therefore closer to

\[
S_{\rm flip}=(M/\hat\sigma_{C},\ n_C^Tv_\xi,\Phi,\kappa_{\rm local},C),
\tag{10}
\]

not scalar margin alone.

### 4.3 Verdict and decisive probe

**DERIVED:** universal class-conditional independence does not follow from margin exactness.
**EMPIRICAL:** `I(F;C|M,X)` is **UNKNOWN** on the current n600 surface. The large class-density
spread is evidence to test, not proof after conditioning.

The decisive no-launch probe is one cross-fitted proper-codelength run with two nested contrasts.
The first directly tests equation (7):

\[
\widehat{\Delta L}_{MX}
=\sum_{i\in\mathrm{heldout}}
\left[-\log_2q_{00}(F_i|M_i,X_i)
+\log_2q_{01}(F_i|M_i,X_i,C_i)\right].
\tag{11}
\]

The second replaces `q00,q01` with
`q10(F|M,X,Phi),q11(F|M,X,Phi,C)`. With calibrated flexible models and held-out evaluation,
`Delta L_MX/N` approaches `I(F;C|M,X)` while the phase-aware contrast estimates the class gain the
actual conditional coder can still exploit. Report bootstrap uncertainty, per-class-pair
calibration, and the bytes required to signal tables/model parameters. Add a third diagnostic
ablation with normalized local Jacobian/velocity; otherwise class may merely proxy a missing scale.

Admission rule:

- class-blind simplification is supported only if the upper confidence bound on net saved bits,
  after table/model overhead, is nonpositive and per-pair calibration residuals are controlled;
- a positive class-aware gain feeds the bit allocator and conditional contour grammar;
- either result remains advisory until an exact packed receiver A/B.

`verdict_scope=FORMULATION x EMPIRICAL-SURFACE`; no class or margin family is killed.

## 5. Conditional arithmetic coding of boundary evolution

Let `A_t` be a receiver-canonical boundary symbol stream and let

\[
\widetilde A_{t+1}=A_{t+1}\ominus T_{X_t}(A_t)
\]

be the innovation after Lie transport. For a true conditional model `p`, the ideal rate is

\[
R_{\rm contour}^{\rm ideal}
=H(\widetilde A_{t+1}|A_t,X_t,R,\sigma_t).
\tag{12}
\]

For an arithmetic/range model `q`, the expected modeled length is

\[
L_q=E[-\log_2q(\widetilde A_{t+1}|A_t,X_t,R,\sigma_t)]
=H_p+ D_{KL}(p\|q),
\tag{13}
\]

before finite coder, table, header, class, event-mark, and container overhead. Thus conditioning pays
exactly by mutual information when the same alphabet/grammar is held fixed:

\[
H(A_{t+1})-H(A_{t+1}|A_t,X_t)=I(A_{t+1};A_t,X_t).
\tag{14}
\]

The proper innovation alphabet should distinguish:

1. small normal displacement / phase symbols on regular transported arcs;
2. topology marks: birth/death, merge/split, hole/junction events;
3. chart/occlusion/movable events;
4. receiver-cell crossings;
5. literal residual pixels only as an escape code.

### Economics against #307

- **MEASURED #307:** `0.8201 B/flip = 6.5608 bits/flip`, n600 witness-alone.
- **DERIVED score-law break-even:** `1.27311 B/flip` if every coded flip survives receiver parse-back
  with no collateral or other overhead.
- **REGISTERED engineering GO bar:** `0.65 B/flip = 5.2 bits/flip`, retaining roughly 2x safety for
  survival/collateral/container uncertainty.

Therefore #307 fails the registered gate but does not prove the conditional family dead. A
transport-conditioned grammar must save at least `0.1701 B/flip` versus #307 before overhead merely
to touch the safety bar. The dominant #307 cost was component anchors, so the highest-value route is
not a better 8-direction code; it is prediction that removes/reuses component anchors and reserves
new anchors for marked events.

**Verdict:** `WORTH-A-PROBE`, not yet `WORTH-A-BUILD`. `verdict_scope=FORMULATION` for the old
unconditional fragmented flip-string coder. Reactivation requires held-out cross entropy below the
bar including model/table overhead, then an exact receiver-packed A/B.

## 6. Tau continuation and homotopy transitions

### 6.1 The crucial correction: temperature alone does not move argmax topology

For fixed logits/fields and positive temperature,

\[
\arg\max_c \phi_c(x)=\arg\max_c \phi_c(x)/\tau.
\tag{15}
\]

In a binary sigmoid, the `0.5` decision boundary is likewise `phi=0` for every positive `tau`.
Therefore the evaluator partition topology is **independent of tau at fixed theta**. Tau annealing
is a homotopy continuation of the *relaxed objective and optimization path*, not a literal motion of
the argmax tie set by itself.

This means a birth schedule cannot be derived from `tau` alone. Topology changes along the learned
path `theta(tau)` when

\[
f_{cd}(x;\theta(\tau_*))=0,
\qquad \nabla_x f_{cd}(x;\theta(\tau_*))=0,
\tag{16}
\]

or a junction transversality condition fails. Away from this discriminant, the isotopy result in
section 2.1 holds.

### 6.2 What can be derived into a schedule

At preserved stage checkpoints, measure persistence pairs and signed critical values of the active
class-difference fields. For a nondegenerate tracked critical point, extrapolate the next zero
crossing only inside a registered trust region; prime the eikonal/retention controller before the
predicted crossing; accept the rung only after the component/junction certificate survives `R`.
This produces a **path-conditioned birth/retention window**, not a universal tau epoch.

The #323 per-class-lambda gate remains necessary because tau continuation supplies no source term
for a component absent from the current basin. It can protect/continue an existing nucleus; it does
not, by equation (15), generate one. The settled Chan-Vese/seed/source machinery is therefore not
superseded.

**Verdict:**

- tau as an objective continuation coordinate: **DERIVED/SETTLED**;
- tau alone as a homotopy birth clock: **NO-GO, FORMULATION**;
- discriminant/persistence tracking along `theta(tau)`: **DERIVED CONDITIONAL METHOD**, empirical
  thresholds owed;
- engineered lambda gates: retained until a measured path-conditioned controller closes their job.

## 7. SE(3) coadjoint orbits and the `dxi 6+k` question

### 7.1 Repository convention and dual action

`tac.lie` fixes the translation-first twist `xi=(rho,omega)` and

\[
\operatorname{Ad}_{(R,t)}=
\begin{bmatrix}R&[t]_\times R\\0&R\end{bmatrix}.
\]

Pair a dual momentum/wrench `mu=(p,L)` by
`<mu,xi>=p dot rho + L dot omega`. Directly dualizing the source-inspected adjoint gives

\[
\boxed{
\operatorname{Ad}^{*}_{(R,t)}(p,L)
=(Rp,\ RL+t\times Rp).
}
\tag{17}
\]

For `p != 0`, the two Casimirs are

\[
C_1=\|p\|^2,\qquad C_2=p\cdot L.
\tag{18}
\]

The generic coadjoint orbit is four-dimensional; the two Casimirs label the orbit. The special
`p=0,L!=0` orbit is a two-sphere labeled by `||L||`. This agrees with the primary Euclidean-group
orbit classification consulted below.

For the **adjoint** action on twists, the analogous screw invariants are
`||omega||^2` and `omega dot rho`; their ratio gives screw pitch away from `omega=0`. Pure
translation is a separate singular chart. Adjoint screw invariants and coadjoint momentum
invariants are related structures, not interchangeable variables.

### 7.2 Coding verdict

**DERIVED:** coadjoint orbits provide a canonical symplectic geometry for the momentum/costate and
separate two invariant labels from four within-orbit coordinates in the generic case. They do not
compress an arbitrary SE(3) pose increment below its six local degrees of freedom unless some
Casimirs, orbit coordinates, or dynamics are already public/predicted.

For a single fixed screw, a six-coordinate local chart can be read as axis direction (2), axis
moment (2), pitch (1), and displacement/angle (1), with degeneracies at pure translation and zero
rotation. A sequence then additionally needs its scalar time law or spline controls and any
chart-change/event marks. The repository's cumulative SE(3) B-spline is exactly the appropriate
kind of temporal chart, but its controls have measured rate only when emitted and packed.

Therefore:

- the **6** in `dxi` is the local dimension of SE(3)/the six scored pose outputs, not a coadjoint
  compression theorem;
- `k` is not derived by orbit theory. It must pay for trajectory curvature/control points,
  parallax/nonrigid residual, gauge phase, or receiver/event corrections actually left after
  conditioning;
- Noether/costate `mu` supplies the rate-distortion dual metric for quantizing `xi`; it is not an
  extra pose coordinate unless the receiver must store it independently;
- near `omega=0`, screw pitch is singular, so a codec needs separate rotation/translation charts
  and a marked chart transition.

**Verdict:** coadjoint structure is **FEED-POSE-CODER** as a coordinate/quantizer design, but
`coadjoint orbit => dxi 6+k` is **NO-GO, FORMULATION**. Exact `k` remains an empirical
rate-distortion outcome.

Primary mathematical cross-checks consulted:

- Philip Arathoon and James Montaldi, *Adjoint and coadjoint orbits of the Euclidean group*, MIMS
  EPrint 2015: `https://eprints.maths.manchester.ac.uk/2292/`.
- Philip Arathoon and James Montaldi, *Hermitian flag manifolds and orbits of the Euclidean group*,
  arXiv:1804.09463.

No theorem from those papers is presented as an archive-byte result.

## 8. Canonical law, DAG feed, and triality

### Equation leg

The isolated proposed equation is `marked_temporal_transport_rate_law_v1` in
`.omx/research/condprob_homotopy_lie_temporal_rate_equation_feed_20260713.md`.

### DAG leg

`.omx/research/condprob_homotopy_lie_DAG_FEED_20260713.md` routes continuous topology,
groupoid-stratum custody, receiver phase, conditional flip testing, and exact byte closure.

### DSL leg

No trainer flag or shared DSL file was invented. A future typed record should carry:

```text
context_hash / xi_source / xi_quantizer / xi_side_information_status
topology_complex / topology_scale / kappa_before / kappa_after
weyl_stratum_before / weyl_stratum_after
event_family / event_mark / event_address / event_derivability
phase_chart / residual_chart / coordinate_bijection_status
class_pair_conditioning / heldout_codelength_bits / model_overhead_bits
contour_bytes / surviving_flips / bytes_per_surviving_flip
receiver_section_hash / parseback_status / axis / evidence_label / verdict_scope
```

`FORMALIZATION_PENDING`: the shared canonical equation/DAG/DSL registries are live sibling surfaces
and were intentionally untouched. This analysis lands only isolated review feeds.

### Six hooks

1. **Sensitivity map:** use conditional surprise `-log2 q(flip|context)` and the normalized
   class-pair facet scale, not raw margin alone, if the probe shows residual class information.
2. **Pareto constraint:** require exact receiver survival plus `B/flips < 0.65` for the registered
   safe contour arm; keep the exact `1.27311` physical break-even visible separately.
3. **Bit allocator:** allocate across `xi`, regular phase, marked topology/chart events, and escape
   residual by measured conditional marginal bits per score unit.
4. **Cathedral/autopilot:** no dispatch from this memo. Block on untyped topology scale, unproved
   coordinate bijection, missing model overhead, or missing receiver section.
5. **Continual learning:** the class-aware codelength delta and each event-family byte row become
   typed posterior anchors; a null result closes only that conditioning formulation.
6. **Probe disambiguator:** required modes are class-blind versus class-pair-aware flip models, and
   latent-continuous versus receiver-digital topology labels.

## 9. Verdict ladder and reformulation queue

### Laws that land

- **DERIVED-EXACT:** conditional chain rule (1), subject to a declared lossless coordinate chart.
- **DERIVED:** topology is constant between discriminant crossings under regularity/transversality.
- **DERIVED:** homotopy is one factor/coarsening of the Weyl regular-stratum label.
- **DERIVED:** scalar margin is distance-to-facet, not probability sufficiency.
- **DERIVED:** tau alone does not move fixed-theta argmax topology.
- **DERIVED:** SE(3) coadjoint Casimirs organize momentum orbits but do not create a byte reduction.

### Scoped negatives

1. Three unweighted terms as an exact total rate: **NO-GO, FORMULATION**.
2. `event set = latent homotopy-transition set`: **NO-GO, FORMULATION x TOPOLOGY-SCALE**.
3. `Weyl strata = homotopy classes`: **NO-GO exact; PARTIAL factor relation**.
4. `flip independent of class | margin,xi` as a universal theorem: **NO-GO, FORMULATION**;
   empirical approximate independence remains UNKNOWN.
5. tau as a standalone birth clock: **NO-GO, FORMULATION**.
6. coadjoint orbit as a derivation of `6+k` archive coordinates: **NO-GO, FORMULATION**.

No FAMILY or PARADIGM is killed.

### Reformulation queue

1. Freeze the marked event ontology and topology scale before measuring entropy.
2. Run the one n600 conditional-codelength probe (equation 11).
3. If class conditioning pays after overhead, use it in both the flip waterfill and transported
   contour grammar; otherwise simplify to the class-blind normalized model.
4. Estimate event-family conditional entropy and anchor reuse after `xi` transport.
5. Emit an exact receiver-packed contour A/B; reject/retain by the registered `0.65 B/flip` bar.
6. Track persistence/discriminant crossings along preserved tau-rung checkpoints; do not infer
   births from tau alone.
7. Only after receiver closure may any result feed pointer review.

## 10. Self-adversarial audit and pointer honesty

- The elegant conjecture was not accepted as an equality; every missing chain-rule term is named.
- `xi` is charged unless explicitly side information or receiver-derived.
- Topology events are separated from chart and receiver-lattice events.
- L85 deterministic spikes are reconciled without calling them aleatoric or latent topology by fiat.
- Digital and continuous topology are not conflated.
- Phase-only transport is conditional, not promoted from the covariance taxonomy to zero entropy.
- Weyl orbit/stabilizer strata are not collapsed into Betti classes.
- The 120x class-density spread is not used as a fake conditional-independence rejection.
- Margin exactness is kept at distance-to-facet scope; probability needs a perturbation law.
- #307's `0.8201 B/flip` is not called a family kill and is compared to both the exact and registered
  engineering thresholds.
- Tau's fixed-theta argmax invariance blocks a fake tau-only birth schedule.
- Adjoint screw invariants and coadjoint Casimirs are not conflated.
- No entropy lower bound is called an individual ZIP-byte measurement.
- No launch, scorer, evaluator, archive, live run, sibling file, or shared registry was touched.

**Pointer delta: UNMOVED.** This unit produced mathematical and routing artifacts only.
