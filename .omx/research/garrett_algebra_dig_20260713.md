# Garrett algebra dig: extensions, orbit rate, representations, modules, and fixed payloads

- **Date:** 2026-07-13
- **Status:** DESIGN / ANALYSIS; uncommitted for main review
- **Scope:** `research_only=true`; no launch, scorer actuation, live-run mutation, or shared-registry edit
- **Authority:** mathematical derivation plus read-only source/code/artifact inspection
- **Pointer delta:** zero. Algebraic grounding is **MEANS**; only a receiver-closed exact byte row can move the pointer.

## Checkpoint 0 — custody and anti-collision

Read fully before analysis: `CLAUDE.md`, `AGENTS.md`, and
`docs/operating_manual_craft_handoff.md`. Read the canonical Weyl memo, its equation and DAG
feeds, the v7.5/v8 specifications, the now-landed foundations memo, the exact
#461 cross-tensor receipt/equation, the #140 pose surfaces, and the current CGauge covariance
law/code. The only files this unit may create are its own memo/equation/DAG-feed artifacts.

Epistemic vocabulary is binding throughout: **MEASURED**, **DERIVED**, **INFERRED**, and
**ASSUMED**. Every negative gets a `verdict_scope`. Existing empirical rows are recalled, not
re-measured. No result in this memo is a score claim.

## Checkpoint 1 — source custody

**SOURCE-INSPECTED:** Paul Garrett, *Abstract Algebra*, especially §§2.6 (group actions,
stabilizers, orbits, counting), 2.8 (automorphisms and semidirect products), 22.5 (the Galois
correspondence), and 27.2–27.5 (tensor products, induced maps, naturality). Garrett's book itself
explicitly says that representation theory appears only in hints, so the representation audit also
uses Garrett's companion notes *Generalities on representations of finite groups* for complete
reducibility, intertwiners, Schur's lemma, tensor products of representations, and isotypes. Group
cohomology and stratified-groupoid descent are not developed in the supplied book; the factor-set
and gluing formulas below are derived directly rather than falsely attributed to Garrett.

The official PDF was available through indexed PDF text. Shell download failed closed on sandbox
DNS and the PDF screenshot endpoint returned a cache miss; no downloaded source or bulky scratch
remains in the repository.

## Executive verdicts

| # | surface | verdict | closed statement and boundary |
|---|---|---|---|
| 1 | semidirect splitting / obstruction | **FEED-171-CGauge** | **FORMULATION: SPLIT-AS-WRITTEN. ARTIFACT: NOT-TYPED.** A typed strict `rtimes` splits tautologically, has neutral factor system, and has `R_twist=0`. The current artifacts do not exhibit a wide normal kernel, strict action functor/object compatibility, or overlap maps, so no actual local/global obstruction class is yet defined. Any section gives set coordinates; neither those coordinates nor a split imply statistically independent streams. |
| 2 | score-fiber orbit count | **WORTH-AN-ARM + FEED-171-CGauge** | **DERIVED:** Burnside gives the exact constructive orbit count. The full score-fiber action has one orbit per score pair. In the frozen fresh-process evaluator configuration, the score-pair image has at most `2^64` values, hence at most 64 ideal fixed-length quotient-label bits per execution axis. Exact ZIP saving remains **UNKNOWN**. |
| 3 | representations / v8 / FiLM | **FEED-398-v8** | **NO-GO** for “five classes are five `H_cov` irreps” and “current FiLM is an intertwiner” (`verdict_scope=FORMULATION x IMPLEMENTATION`). Frozen label idempotents make class-channel decomposition/bookkeeping canonical; they do not derive v8's actual edge carriers or rate allocation. Current FiLM equivariance is untyped/unproved. |
| 4 | modules / cross-tensor structure | **FEED-110 + FEED-242** | **DERIVED conditional gate:** common isotypes and nonzero intertwiner spaces identify algebraically admissible equivariant couplings, not correlation, low rank, or byte savings. The #461 transpose is an exact storage isomorphism, not yet an `H_cov`-module isomorphism; the shared-VQ null does not prove absence of a common submodule. |
| 5 | Galois-style lattice | **NO-GO global anti-isomorphism + FEED-infdesc-171** | **DERIVED:** arbitrary actions give an order-reversing Galois connection. An anti-isomorphism holds only between closure-fixed subgroups/subgroupoids and closure-fixed invariant subpayloads. The stratified global claim additionally owes effective descent. |

The two arm-strength statements are therefore narrower than the prompt's tempting versions: the
strict semidirect **formulation** is split-as-written while the artifact extension is not yet typed,
and the orbit-rate law closes as exact finite algebra plus an evaluator-type ceiling, not as a
measured compressed-byte saving.

## Checkpoint 2 — the two highest-value results

### 2.1 Splitting result: strict `rtimes` would split; the current extension is not yet typed

Let

\[
  1\longrightarrow K_\sigma\longrightarrow E_\sigma
  \overset{q_\sigma}{\longrightarrow}H_{\mathrm{cov},\sigma}
  \longrightarrow1
\]

be the extension on one fixed regular stratum. If the Weyl memo's notation
`K_sigma rtimes H_cov,sigma` is literal, it includes a strict action
`alpha_sigma : H_cov,sigma -> Aut(K_sigma)` and the homomorphic section
`s_sigma(h)=(1,h)`. Therefore

\[
  q_\sigma\circ s_\sigma=\operatorname{id},\qquad
  s_\sigma(h_1)s_\sigma(h_2)=s_\sigma(h_1h_2),
\]

so the extension is **SPLIT BY DEFINITION IN THAT STRICT FORMULATION**. A nontrivial action makes the
product non-direct, not non-split. The present artifacts use the `rtimes` notation and say that
covariance acts by conjugation, but do not exhibit the wide normal kernel, action functor/object
compatibility, or overlap data needed to independently reconstruct the groupoid extension. The
actual chartwise extension is therefore **NOT-TYPED**, not empirically or independently
`DERIVED-SPLIT`.

The Weyl kernel is a join of subgroupoids, so the displayed exact sequence is isotropy-group
notation (or a one-object chart) for the corresponding groupoid extension. A strict semidirect
**groupoid** likewise has the canonical splitting functor `h -> (1,h)`. Classical group `H^2`
notation becomes legitimate only after restricting to an isotropy group or typing an abelian
coefficient group bundle and groupoid cohomology. This distinction does not change the
strict-formulation split verdict, but it leaves the artifact split status not-typed and narrows where
the cocycle formulas live.

For an arbitrary set-theoretic section `s`, define its conjugation lift and factor set

\[
  \alpha_h^s(k):=s(h)ks(h)^{-1},
  \qquad
  \omega_\sigma(h_1,h_2)
  :=s(h_1)s(h_2)s(h_1h_2)^{-1}\in K_\sigma .
\]

Associativity gives the twisted cocycle identity

\[
  \omega(h_1,h_2)\,\omega(h_1h_2,h_3)
  =\alpha^s_{h_1}\!\bigl(\omega(h_2,h_3)\bigr)\,
    \omega(h_1,h_2h_3),
\]

with the usual coboundary change under a changed section. For an abelian coefficient kernel this is
an `H^2(H_cov,sigma,K_sigma)` class; for the nonabelian joined kernel it is a pointed nonabelian
factor-set class, not automatically an abelian cohomology group. In the nonabelian case the section's
conjugation lifts also obey

\[
\alpha^s_{h_1}\alpha^s_{h_2}
=\operatorname{Ad}_{\omega(h_1,h_2)}\alpha^s_{h_1h_2},
\]

so one must retain the full Schreier factor system rather than pretending that `alpha` is already a
strict action.

Every set-theoretic section already gives a unique set coordinate `e=k s(h)`. In those coordinates,

\[
(k_1,h_1)(k_2,h_2)
=\bigl(k_1\alpha^s_{h_1}(k_2)\omega(h_1,h_2),h_1h_2\bigr).
\]

Splitting means that an admissible section change makes `omega'=1` and `alpha'` a strict action, so
this multiplication becomes the untwisted semidirect law. It does **not** mean that coordinates
exist only in the split case.

**FORMULATION verdict:** the factor system of a typed literal semidirect chart is neutral (the
abelian cocycle class is trivial), and its twist-rate contribution is zero. **ARTIFACT verdict:** the
local extension data and transition maps are absent, while topology changes alter the
kernel/isotropy rank. Thus neither a local nontrivial class nor a single global extension by one
fixed `K` is yet typed, much less proved to have a nonzero `H^2` class. Global independent
factorization is **NOT-TYPED / GLUING-AUDIT-OWED**, not “obstructed” by a derived nonzero cocycle.

Any chosen section gives the coordinate pair `e=k s(h)`. For source variables `K,H`, the ideal joint
rate is

\[
H(K,H)=H(H)+H(K\mid H),
\]

not `H(K)+H(H)` unless `I(K;H)=0`. Thus even in the split formulation, independent payload streams
remain a statistical/codec question.

Let `mathcal A` denote the typed action/atlas data and let the section-invariant extension-class
variable be

\[
\Theta:=
\begin{cases}
[\omega]\in H^2(H,K),&K\text{ an abelian }H\text{-module},\\
[(\alpha^s,\omega^s)]_{\rm Schreier},&K\text{ nonabelian}.
\end{cases}
\]

The irreducible ideal twist-rate term is named

\[
\boxed{
R_{\mathrm{twist}}^{\rm ideal}
:=H(\Theta\mid q_H,\mathcal A,\text{public receiver data}).
}
\]

Raw `omega^s` is section-dependent and therefore is not itself the invariant rate object. For an
actual realization, minimize over admissible sections,

\[
R_{\mathrm{twist}}^{\rm code}
:=\min_s L(\alpha^s,\omega^s\mid q_H,\mathcal A,\text{public receiver data}),
\]

and assign any excess over the class information to chart/section/grammar realization. Public or
already-coded derivability makes the conditional information zero; being merely “video-derived” is
not sufficient to charge it. In the **STRICT-SEMIDIRECT FORMULATION**, `Theta` is neutral and the
canonical section has `omega=1`, so the twist term is zero. For the actual local/global artifact,
`mathcal A`, `Theta`, and hence `R_twist` are **NOT-TYPED**.

`verdict_scope: STRATUM x FORMULATION — this does not prove global triviality, and it does not infer
a nonzero obstruction from state dependence alone.`

### 2.2 Orbit-count law: exact combinatorial closure, no numeric archive-byte closure

Write `F_s=S^{-1}(s)` and

\[
  G_S\cong\prod_{s\in\operatorname{im}S}\operatorname{Sym}(F_s).
\]

For `x in F_s`, the full symmetric factor acts transitively, while every other fiber factor fixes
`x`. Hence

\[
  |G_S\!\cdot x|=|F_s|,
  \qquad
  \operatorname{Stab}_{G_S}(x)
  \cong \operatorname{Sym}(F_s\setminus\{x\})
       \times\!\prod_{t\ne s}\operatorname{Sym}(F_t).
\]

Therefore each score fiber is exactly one orbit and

\[
  X/G_S\cong\operatorname{im}S,
  \qquad |X/G_S|=|\operatorname{im}S|.
\]

Double-counting pairs `(g,x)` with `g x=x` gives Burnside's formula without importing an unproved
name:

\[
  \boxed{
  |X/G|=\frac1{|G|}\sum_{g\in G}|\operatorname{Fix}_X(g)| .}
\]

For the full `G_S`, this equals `|im S|`; conditioned on a fixed `s`,
`|F_s/Sym(F_s)|=1`. Thus the **ideal fixed-length** quotient-label cost is
`ceil(log2 |im S|)` bits globally and zero additional identity bits within a known score fiber.
For a random source, the corresponding ideal expected rate is `H(S(X))`, not
`log2 |im S|` unless the score labels are uniform.

For an actually constructible finite subgroup `H_s <= Sym(F_s)`, the operational algebraic target
is

\[
  N_s:=|F_s/H_s|
      =\frac1{|H_s|}\sum_{h\in H_s}|\operatorname{Fix}_{F_s}(h)|,
  \qquad R_s^{\rm fixed}=\lceil\log_2 N_s\rceil .
\]

Relative to an identity index over `F_s`, the ideal combinatorial saving obeys the exact bound

\[
  0\le \log_2|F_s|-\log_2N_s\le\log_2|H_s|,
\]

because every orbit has size at most `|H_s|`. This turns the Weyl arm's unknown into a **symbolic
DERIVED bound**, but not a numerical byte bound: the repository has neither enumerated `F_s` nor a
finite receiver-closed `H_s` with fixed-point counts, and Brotli/ZIP length is not log-cardinality.
The full `G_S` is nonconstructive and cannot itself authorize a codec.

`verdict_scope: REPRESENTATION x RECEIVER — the combinatorial quotient is exact; compressed archive
saving remains UNKNOWN until the foundations enumeration supplies a finite admissible set and the
receiver supplies legal orbit representatives.`

### 2.3 A numeric ideal-label ceiling from the actual evaluator type

The Weyl memo fixes

\[
X=\bigl(\{0,\ldots,255\}^{874\times1164\times3}\bigr)^{2P},
\qquad P=600,
\]

so an unconstrained exact witness identity contains

\[
8\cdot 600\cdot2\cdot874\cdot1164\cdot3
=29{,}299{,}276{,}800
\]

raw bits. Read-only inspection of the frozen `upstream/evaluate.py` shows that both the Pose and Seg
accumulators are scalar `torch.zeros` tensors with the default float32 dtype, and that the returned
pair is obtained from those tensors by division followed by `.item()`. No upstream default-dtype
override exists, so the **fresh-process frozen evaluator configuration** retains PyTorch's float32
default. Under the repository's finite-result admissibility contract (a NaN/Inf scorer row is
invalid), therefore, on any one declared execution axis,

\[
|\operatorname{im}S|\le 2^{32}\,2^{32}=2^{64},
\qquad
\boxed{\left\lceil\log_2|X/G_S|\right\rceil\le64\ \text{bits}.}
\]

This is a **DERIVED apparatus-type ceiling**, not a measurement and not an eight-byte codec. It says
that an omniscient ideal score-class label needs at most 64 fixed-length bits, whereas identifying an
arbitrary member of `X` needs 29,299,276,800 raw bits. The corresponding ideal identity-removal
lower bound is 29,299,276,736 raw bits. None of these numbers include a receiver-computable section,
program/container overhead, legality, or generalization. The conservative statistic `T=(A,P)` does
not inherit the 64-bit bound. CPU and CUDA remain distinct images even though the type ceiling holds
on each.

This closes the question “can algebra derive a bound?” as follows:

- **YES:** exact orbit-count formula, exact symbolic saving bound, and a 64-bit ideal score-label
  ceiling for this evaluator snapshot;
- **NO:** no lower bound on realizable archive bytes and no numerical Brotli/ZIP saving, because a
  legal section and finite constructive orbit atlas are absent.

`verdict_scope: FRESH-PROCESS-EVALUATOR-CONFIG x AXIS x IDEAL-CODE — any default-dtype, import,
scorer, or accumulator change requires re-derivation; printed decimal rounding is not used in this
bound, and nonfinite rows are excluded as inadmissible.`

## 3. Representation theory: what is and is not canonical in v8

### 3.1 Conditional isotypic law

Let `V` be a linear frame, logit, feature, or weight space carrying a representation `rho` of a
finite group `H`, over characteristic zero. Maschke complete reducibility and Schur's lemma give

\[
\boxed{
V\cong\bigoplus_{\lambda}M_\lambda\otimes U_\lambda,
\qquad
\rho(h)=\bigoplus_\lambda I_{M_\lambda}\otimes\rho_\lambda(h),
}
\tag{9}
\]

where `U_lambda` are inequivalent irreducibles and `M_lambda` are multiplicity spaces. The same form
holds for compact groups with continuous finite-dimensional unitary representations. It does **not**
hold automatically for the full linearized `SE(3)`-like covariance chart: that group is noncompact,
translations can act non-semisimply, and the current `H_cov` is partial/state-dependent. The exact
induced image on the finite uint8 alphabet would be finite if it were globally defined, but the Weyl
artifact only declares regular-chart actions. Thus (9) is a per-finite-image/per-compact-isotropy
tool, not a global theorem about the present stratified pseudogroup.

### 3.2 Five class carriers are multiplicities, not five covariance irreps

On one typed pre-argmax linear chart, let `H_cov,sigma` be a finite/compact induced
chart/isotropy group. Assume it acts spatially, does not relabel SegNet semantics, and commutes with
the class-idempotent action. With five class fields,

\[
V_{\rm class}=\mathbb k^5\otimes W,
\qquad
\rho_{\rm class}(h)=I_5\otimes\rho_W(h),
\qquad h\in H_{\rm cov,\sigma}.
\]

If \(W\cong\bigoplus_\lambda M_\lambda\otimes U_\lambda\), then

\[
V_{\rm class}\cong
\bigoplus_\lambda(\mathbb k^5\otimes M_\lambda)\otimes U_\lambda.
\tag{10}
\]

**DERIVED:** covariance sees the class axis as five equivalent multiplicity copies. Its central
isotypic projectors select `lambda`; they cannot select Road versus Lane versus Undrivable versus
Movable versus MyCar. Any invertible change of basis on `k^5` commutes with `H_cov,sigma`, so
covariance alone does not canonically privilege the current class basis.

The class-channel direct-sum decomposition and its bookkeeping become canonical relative to the
frozen scorer semantics by adjoining the commutative class-idempotent algebra

\[
A_{\rm cls}\cong\mathbb k^5,
\qquad e_ce_d=\delta_{cd}e_c,
\qquad \sum_ce_c=1.
\]

For a finite induced image, modules over
\(A_{\rm cls}\otimes\mathbb k[H_{\rm cov,\sigma}]\) decompose as below. A compact unitary chart has
the same joint commuting-action decomposition. In either case,

\[
\boxed{
V\cong
\bigoplus_{c,\lambda}
(e_c\mathbb k^5\otimes M_\lambda)\otimes U_\lambda.
}
\tag{11}
\]

This is the correct, narrow canonicity statement: the **class-channel decomposition/bookkeeping** is
canonical relative to frozen label idempotents plus the covariance action, not because classes are
covariance irreps. Bit/rate allocation is not selected by (11); it still needs measured
rate-distortion costates/KKT conditions and exact bytes.

The v8 edge-centric carrier would require a separately typed direct-sum edge payload, fixed edge
index algebra, compatible `H_cov` action, and incidence/merge-diff-correct reconciliation maps. The
current spec supplies the design obligation, not a representation-theoretic derivation of that edge
algebra. Edge canonicity is therefore **DESIGN/ASSUMED UNTIL TYPED**. Tropical argmax further makes
the global object stratified and nonlinear; (11) governs only each declared pre-argmax class/logit
chart.

### 3.3 Current FiLM morphism status is untyped and unproved

For a code/conditioning state `z`, write a FiLM family as

\[
F_z(v)=D_{1+\gamma(z)}v+\beta(z).
\]

It is an equivariant affine family only if, for declared actions on `z`, input, and output,

\[
\boxed{
D_{1+\gamma(hz)}\rho_V(h)=\rho_W(h)D_{1+\gamma(z)},
\qquad
\beta(hz)=\rho_W(h)\beta(z),
}
\tag{12}
\]

and every intervening linear layer/nonlinearity obeys its corresponding equivariance condition. A
homogeneous coordinate linearizes each **fixed-`z`** affine map in `v`; it does not automatically
linearize the whole conditioning family. A family-level representation morphism additionally needs
a typed action on the conditioning-feature lift, and nonlinear SiLU/`sin`/`tanh` paths generally need
an enlarged feature representation whose action is also proved.

**RE-DERIVED from code:** V9 uses dense code-to-layer modulation. Pose-FiLM v1 uses `sin`/`tanh` in
its conditioning path; pose-FiLM v2 instead uses SiLU conditioning, linear gamma/beta heads, and a
1x1 residual projection. No variant declares the relevant `H_cov` actions, constrains its weights to
the required intertwiner/commutant spaces, or tests (12). Consequently equivariance is
**UNTYPED/UNPROVED AND NOT GUARANTEED BY CONSTRUCTION**. An approximately covariant learned family
remains possible but unmeasured.

**Verdict — FEED-398-v8.** Preserve v8's scorer-semantic edge-centric design, type its edge payload
and incidence action, add a joint `A_cls x H_cov` equivariance audit, and call FiLM a morphism only
after (12) plus receiver parity passes. **NO-GO** for the two stronger current claims.

`verdict_scope: FORMULATION x IMPLEMENTATION — this does not kill per-class carriers, FiLM, or an
equivariantly constrained successor.`

## 4. Modules and tensors: explain #461 without laundering it into a theorem

### 4.1 What the settled bytes prove

The #461 receipt is recalled, not re-run:

- **MEASURED:** exact archive `63,659 B -> 63,242 B`, delta `-417 B`, with the full decoded quantized
  state hash unchanged;
- **MEASURED:** selected 2-D weight-axis transposes reduce the base stream by 180 B and the isolated
  archive by 149 B;
- **MEASURED:** frame-separated modulo-256 pair deltas reduce the code stream by 268 B and the
  isolated archive by 251 B;
- **MEASURED/DERIVED:** tensor identity carries 1.1262153531727064 bits/weight of distribution
  information, while current pooled storage wins only 26 B over separate streams;
- **INSTANCE x FORMULATION null:** an additional exact post-hoc shared scalar value codebook on the
  already-quantized checkpoint has no distinct repeated payload to remove.

### 4.2 Storage isomorphism versus module isomorphism

Let `(V_i,rho_i)` and `(V_j,rho_j)` be payload tensor modules. A linear bijection `P:V_i->V_j` is an
`H`-module isomorphism exactly when

\[
\boxed{P\rho_i(h)=\rho_j(h)P\quad\text{for every }h\in H.}
\tag{13}
\]

The #461 transpose is an exact coordinate permutation and its inverse is applied before evaluation.
That proves a **storage-space vector isomorphism** and bit identity. No covariance representation was
declared or checked, so calling it an `H_cov`-module isomorphism would be fake. The byte win arises
because Brotli is order-sensitive. The modulo-256 delta transform is likewise an exact group
automorphism of the discrete table coordinates, but its paying signal is temporal dependence, not a
proved covariance submodule.

The shared-VQ null also does **not** imply “no common submodule.” A scalar value alphabet is not a
linear invariant subspace. It tests one post-quantization coding formulation. Common representation
types could exist with tensor-specific value histograms, and a shared module could require a learned
basis rather than a common scalar codebook.

### 4.3 A predictive, training-time module test

In the semisimple regime over `k=R` or `C`, write

\[
V_i=\bigoplus_\lambda M_{i\lambda}\otimes U_\lambda,
\qquad
V_j=\bigoplus_\lambda M_{j\lambda}\otimes U_\lambda.
\]

Then

\[
\operatorname{Hom}_H(V_i,V_j)
\cong
\bigoplus_\lambda
\operatorname{Hom}(M_{i\lambda},M_{j\lambda})
\otimes\operatorname{End}_H(U_\lambda).
\tag{14}
\]

Over an algebraically closed characteristic-zero field with irreducible `U_lambda`, Schur reduces
`End_H(U_lambda)` to scalars. A nonzero intertwiner has invariant kernel and image; under complete
reducibility it therefore identifies an algebraically admissible equivariant coupling/common
representation type. It does **not** prove that the realized tensors are correlated, low-rank, tied,
or cheaper to code; even a ubiquitous trivial isotype can make `Hom_H` nonzero.

For a finite receiver-closed action, project any candidate cross-tensor map `A` onto the intertwiner
space using the Reynolds average

\[
\boxed{
\Pi_H(A)=\frac1{|H|}\sum_{h\in H}
\rho_j(h)A\rho_i(h)^{-1}.}
\tag{15}
\]

For a compact group, replace the sum by normalized Haar integration. If only generators are known,
solve or penalize `A rho_i(g)-rho_j(g) A=0` on those generators. Before any SVD/rank decision, choose
`H`-invariant positive-definite inner products (by averaging Gram matrices for finite/compact `H`),
unitarize the representations, and operate on the induced multiplicity-block maps. Ordinary
coordinate singular vectors of `Pi_H(A)` need not be invariant for nonorthogonal representations;
degenerate singular subspaces also have no canonical basis.

This yields a concrete candidate gate:

- after invariant-metric unitarization, test low rank only in the isotypic multiplicity maps of
  `Pi_H(A)`, and require realized correlation plus receiver-measured rate value;
- keep nonmatching isotypes tensor-local;
- allow any exact bijective **serialization** chart, including #461's transpose, when its inverse is
  applied before semantics. If a map remains in the semantic/model forward, require commutant
  membership for arrow-by-arrow equivariance; a normalizer is sufficient only when the induced
  action automorphism is separately typed and allowed;
- compare the resulting exact archive against the unconstrained chart, because module structure
  does not guarantee compression.

A suitable training-time structural term is

\[
\Omega_{\rm module}
=\sum_{(i,j),g}
\lVert A_{ij}\rho_i(g)-\rho_j(g)A_{ij}\rVert_{G_i\to G_j}^2
+\lambda_R\widehat R(A_{ij}),
\tag{16}
\]

where `G_i,G_j` are `H`-invariant Gram metrics and

\[
\lVert B\rVert_{G_i\to G_j}^2
:=\lVert G_j^{1/2}BG_i^{-1/2}\rVert_F^2
=\operatorname{tr}(G_i^{-1}B^*G_jB).
\]

`lambda_R` is admitted only at stage boundaries under the existing contract. Equation (16) is a
design FEED, not evidence that the current covariance matrices, data correlation, low-rank block,
or paying intertwiner exists.

### 4.4 Relation to the low-rank pose codec

The #140 SVD surface factors a `600 x 6` pose table in the ordinary tensor product
`V_time tensor V_pose`. Low matrix rank is not automatically low **module** rank. It becomes
covariance-predictive only if the action factorizes across the two axes and the retained left/right
spaces are invariant (or are first projected into the appropriate isotypes). Conversely, the #461
lossless integer residual coder exploits temporal arithmetic structure and should not be conflated
with #140's lossy SVD geometry.

**Verdict — FEED-110 + FEED-242.** Build no generic “shared VQ again” arm from #461. The arm-worthy
successor is a no-launch first gate that types `rho_i`, computes/solves (14)-(15), installs invariant
metrics, and then checks realized multiplicity-block correlation/rank. A nonzero shared isotype only
admits a candidate; it does not authorize tying or low rank. Reactivate a codec only with exact
byte-close and scorer gates.

`verdict_scope: FORMULATION x INSTANCE for the VQ null; DESIGN x CONDITIONAL-SEMISIMPLE for the
module predictor. Neither is a family kill.`

## 5. Galois-style correspondence: a connection, not a free anti-isomorphism

Let a group `G` act by automorphisms on a payload observable algebra `A`. For a subgroup `H<=G` and
a subalgebra/subpayload `B<=A`, define

\[
A^H=\{a:h\cdot a=a\ \forall h\in H\},
\qquad
\operatorname{Stab}(B)=\{g:g\cdot b=b\ \forall b\in B\}.
\]

Then

\[
\boxed{
B\subseteq A^H
\quad\Longleftrightarrow\quad
H\subseteq\operatorname{Stab}(B).
}
\tag{17}
\]

Both maps reverse inclusion. They induce closure operators

\[
\operatorname{cl}_G(H)=\operatorname{Stab}(A^H),
\qquad
\operatorname{cl}_A(B)=A^{\operatorname{Stab}(B)}.
\tag{18}
\]

**DERIVED:** (17) is a Galois connection. It restricts to a lattice anti-isomorphism only between
the closure-fixed objects. For arbitrary actions, two different subgroups can have the same
invariants and an arbitrary subpayload need not be the full fixed algebra of any subgroup. Garrett's
finite Galois correspondence is stronger because the field-extension hypotheses force the relevant
closures; those hypotheses do not transfer merely because both sides form lattices.

For a groupoid acting on a bundle/sheaf of payload algebras, the same construction works with wide
subgroupoids and invariant sections, **provided the action and fixed-section functor are typed**.
Per regular stratum this gives a closure-lattice walk. A global stratified anti-isomorphism needs:

1. compatible coefficient/payload bundles on overlaps;
2. restriction maps preserving the action;
3. effective descent so locally invariant sections glue uniquely enough for the receiver; and
4. closure-fixed subgroupoids/subpayloads.

The current atlas changes isotropy rank at topology events and provides no such global descent
certificate. Therefore the full “subgroupoid lattice anti-isomorphic to payload reductions” claim is
**NO-GO at STRATIFIED-GROUPOID GENERALITY**. The per-stratum Galois connection and its closure-fixed
lattice remain a valid **FEED-infdesc-171**. A solver can walk that smaller lattice only after each
node records its receiver section and actual rate delta; inclusion alone does not rank byte value.

`verdict_scope: FORMULATION — this rejects the unqualified global anti-isomorphism, not invariant
subpayload search or a future descent-complete groupoid representation.`

## 6. Composition with the landed foundations memo

The companion foundations memo has now landed. Its exact objects are:

\[
D/E_{U,D}\cong U(D),
\qquad
E_G\subseteq E_{U,D},
\qquad
H(q_G(W))-H(U(W))=H(q_G(W)\mid U(W))\ge0.
\tag{19}
\]

The composition is **not** a Cartesian product called “setoid quotient times algebra.” It is a typed
pipeline:

```text
legal support D and PER E_(U,D)
        |
        v
semantic quotient D/E_(U,D) ~= U(D)              [foundations]
        ^
        | coarsening; fiber-completeness owed
constructive orbit quotient D/E_H                 [foundations + action]
        |
        +--> Burnside/fixed-point count            [this memo]
        +--> chart extension and factor set        [this memo]
        +--> module/isotypic decomposition         [this memo]
        +--> closure-fixed invariant lattice       [this memo]
        |
        v
legal deterministic receiver section + exact ZIP [apparatus; still owed]
```

Foundations owns the legal support, PER/setoid quotient, generated orbit relation, enumeration,
fiber-completeness, and descent certificates. Algebra consumes a **concrete** action inside that
setoid:

- Burnside counts how many constructive orbits remain inside semantic fibers;
- the extension audit decides whether some section makes local arrow multiplication untwisted
  semidirect; set coordinates exist for every section;
- representation/module theory identifies algebraically admissible equivariant-coupling candidates
  within linear charts;
- the Galois connection organizes only closure-fixed invariant subpayloads.

Let `mathcal A` denote any nonpublic action/atlas choice, `Theta` its section-invariant extension
class, and `Gamma` all remaining chart/gluing, receiver-section, cocycle-table realization, and
grammar choices. Because `U` is a function of the constructive orbit label `q_H`, the exact chain
rule is

\[
\boxed{
H(q_H,\mathcal A,\Theta,\Gamma)
=H(U)
+H(q_H\mid U)
+H(\mathcal A\mid q_H)
+H(\Theta\mid q_H,\mathcal A)
+H(\Gamma\mid q_H,\mathcal A,\Theta).
}
\tag{20}
\]

Conditioned public receiver data is implicit in every term. This identifies the invariant ideal
`R_twist=H(Theta | q_H, mathcal A, public)` and the semantic refinement debt `H(q_H|U)` without
assuming independence. In the strict-semidir formulation `Theta` is neutral, so `R_twist=0`; this
does not kill conditional action/atlas debt or realization/gluing debt. Public generic data has zero
conditional information, while any realization bytes remain exact grammar cost. The **expected**
length of a uniquely decodable ensemble code is lower-bounded by the joint entropy; an individual
Brotli/ZIP length is not pointwise bounded by entropy and must be measured.

This preserves the sibling boundary: setoid/PER semantics and enumeration remain foundations work;
the factor-set, action count, modules, and closure lattice are the complementary algebra layer.

## 7. Canonical law, DAG FEED, and triality

### Canonical equation FEED

The isolated proposed equation is
`stratified_extension_orbit_rate_law_v1` in
`.omx/research/garrett_extension_orbit_rate_equation_feed_20260713.md`. Its two exact clauses are:

\[
\operatorname{ob}(E_\sigma)=\text{neutral}
\Longleftrightarrow E_\sigma\text{ splits over the declared action},
\qquad
|D/H|=|H|^{-1}\sum_h|\operatorname{Fix}_D(h)|.
\]

Here `ob(E_sigma)` is `[omega_sigma]` in ordinary `H^2` only for an abelian kernel with fixed action;
for a nonabelian kernel it is the class of the full Schreier factor system `(alpha,omega)`.

The evaluator-type ceiling and the conditional operational rate terms are explicitly scoped there.

### DAG FEED

The isolated DAG is
`.omx/research/garrett_algebra_dig_DAG_FEED_20260713.md`. It makes setoid support and action custody
parents of every algebra node and makes a legal receiver section plus exact archive A/B parents of
any pointer review.

### Proposed typed DSL record; no implementation edit

`algebra_orbit_accounting` should eventually carry:

```text
support_id / observable_id / hardware_axis
stratum_id / kernel_bundle_id / covariance_action_hash
section_kind / section_homomorphic / overlap_transition_hash
extension_class_kind / extension_class_status / extension_class_hash
factor_system_realization_hash / derivability_status / class_conditional_info_bits
finite_action_order / fixed_point_counts_hash / orbit_count / quotient_bits
receiver_section_hash / parseback_status
twist_class_bits / factor_system_realization_bytes / archive_bytes_before / archive_bytes_after
evidence_label / verdict_scope
```

No trainer flag, argv, or shared DSL file was invented or edited.

### Six-hook wire-in

1. **Sensitivity map — FEED:** distinguish exact orbit, tangent isotype, commutant direction, and
   merely correlated storage order.
2. **Pareto constraint — FEED:** exact statistic closure, legal receiver section, and exact bytes are
   separate mandatory coordinates.
3. **Bit allocator — FEED:** admit zero twist bits only for a typed neutral/public-derivable extension
   class with zero conditional information; separately account for action/atlas and realization
   bytes. Use isotypic multiplicities only as candidates.
4. **Cathedral/autopilot — blocked by design:** no dispatch until finite action, section, parse-back,
   and storage preflight exist. This unit grants no launch authority.
5. **Continual-learning posterior — FEED:** future extension-class/section realization, fixed-point,
   intertwiner-rank, and archive A/B rows become typed anchors rather than prose conclusions.
6. **Probe disambiguators — required:** split-vs-statistical-independence; covariance-irrep-vs-class
   idempotent; module-isomorphism-vs-storage permutation; closure connection-vs-global lattice.

`FORMALIZATION_PENDING`: shared equation/DAG/DSL registries were not edited because the mission is
analysis-only and live siblings own those surfaces.

## 8. Self-adversarial audit and pointer honesty

- A literal, typed `rtimes` was not relabeled obstructed: it is split by definition. The artifact's
  action/normality/overlap typing was not silently upgraded from notation to implementation proof.
- State dependence was not converted into a fake nonzero `H^2` class. A kernel coefficient bundle,
  overlap action, and sections are prerequisites.
- For a nonabelian kernel, the factor set is not silently promoted to an abelian cohomology group.
- Algebraic split coordinates were not equated with independent entropy streams:
  `H(K,H)=H(H)+H(K|H)`, and equality with `H(K)+H(H)` requires independence.
- A raw section cocycle was not treated as invariant payload. Only positive conditional information
  in the extension class, given the typed action/atlas and public/already-coded data, contributes
  ideal `R_twist`; section-table realization is separate.
- Burnside was applied only to finite actions. `log orbit-count` was not equated with Brotli/ZIP
  length, and the nonconstructive full symmetric group was not treated as a legal codec.
- The 64-bit ceiling is tied to the frozen fresh-process evaluator configuration and accumulator
  types, per axis; it is not an eight-byte archive claim and does not apply to `T=(A,P)`.
- v8 class channels were not mislabeled irreps. Frozen scorer idempotents select canonical
  class-channel bookkeeping; edge carriers remain to be typed and allocation remains measured.
- Current FiLM was not called equivariant without typed actions or an intertwiner test.
- The #461 storage transpose and shared-VQ null were kept at their exact instance/formulation scope.
- The module predictor is conditional on finite/compact semisimplicity; no global noncompact
  `SE(3)` complete-reducibility claim was made.
- A nonzero intertwiner was not promoted to data correlation, low rank, or byte saving; invariant
  metrics and multiplicity-block evidence remain gates.
- The Galois anti-isomorphism was restricted to closure-fixed objects and global descent was left
  open.
- No launch, eval, archive mutation, live-sibling edit, shared-registry edit, or commit occurred.

**Pointer delta: UNMOVED.** Grounding is **MEANS**. Only a byte-closed, receiver-consumed exact row
with complete custody on its declared contest axis can move the pointer.

## STORES CONSULTED / provenance

External primary mathematical sources:

- Paul Garrett, *Abstract Algebra* (complete text), especially group actions/orbits,
  automorphisms/semidirect products, the finite Galois correspondence, and tensor products:
  `https://www-users.cse.umn.edu/~garrett/m/algebra/Whole.pdf`.
- Paul Garrett, *Generalities on representations of finite groups*, for complete reducibility,
  intertwiners, Schur's lemma, tensor products, and isotypic components:
  `https://www-users.cse.umn.edu/~garrett/m/repns/notes_2014-15/03_generalities_finite.pdf`.

Local contract and mathematical state:

- `CLAUDE.md`, `AGENTS.md`, `docs/operating_manual_craft_handoff.md`;
- `.omx/research/weyl_symmetry_group_unification_20260713.md` plus its equation/DAG FEEDs;
- `.omx/research/infdesc_foundations_dig_20260713.md` plus its equation/DAG FEEDs;
- `.omx/research/SPEC_v8_perclass_decomposition_20260708.md` and v8.1/#398 implementation surfaces;
- `.omx/research/witness_crosstensor_structure_rate_20260713.md` and its canonical equation/codec;
- #140 low-rank pose DSL/apply-pass surfaces;
- current V9/v2 FiLM receiver code and CGauge covariance/equation artifacts;
- frozen `upstream/evaluate.py` and `upstream/modules.py` accumulator/statistic code;
- latest required sister memos, directives, canonical task/lane/subagent stores, and graph recall.

No transient rank, score, or pointer claim is introduced. Every negative is scoped; every family
reactivation criterion remains explicit.
