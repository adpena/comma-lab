# Canonical equation FEED — stratified extension-orbit rate law v1

**Date:** 2026-07-13

**Proposed equation id:** `stratified_extension_orbit_rate_law_v1`

**Status:** DESIGN / equation feed; not registered

**Authority:** DERIVED mathematics plus read-only evaluator-type inspection

**Pointer:** UNMOVED

## One-line law

Every set-theoretic section of a typed covariance extension supplies kernel/covariance coordinates.
The extension has **untwisted semidirect multiplication** exactly when some section is homomorphic,
equivalently when the full extension/factor-system class is neutral. A finite constructive action
then has an exact Burnside orbit count, while positive conditional transition information and failure
to span the full semantic fibers remain rate debts.

## Canonical form

Let

\[
1\longrightarrow K_\sigma\longrightarrow E_\sigma
\overset q\longrightarrow H_\sigma\longrightarrow1
\]

be a typed extension with section `s`, and let `H<=Sym(D)` be a finite receiver-valid action that
preserves an observable `U:D->Y`. For an arbitrary section, define

\[
\alpha_h^s(k):=s(h)ks(h)^{-1},
\qquad
\omega_\sigma(h_1,h_2)
=s(h_1)s(h_2)s(h_1h_2)^{-1}.
\]

For the Weyl subgroupoid kernel, read this as an isotropy-group chart of a typed groupoid extension;
the strict semidirect groupoid itself has the same canonical splitting functor. A global class needs
an abelian coefficient group bundle (or the corresponding nonabelian factor-system data).

Then the proposed canonical law is

\[
\boxed{
\begin{gathered}
\operatorname{ob}(E_\sigma)=\mathrm{neutral}
\quad\Longleftrightarrow\quad
\exists\ \text{homomorphic section }s
\quad\Longleftrightarrow\quad
E_\sigma\cong K_\sigma\rtimes H_\sigma,\\[2mm]
N_H:=|D/H|
=\frac1{|H|}\sum_{h\in H}|\operatorname{Fix}_D(h)|,
\qquad
R_H^{\rm fixed}=\lceil\log_2N_H\rceil,\\[2mm]
H(q_H(W))
=H(U(W))+H(q_H(W)\mid U(W)),\\[2mm]
H(q_H,\mathcal A,\Theta,\Gamma)
=H(U)+H(q_H\mid U)
+H(\mathcal A\mid q_H)
+H(\Theta\mid q_H,\mathcal A)
+H(\Gamma\mid q_H,\mathcal A,\Theta).
\end{gathered}
}
\tag{Garrett-extension-orbit-v1}
\]

The last equivalence means an **isomorphism of extensions**, for the strict action induced by the
homomorphic section: it commutes with the inclusion of `K_sigma` and projection to `H_sigma`. An
arbitrary section still gives a set bijection `E_sigma ~= K_sigma x H_sigma`, but with multiplication

\[
(k_1,h_1)(k_2,h_2)
=\bigl(k_1\alpha^s_{h_1}(k_2)\omega(h_1,h_2),h_1h_2\bigr).
\]

The obstruction symbol is typed by

\[
\operatorname{ob}(E_\sigma)=
\begin{cases}
[\omega_\sigma]\in H^2(H_\sigma,K_\sigma),
&K_\sigma\text{ abelian with fixed action},\\
[(\alpha^s_\sigma,\omega_\sigma)]_{\rm Schreier},
&K_\sigma\text{ nonabelian}.
\end{cases}
\]

Set `Theta:=ob(E_sigma)`, let `mathcal A` be the typed action/atlas choice, and reserve `Gamma` for
chart/gluing, section, cocycle-table realization, and grammar data.

Here

\[
R_{\rm fiber\ debt}=H(q_H(W)\mid U(W))\ge0,
\qquad
R_{\rm twist}^{\rm ideal}
=H(\Theta\mid q_H,\mathcal A,\text{public receiver data}).
\]

For a literal semidirect chart, use the canonical section `(1,h)`, so `omega=1` and
`Theta` is neutral, and `R_twist=0`. Raw `omega^s` is section-dependent; a realized code term must
either live in `Gamma` or be minimized over admissible sections:

\[
R_{\rm twist}^{\rm code}
:=\min_s L(\alpha^s,\omega^s\mid q_H,\mathcal A,\text{public receiver data}).
\]

Only class/transition information not determined by already-coded variables and public receiver
data has positive conditional rate. Any code/container bytes needed to realize public logic still
belong to exact grammar accounting. The chain rule is exact; the **expected** length of a uniquely
decodable ensemble code is bounded below by it. Individual Brotli/ZIP length is not pointwise
entropy-bounded and must be measured.

## Nonabelian precision

For abelian `K_sigma` with a fixed `H_sigma`-module action, `[omega_sigma]` is a class in ordinary
`H^2(H_sigma,K_sigma)`. For a nonabelian kernel, a section induces a Schreier factor system:

\[
\alpha^s_{h_1}\alpha^s_{h_2}
=\operatorname{Ad}_{\omega(h_1,h_2)}\alpha^s_{h_1h_2},
\]

together with the twisted associativity identity. The splitting obstruction is then a pointed
nonabelian factor-set class, not automatically an abelian cohomology-group element. A global
stratified class additionally requires a coefficient group bundle and compatible overlap action.

## Score-fiber specialization

For `U=S=(d_seg,d_pose)` and the full score-fiber group

\[
G_S\cong\prod_s\operatorname{Sym}(S^{-1}(s)),
\]

each fiber is one orbit and `D/G_S ~= im S`. In the frozen fresh-process evaluator configuration,
`d_seg` and `d_pose` are returned from two float32 accumulators and no default-dtype override occurs.
On each declared execution axis, under the finite-result admissibility contract,

\[
\boxed{|\operatorname{im}S|\le2^{64},
\qquad R_{G_S}^{\rm fixed}\le64\ \text{bits}.}
\tag{Evaluator-type ceiling}
\]

For a constructible subgroup `H_s<=Sym(S^{-1}(s))`,

\[
N_s=|S^{-1}(s)/H_s|
=|H_s|^{-1}\sum_{h\in H_s}|\operatorname{Fix}_{S^{-1}(s)}(h)|,
\]

and

\[
0\le\log_2|S^{-1}(s)|-\log_2N_s\le\log_2|H_s|.
\]

## Domain of validity

Included:

- a fixed legal support and frozen observable;
- a typed regular-stratum extension and declared action;
- finite group actions for the Burnside sum;
- abelian `H_sigma`-module coefficients where ordinary group `H^2` notation is used;
- one explicit fresh-process evaluator configuration and execution axis for the 64-bit type ceiling;
- finite (non-NaN/non-Inf) scorer outputs for the score-fiber equivalence and type ceiling;
- public/already-coded derivability and conditional-information custody for every transition datum.

Excluded:

- inferring a global cohomology class from state dependence alone;
- inferring statistical independence from an algebraically split extension;
- treating a public/derivable extension class as positive twist information, or raw section cocycles
  as invariant class data;
- converting `log orbit-count` into Brotli/ZIP bytes;
- treating the full nonconstructive score-fiber symmetric group as a legal receiver;
- transferring the 64-bit ceiling to the conservative statistic `T=(A,P)`;
- any pointer movement without a receiver-closed exact archive row.

## Evidence status for this repository

| clause | status | boundary |
|---|---|---|
| literal regular-chart semidirect formulation | **SPLIT BY DEFINITION** | actual Weyl wide normal kernel/action functor/object compatibility not exhibited; artifact extension NOT-TYPED |
| global obstruction class | **UNKNOWN / NOT YET TYPED** | no fixed coefficient bundle or transition system |
| `R_twist` | **0 in strict-semidir formulation** | actual local/global term NOT-TYPED |
| full score-fiber orbit count | **DERIVED one orbit per score pair** | exact but nonconstructive |
| evaluator quotient-label ceiling | **DERIVED <=64 bits per fresh-process config/axis** | ideal class label, not codec bytes |
| constructible subgroup fixed-point table | **OWED** | no finite receiver-closed action enumerated |
| archive saving | **UNKNOWN** | section, grammar, parse-back, exact ZIP A/B owed |

## Falsification and reactivation

- Reconstructing a typed extension whose factor system cannot be neutralized would prove that the
  Weyl `rtimes` notation is inapplicable to that actual chart; it would not refute the tautology that
  a typed semidirect product splits. This is chart/action scoped.
- Refute a claimed orbit count by reproducing the finite action table and finding a fixed-point or
  closure mismatch.
- Recompute the 64-bit bound after any evaluator dtype/statistic change.
- A receiver or byte failure rejects that constructive section/formulation, not the abstract
  score-fiber theorem.

## Proposed consumers

- capstone #171 / CGauge overlap and section ledger;
- v8 #398 class-idempotent plus covariance audit;
- #110/#242 module-structured training gate;
- bit allocator zero-bit admission;
- foundations `constructive_quotient_debt` accounting;
- receiver-section and parse-back preflight.

## Triality note

- **Equation:** this file.
- **DAG:** `.omx/research/garrett_algebra_dig_DAG_FEED_20260713.md`.
- **DSL:** proposed `algebra_orbit_accounting` record in the main Garrett memo.

The central equation registry remains untouched. This is an isolated design FEED pending main
review and formalization.
