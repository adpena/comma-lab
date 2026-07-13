# Canonical equation FEED - kernel-setoid / PER quotient rate law

- Date: 2026-07-13
- Proposed equation id: `evaluator_kernel_setoid_rate_law_v1`
- Status: DESIGN / equation FEED; uncommitted; not registered
- `research_only=true`
- Authority: `DERIVED-EXACT` finite-set mathematics; practical byte gain `UNKNOWN`
- Pointer delta: `UNMOVED`

## Purpose

This is a type refinement of `weyl_evaluator_fiber_rate_law_v1`. The state-dependent
constructive symmetry may remain a stratified groupoid, but its **orbit space** is the quotient
of the equivalence relation induced by existence of an arrow. On an admissible support this is
most economically stated as a setoid; on an ambient space where some objects are inadmissible it
is a partial equivalence relation (PER). No global group action or group closure is required.

## Typed data

Let:

- `X` be the ambient finite uint8 witness set;
- `D subseteq X` be the nonempty admissible support;
- `U : D -> Y` be the declared frozen observable (`T=(A,P)` for the conservative statistic or
  `S=(d_seg,d_pose)` for the coarser score pair);
- `W` be a `D`-valued random witness;
- `Dec : Z partial-> D` be the legal deterministic receiver; and
- `ell(C(z))` be the exact counted length after the real container/packer.

Define on the ambient set `X`

\[
x\mathrel{\approx_{U,D}}y
\quad\Longleftrightarrow\quad
x,y\in D\ \wedge\ U(x)=U(y).
\tag{1}
\]

`approx_(U,D)` is symmetric and transitive on `X`, and

\[
\operatorname{supp}(\approx_{U,D})
:=\{x\in X:x\approx_{U,D}x\}=D.
\tag{2}
\]

It is therefore a PER on `X`. Its restriction
`E_(U,D) := approx_(U,D) intersect (D x D)` is an equivalence relation, so `(D,E_(U,D))`
is the relevant setoid.

## Canonical law

Let `q_U : D -> Q_U := D/E_(U,D)` be the quotient map. Then

\[
\boxed{
\begin{gathered}
\bar U:Q_U\longrightarrow U(D),\qquad \bar U([x])=U(x),\\
\bar U\text{ is a bijection},\\
R^{\rm sem}_{U,D}(0)
:=H(q_U(W))=H(U(W)),\\
L^*_{U,D,\mathrm{Dec}}(x)
:=\min_{z:\,\mathrm{Dec}(z)\downarrow}
\bigl\{\ell(C(z)):\mathrm{Dec}(z)\in D,\ q_U(\mathrm{Dec}(z))=q_U(x)\bigr\}.
\end{gathered}}
\tag{Kernel-setoid-rate-v1}
\]

The minimum is `+infinity` when the receiver reaches no representative of the class. The equality
`q_U(Dec(z))=q_U(x)` may equivalently be written `U(Dec(z))=U(x)`.

The first equality is the ideal semantic quotient entropy, not an archive-byte prediction. The
last line is the contest-operational law: it pays for the shortest receiver-reachable representative
after the actual packer.

## Relation to a state-dependent groupoid

For a groupoid `G => D`, define its orbit relation

\[
x\mathrel{E_G}y
\quad\Longleftrightarrow\quad
\operatorname{Hom}_{G}(x,y)\ne\varnothing.
\tag{3}
\]

Identities, inverses, and composition make `E_G` an equivalence relation. Hence the ordinary
groupoid orbit space is exactly the setoid quotient `D/E_G`. The setoid forgets arrow multiplicity
and isotropy; the groupoid retains those data for charts, stabilizers, and constructive transport.

If every arrow preserves `U`, then

\[
E_G\subseteq E_{U,D}.
\tag{4}
\]

Call the atlas **fiber-complete** exactly when equality holds. The thin fiber groupoid
`D x_U D => D` realizes equality by construction; a named constructive atlas need not. Under
the inclusion (4), `q_U(W)` is a deterministic function of `q_G(W)`, so the semantic cost of an
incomplete constructive atlas is

\[
\boxed{
H(q_G(W))-H(q_U(W))
=H(q_G(W)\mid U(W))\ge 0.
}
\tag{5}
\]

Equation (5) is the `constructive_quotient_debt`: how much same-statistic equivalence remains
unconnected by the available receiver-realizable arrows. It is zero exactly when the atlas is
fiber-complete on the support of `W`.

For a state-dependent one-step relation `R` that is not already a groupoid, quotient notation is
invalid until the legal-path equivalence is generated:

\[
E_R:=\bigl(\Delta_D\cup R\cup R^{-1}\bigr)^*,
\tag{6}
\]

where `*` is finite transitive closure with every intermediate object constrained to `D`. If each
step preserves `U`, then `E_R subseteq E_(U,D)`; equality remains a separate fiber-completeness
obligation.

## Proof sketch and provenance

1. **`DERIVED-EXACT`:** reflexivity on `D`, symmetry, and transitivity follow from equality in
   `Y`; outside `D`, reflexivity fails, giving the PER typing.
2. **`DERIVED-EXACT`:** `bar U` is well-defined and injective because equal classes are exactly
   equal `U` fibers; it is surjective onto `U(D)` by definition. This is the kernel-equivalence
   specialization of Newstead, Exercise 5.2.4 and Theorem 5.2.35.
3. **`DERIVED-EXACT`:** a bijective relabeling preserves discrete entropy, yielding the semantic
   equality.
4. **`DERIVED-EXACT`:** groupoid axioms prove that (3) is an equivalence relation. Arrowwise
   `U`-invariance proves (4); the entropy chain rule proves (5).
5. **`DERIVED-EXACT`:** (6) is the least equivalence relation containing the legal one-step atlas.
6. **`UNKNOWN`:** existence of a small, receiver-computable section
   `sigma : Q_U -> D` with `q_U o sigma = id` and an archive saving. Neither follows from the
   quotient theorem.

## What the payload statement may and may not say

Permitted statement:

> The semantic payload is a code for the kernel-setoid class, and the legal receiver supplies a
> deterministic representative of that class.

Not permitted without more proof:

> Divide frames by the named transformations and remove that fraction of archive bytes.

The first statement uses only an equivalence quotient. The second wrongly assumes fiber
completeness, a legal receiver section, parameter-space pullback, entropy coding, and byte survival.
Orbit coordinates are not payload; a transverse class label plus the decoder section is.

## Falsification and `verdict_scope`

- A counterexample to symmetry/transitivity or to the bijection is a type/definition error in this
  equation.
- Failure of one proposed atlas to be fiber-complete is `verdict_scope=FORMULATION`, not a kill of
  the evaluator-fiber family.
- Failure of one section to survive parse-back is `verdict_scope=INSTANCE` or `FORMULATION`, as
  preregistered.
- A nontransitive raw one-step relation is **NO-GO** only as a direct quotient
  (`verdict_scope=FORMULATION`); its generated equivalence/groupoid remains open.

## Producers, consumers, and formalization status

Proposed producers:

- frozen observable identifier and support predicate;
- arrow/step invariance receipts;
- generated-orbit component ids;
- fiber-completeness counterexample or proof receipt;
- legal receiver-section and parse-back hashes.

Proposed consumers:

- capstone-171-CGauge quotient accounting;
- sensitivity-map exact-null versus tangent/near-null typing;
- bit allocator zero-bit admission;
- Pareto gate on exact statistic preservation versus counted bytes;
- probe disambiguator: `statistic_kernel` versus `constructive_orbit`.

`FORMALIZATION_PENDING`: no Python canonical-equation object, registry append, or tests are landed
because this unit is analysis-only and the shared equation registry is under live sibling ownership.
The isolated DAG FEED is
`.omx/research/sub015_DAG_infdesc_foundations_20260713.md`; the proposed DSL fields are recorded in
the parent memo. Registration requires main review and serializer custody.

## Sources consulted

- Clive Newstead, *An Infinite Descent into Pure Mathematics*, v0.7, §§5.2
  (`https://cnewstead.codeberg.page/infdesc/infdesc_v0.7.pdf`).
- `.omx/research/weyl_symmetry_group_unification_20260713.md` and its existing equation/DAG FEEDs.
