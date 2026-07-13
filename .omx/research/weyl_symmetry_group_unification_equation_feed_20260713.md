# Canonical equation FEED — Weyl evaluator-fiber rate law v1

**Date:** 2026-07-13

**Proposed equation id:** `weyl_evaluator_fiber_rate_law_v1`

**Status:** DESIGN / equation feed; not registered

**Authority:** DERIVED-EXACT mathematics with explicitly conditional empirical dimensions

**Pointer:** UNMOVED

## One-line law

For a legal witness space `X`, frozen statistic `U`, and legal deterministic decoder
`D`, the exact invariance object is the fiber groupoid `G_U = X x_U X`; its bisection
group permutes each fiber independently; the ideal invariant rate is `H(U(X))`, while
the contest-valid single-instance rate is the shortest legal payload decoding into the
same `U` fiber.

## Canonical form

\[
\boxed{
\begin{gathered}
\mathfrak G_U:=X\times_UX\rightrightarrows X,\\
\operatorname{Bis}(\mathfrak G_U)
\cong\prod_{u\in\operatorname{im}U}\operatorname{Sym}(U^{-1}(u)),\\
R_U(0)=H(X/\mathfrak G_U)=H(U(X)),\\
L^*_{U,D}(x)=\min_z\{|C(z)|:U(D_{\rm legal}(z))=U(x)\}.
\end{gathered}
}
\tag{Weyl-fiber-rate-v1}
\]

Use `U=S=(d_seg,d_pose)` for the maximal scalar-score invariance and `U=T=(A,P)` for
the conservative frozen-scorer-statistic compiler target.

## Derivation status

1. `X/G_U ~= im U` follows exactly from equality of fibers.
2. The bisection group of the equivalence-relation groupoid is the direct product of
   symmetric groups over fibers because each fiber may be permuted independently.
3. The Shannon equality is exact for an ideal lossless code of the orbit label.
4. The MDL form adds the contest's legal-decoder section requirement. It prevents a
   formal score quotient from smuggling scorer weights, GT statistics, or source data.

Items 1-4 are **DERIVED-EXACT**. They do not assert that a practical section exists.

## Domain of validity

Included:

- finite legal uint8 witness space, or a measurable extension with the corresponding
  measurable automorphism group;
- fixed frozen scorer and fixed source reference;
- exact statistic equality for the zero-distortion statement;
- deterministic, portable, contest-legal decoder in the MDL statement;
- section-local exact byte measurement rather than dimension-proportional estimates.

Excluded:

- a claim that `H(S(X))` is achievable without a legal section;
- a claim that a tangent null direction integrates to a finite score orbit;
- a claim that frame-space dimension equals compressed archive bytes;
- generic physical `SE(3)` motion as an invariance of a fixed-reference score;
- pointer movement without an exact archive/hardware-axis row.

## Conditional dimension anchors

| anchor | value | label | condition |
|---|---:|---|---|
| blind coordinate subgroup | 1,385,424 pair RGB coords; 22.696926% | MEASURED exact | arbitrary n600 fill survives both preprocessors bit-identically |
| real-linear `ker R` | 4,924,368 / 6,104,016; 80.674232% | MEASURED + DERIVED | real lift / legal partial translations |
| joint preprocessor null | 5,219,280 / 6,104,016; 85.505674% | DERIVED-EXACT local | fixed unclipped YUV6 stratum |
| Pose tangent null | 589,818 / 589,824; 99.998983% | MEASURED tangent-only | four frame-0 working points; finite-width promotion falsified |
| robust argmax occupancy | 112,263,287 / 117,964,800; 95.166768% | MEASURED occupancy | source logit margin >2; not deformation volume |
| current direct V9 blind-byte saving | 0 B | DERIVED from payload homes | current blob contains weights + latent code, not camera pixels |
| mod32 to mod19 raw code delta | -15,600 int8 symbols | DERIVED from table shape | compressed/archive and distortion deltas unknown |

## Required empirical anchor before registration as a byte law

Registration as a mathematical equation may proceed after review. Registration as an
empirically calibrated byte law requires one receiver-consumed section A/B with:

1. exact same input/checkpoint/config except the canonical orbit representative;
2. exact decode and parse-back custody;
3. exact `T` equality, or an explicitly declared `S`-cell tolerance;
4. section and total archive bytes before/after;
5. archive hashes and a legal receiver-section hash;
6. exact evaluator row on its declared hardware axis if promotion is requested.

## Falsification and reactivation

The mathematical fiber law is falsified only by a definition/type error. A proposed
constructive quotient is rejected if its receiver cannot derive the section, if exact
statistic equality fails, or if compressed bytes do not improve. Such a rejection is
FORMULATION scoped. Reactivate with a new chart, receiver section, or entropy code; do
not kill the evaluator-fiber family.

## Proposed consumers

- V9 CGauge byte-close compiler: section-level `weyl_orbit_accounting` row.
- capstone-171-CGauge: formal rate-gap and receiver-section obligation.
- bit allocator: zero-bit admission only after finite exact-statistic closure.
- sensitivity map: distinguish exact fiber, local tangent, and near-null directions.
- probe disambiguator: decoded-frame-preserving parameter gauge versus scorer-fiber move.

## Triality note

- Equations: this file.
- DAG: `.omx/research/weyl_symmetry_group_unification_DAG_FEED_20260713.md`.
- DSL: the proposed row schema in
  `.omx/research/weyl_symmetry_group_unification_20260713.md` section 5.3.

The shared equation registry is intentionally untouched because this is design-only and
the live sibling ownership map includes a concurrent canonical-equation landing.
