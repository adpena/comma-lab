# Codex premise falsification — measured MDL is not a universal-K lower bound

Date: 2026-07-18

Lane: `lane_mdl_ms_complex_k_lower_bound_20260718`

Scope: `CLAIM_LEVEL_ONLY` · `research_only=true` · `[macOS-CPU advisory] NON-PROMOTABLE`

Pointer: `UNCHANGED` · score/launch authority: `NONE`

## Falsified premise

`A concrete lossless MDL of the n600 digital/MS-like complex is a numeric lower bound on individual
universal Kolmogorov complexity.`

**Verdict: FALSIFIED_AT_CLAIM_LEVEL [DERIVED].** For fixed decoder `D_C` and emitted code `c_C(T)`,

\[
K_U(T\mid D_C) \le |c_C(T)|+O_U(1),
\]

so the measured codelength is a declared-family upper bound. For fixed evaluator `E`, the true
`K_U(T_E|E)` with `T_E=(S,P)` (SegNet argmax plus frozen PoseNet output) would lower-bound an exact
witness up to a machine constant, but it is uncomputable and was not numerically lower-bounded by
this lane. Temporal ξ belongs to the carrier description and is not evaluator output.

## Canonical evidence pointers

- Full derivation, n600 counts, Seg/ξ split, strict rate ceiling, and round-1 audit:
  `.omx/research/mdl_ms_complex_K_lower_bound_20260718.md`
- Machine-readable receipt:
  `.omx/research/mdl_ms_complex_K_lower_bound_20260718.json`
- Reproducible measurement:
  `tools/measure_mdl_ms_complex_k_lower_bound.py`
- Consumer routing and corrected equation debt:
  `.omx/research/mdl_ms_complex_K_lower_bound_DAG_FEED_20260718.md`

## Required consumer behavior

Do not reuse the old `MDL(MS) <= K` bracket as a universal statement. Consume concrete description
lengths only as achievability evidence from above. MAIN review must adjudicate the corrected equation
candidate before canonical registration. No representation family is killed; the verdict is scoped to
the invalid inequality and to code instances lacking exact receiver closure.
