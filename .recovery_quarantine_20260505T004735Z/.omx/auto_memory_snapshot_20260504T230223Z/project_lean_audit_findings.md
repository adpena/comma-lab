---
name: Lean Audit Findings (Tao Grade)
description: Tao audit of proofs — correct but 4 missing theorems, compilation risks, optimality gap in Theorem 1
type: project
originSessionId: ecc348c1-2829-48d3-9d4c-10ed2aa188a8
---
## Status: Correct but not publication-ready

The proofs are mathematically sound. Zero sorry. But:

1. Theorem 1 proves a ratio identity, not optimality — needs a formal hypothesis
   about linear gradient-to-distortion relationship.

2. KL T² compensation (Hinton result) is asserted in comments but never proved.
   This is the most important missing theorem for the paper.

3. Convergence of the adaptive system is not proved.
   Does iterating w_s ← 20*sqrt(10*p) produce a fixed point?

4. Tight ceiling (lim A(bw) = 1/β as bw→∞) is not proved.
   Simple tendsto argument in Mathlib.

5. Compilation risks: div_le_div_of_nonneg_right signature, sq_le_sq' name,
   field_simp with nested divisions.

## For arXiv
- Convert to Lake project with Mathlib imports
- Add 4 missing theorems
- Add linearity assumption to Theorem 1
- Replace 4 axioms with Mathlib lemma references

## Proxy Correction (Contrarian)
- Proxy failure was pose-cap specific, NOT fundamental to KL distill
- With cap removed, proxy should be trustworthy
- Calibrate α_p with 3 authoritative evals of cap-free checkpoints
- confidence(p) = min(1, p/p_baseline) — flag when p < 0.3*baseline
