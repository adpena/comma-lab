# Field-Equation Planner

Date: 2026-05-06
Agent: Codex
Evidence grade: derivation / empirical software tests

## Purpose

The planner turns cross-paradigm archive atoms into a common mathematical
surface:

```text
J = expected_total_score_delta
  + lambda_byte * byte_violation
  + lambda_seg * seg_violation
  + lambda_pose * pose_violation
  + lambda_conf * confidence_violation
```

It treats each atom as a first-order perturbation around the current exact
frontier and emits Frechet-style derivatives:

```text
d_score / d_epsilon
d_seg_dist / d_epsilon
d_pose_dist / d_epsilon
d_bytes / d_epsilon
```

Optional pair rows encode second-order Volterra interactions:

```text
Delta S(A + B) = Delta S(A) + Delta S(B) + Delta^2 S(A, B)
```

## Change

Added:

- `src/tac/optimization/field_equation_planner.py`
- `tools/build_field_equation_plan.py`
- `src/tac/tests/test_field_equation_planner.py`

The output includes:

- KKT-style residuals and blockers.
- MDL description-length terms.
- A research-basis manifest tying each atom family to source papers, local
  variables, contest terms, charged-byte contracts, and hardening blockers.
- A planning-only theoretical floor estimate.
- A trainable surrogate contract for future differentiable selectors or
  atom-amplitude learners.

## Boundaries

This does not claim a score, dispatch GPU work, or prove a Shannon floor. The
floor field is an optimistic derivation under stated independence/interactions
and remains blocked on exact stacked archive CUDA evidence.

Paper-derived priors are also planning-only. They can justify variables and
proposal mechanisms, but every row still carries
`research_basis_is_not_score_evidence` until byte-closed exact CUDA evidence
exists.

## Verification

```text
.venv/bin/python -m pytest src/tac/tests/test_field_equation_planner.py -q
```
