# Bayesian Experimental Design For Exact CUDA Candidate Selection

Date: 2026-05-06
Owner: codex
Evidence grade: empirical planning tool
Score claim: false
Dispatch attempted: false
GPU launched: false

## Scope

This ledger records a planning-only Bayesian experimental design surface for
selecting which candidate archives/configs deserve scarce exact CUDA auth eval.
It does not claim scores, launch jobs, or promote candidates.

Implemented surfaces:

- `src/tac/optimization/bayesian_experimental_design.py`
- `tools/rank_exact_eval_information_gain.py`
- `src/tac/tests/test_bayesian_experimental_design.py`

## Deterministic Formulas

Expected improvement uses the closed-form minimization acquisition for a
Gaussian score surrogate:

`EI = (best - mu) * Phi((best - mu) / sigma) + sigma * phi((best - mu) / sigma)`

The incumbent score must be exact CUDA evidence supplied by the caller or an
upstream custody artifact. The module only consumes the value; it does not make
that score claim.

Expected information gain uses Gaussian entropy reduction:

`EIG = 0.5 * log(prior_variance / posterior_variance)`

Family uncertainty reduction follows a joint-Gaussian update. An exact eval of
one candidate reduces uncertainty for its source family and for related
families via caller-provided coupling coefficients in `[0, 1]`.

## Dispatch Discipline

Every report and row carries:

- `score_claim=false`
- `promotion_eligible=false`
- `dispatch_attempted=false`
- `gpu_launched=false`
- `dispatch_blockers=[...]`

`ready_for_exact_eval_dispatch` remains false unless the caller provides exact
archive custody with a real archive path, matching SHA-256, and matching byte
count. Even when readiness is true, the tool remains rank-only and launches no
GPU work.

## Current Status

The module ranks candidates by:

`acquisition = expected_improvement_weight * EI + information_gain_weight * EIG`

This creates an auditable bridge between public-frontier byte opportunities,
family-level uncertainty, and the next exact-eval queue decision without
turning proxy predictions into score evidence.
