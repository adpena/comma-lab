# Engineered Correction Component-Trace Gate

Date: 2026-05-06

## Status

Implemented guarded local-patch readiness. This is not a score claim and does
not make engineered corrections eligible for exact-eval dispatch.

## What Landed

- `tac.engineered_correction_readiness.audit_sparse_corrections(...)` accepts
  an optional PR85 scorer-gradient atom plan via `component_trace_plan`.
- `tools/audit_engineered_corrections.py` exposes:
  - `--component-trace-plan`
  - `--require-component-trace-plan`
- The gate validates that the component-trace plan is planning-only, non-score,
  non-dispatch, digest-stable, CUDA-cross-checked to exact eval, and contains
  positive component-trace atoms whose own dispatch gates remain blocked.

## Required Plan Contract

The accepted plan schema is:

```text
pr85_scorer_gradient_atom_opportunity_v1
```

Required safety properties:

- `planning_only=true`
- `score_claim=false`
- `dispatch_performed=false`
- `remote_jobs_dispatched=false`
- `stable_plan_digest_sha256` matches the canonical payload digest
- at least one `diagnostic_component_trace` input with
  `trace_cross_checked_to_exact_eval=true`
- at least one positive component-trace atom
- every atom remains `promotion_eligible=false`
- every atom dispatch gate remains `dispatchable=false`

## Remaining Blocker

This unlocks a guarded local patch readiness path only. A score-lowering
candidate still needs a byte-closed archive that consumes the charged correction
bytes, a non-noop candidate diff, archive preflight, lane claim, and exact CUDA
auth eval.
