# Component Sensitivity Byte-Plan Manifest - 2026-05-06

This is a planning/DX ledger, not a score claim and not a dispatch packet.

- tool surface: `experiments/sweep_owv3_byte_plan.py --manifest-only`
- hidden-gem key: `component_sensitivity_byte_allocator`
- score_claim: `false`
- dispatch_attempted: `false`
- ready_for_exact_eval_dispatch: `false`
- evidence_grade: `planning_manifest_only`

## Contract

The manifest-only mode records the deterministic OWV3 byte-plan grid, candidate
IDs, sensitivity-map file custody, frontier comparator metadata, and explicit
fallback-action accounting without loading renderer models, building archives,
or running scorer/eval code.

Dispatch remains blocked by design:

- `manifest_only_no_archive_bytes`
- `cuda_auth_eval_required_for_score`
- `component_balanced_sensitivity_required_before_promotion`
- `authoritative_cuda_sensitivity_required` when the caller has not opted into
  non-authoritative planning
- `missing_sensitivity_map` when the input sensitivity artifact is absent

## Use

Use this as the fast preflight before the heavier OWV3 sweep. It is suitable
for DX iteration, grid review, and fallback-policy review. It is not suitable
for ranking, score claims, promotion, or remote dispatch. The full sweep and
exact CUDA auth eval remain required before any candidate can become evidence.
