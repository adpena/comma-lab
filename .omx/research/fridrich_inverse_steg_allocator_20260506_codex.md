# Fridrich Inverse-Steganography Allocator

Date: 2026-05-06

## Status

Implemented guarded planner surface. This is optimizer feedback only, not a
score claim and not an exact-eval dispatch surface.

## What Landed

- `tac.uniward_delta.build_detector_cost_manifest(...)` emits deterministic
  atom rankings from charged bytes, detector capacity, and positive scorer
  sensitivity.
- `tac.hnerv_section_repack.detector_cost_atoms_from_section_plan(...)` maps
  HNeRV section-repack rows into detector-cost atoms.
- `tac.engineered_correction_readiness.detector_cost_atom_from_correction_report(...)`
  maps sparse correction readiness reports into detector-cost atoms.

## Formula

The planner ranks rows with:

```text
positive_scorer_sensitivity * detector_capacity / charged_bytes
```

`detector_capacity` must be grounded in one of:

- explicit UNIWARD/Fridrich capacity feedback,
- `1 / (1 + detector_cost)`,
- HNeRV section entropy proxy `entropy_bits_per_byte / 8`.

Rows missing detector capacity or positive scorer sensitivity are retained with
zero priority and explicit risk reasons. This preserves negative/unknown signal
without silently promoting arbitrary atoms.

## Dispatch Blockers

All emitted manifests set:

- `score_claim=false`
- `dispatch_attempted=false`
- `ready_for_exact_eval_dispatch=false`

The fixed blockers are:

- `fridrich_detector_cost_is_optimizer_feedback_only`
- `requires_charged_archive_consumption`
- `requires_archive_manifest_preflight`
- `requires_exact_cuda_auth_eval_on_candidate`

## Verification

Focused tests cover HNeRV section rows and engineered sparse corrections feeding
the detector-cost manifest, including non-promotable dispatch state.
