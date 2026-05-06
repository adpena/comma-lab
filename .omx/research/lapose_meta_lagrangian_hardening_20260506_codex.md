# LA-POSE Meta-Lagrangian Hardening - 2026-05-06

## Summary

Goodall's read-only adversarial review found that LA-POSE planning artifacts
were dropping hard-pair identity and that globally allocated component-response
deltas could enter the meta-lagrangian ledger as rankable atoms. Both are
scientific-rigor bugs: they make cross-paradigm allocation less faithful to the
hard-pair evidence surface and can overstate evidence that was inferred rather
than measured per pair.

## Fixes

- `inputs_from_pair_metric_payload` now carries `hard_pair_rank` and
  `hard_pair_support=[pair_index]` on latent actions and pair opportunities.
- `records_from_component_response` preserves `hard_pair_rank` and defaults
  missing `hard_pair_support` to `[pair_index]` rather than silently dropping
  identity.
- `build_motion_atom_manifest` preserves `hard_pair_rank` and
  `allocation_inference` into emitted atoms.
- `meta_lagrangian_allocator` marks `allocation_inference=true` atoms
  non-rankable with blocker `allocated_global_response_not_rankable`.

## Rigor Boundary

Global component-response allocation remains useful proposal feedback, but it
is not per-pair measured evidence. It may produce planning ledgers and cross-
paradigm feature surfaces, but it cannot drive rankable promotion or dispatch
without a byte-closed archive consumer and exact CUDA auth eval.

## Verification

```bash
.venv/bin/python -m pytest \
  src/tac/tests/test_lapose_lite_inputs.py \
  src/tac/tests/test_lapose_motion_evidence.py \
  src/tac/tests/test_lapose_motion_atoms.py \
  src/tac/tests/test_lapose_planning_chain.py \
  src/tac/tests/test_meta_lagrangian_allocator.py \
  -q
```

Result: `20 passed`.

## Remaining LA-POSE Work

- Add class-support ingestion from mask/sensitivity artifacts with class-id
  validation and source hashes.
- Add openpilot policy/world manifests and fail closed on zero-feature fallback
  outside explicit smoke mode.
- Add deterministic tool manifests with input SHAs, argv, runtime/repo hash,
  output self-hash, and byte-identical rerun tests.
- Route LA-POSE atoms into concrete charged builders: CMG3/PMG residual
  policy, pose residual policy, foveation centers, and HNeRV/joint-codec
  conditioning. Each builder remains `score_claim=false` until exact CUDA.
