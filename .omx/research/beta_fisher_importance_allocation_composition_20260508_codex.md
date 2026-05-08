# Beta-Fisher Importance Allocation Composition - 2026-05-08

## Scope

This note records a narrow manifest-level integration between the generic
Jacobian/Fisher importance allocator and the PR101 no-dead-K lossy coarsening
archive builder.

The generic allocator remains planning-only. It may consume per-weight or
per-tensor importance, boundary mass, and texture capacity, then emit selected
per-tensor planning rows. The no-dead-K builder may now read
`allocation.selected_by_tensor[].K` from that generic manifest after matching
`allocation.target_distortion` to the requested `--selected-Ks-rms-target`.

## Guardrails

- Accepted generic manifests must use `allocation.objective == "target_distortion"`.
- Accepted generic manifests must declare schema
  `jacobian_fisher_importance_allocator.v1`; unrelated manifests with an
  `allocation` object are not interpreted as K vectors.
- `allocation.target_distortion` must match the requested RMS target exactly
  within numeric tolerance.
- Every selected tensor row must match the builder's `FIXED_STATE_SCHEMA`
  tensor order by both `tensor_index` and `tensor_name`.
- Every selected tensor row must carry an integer `K`.
- Source dispatch blockers and evidence semantics are preserved into the CPU
  build guard fields.
- CPU/MPS/proxy importance remains non-promotable planning metadata; exact CUDA
  archive evidence is still required before score, rank, promotion, or kill use.

## Integration Point

The adapter lives in:

- `tools/build_admm_x_lossy_coarsening_path_b_step6_no_dead_k.py`

The focused tests live in:

- `src/tac/tests/test_build_admm_x_lossy_coarsening_path_b_step6_no_dead_k.py`

## Evidence Status

Evidence grade: planning/control only. This change does not claim score
movement, does not dispatch eval, and does not by itself prove any archive is
better. It only makes the downstream archive builder able to consume a typed
importance-allocation plan without editing source constants.
