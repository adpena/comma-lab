# Jacobian-Weighted Importance Allocator - 2026-05-08

Scope: CPU-side foundation for the user's "pixel is weight is deterministic"
allocation idea. No GPU dispatch was launched. No score, promotion, ranking,
kill, or family-status claim is supported by this artifact.

## Existing Surfaces Inspected

- `src/tac/optimization/lagrangian_per_tensor_allocation.py`
  - Canonical per-tensor lambda allocator.
  - Existing weight convention: higher weight protects a tensor because the
    cost includes `lambda * weight[t] * rel_err[t]^2`.
  - Already has `UniwardWeightedAllocator` and `JacobianWeightedAllocator`
    hooks for tensor-order weights.
- `src/tac/optimization/beta_fisher_lossy_weights.py`
  - Converts Fisher/sensitivity maps plus boundary and texture/capacity priors
    into tensor weights for lossy-coarsening selected-K output.
- `src/tac/optimization/jacobian_weighted_selected_k.py`
  - PR106 selected-K bridge that requires future CUDA-authored, non-proxy
    Jacobian importance metadata before producing selected K vectors.
- `src/tac/codec/cost_curves.py`
  - Current lossy-coarsening K curves use `K`, `byte_proxy`, and `rel_err`.
- `src/tac/bit_allocator.py`
  - Prior per-weight Fisher bit-budget allocator with monotonic bit allocation
    and budget conservation tests.
- `src/tac/uniward_texture.py` and UNIWARD ledgers
  - UNIWARD-like texture/capacity remains a capacity prior only, not score
    authority.

## New CPU Primitive

Added `src/tac/optimization/jacobian_fisher_importance_allocator.py`.

It accepts:

- per-tensor importance, or per-weight importance reduced by `mean`, `sum`,
  `max`, or `rms`;
- optional boundary mass, which raises protection weight;
- optional texture/capacity, which lowers protection weight only after score
  importance is present;
- candidate curves using current lossy-coarsening fields (`K`, `byte_proxy`,
  `rel_err`) or generic precision fields (`precision`, `bits`, `bytes`,
  `error`).

It emits:

- one selected candidate per tensor;
- selected `K` / precision / bit-depth fields when present;
- `total_bytes`, weighted RMS error, unweighted RMS error, max error;
- tensor-order allocator weights compatible with the canonical Lagrangian
  convention.

Supported objectives:

- `target_distortion`: choose the lowest-byte candidate vector whose weighted
  RMS error meets the target.
- `byte_budget`: choose the lowest weighted-error candidate vector whose total
  bytes fit the budget.

The selector is deterministic, fail-closed on malformed curves/importance,
and keeps non-promotable metadata in the result.

## CLI

Added `tools/jacobian_fisher_importance_allocator.py`.

The CLI is a thin JSON wrapper around the module:

```text
.venv/bin/python tools/jacobian_fisher_importance_allocator.py \
  --curves-json <curves.json> \
  --importance-json <importance.json> \
  --target-distortion 0.03 \
  --output-json <manifest.json>
```

or:

```text
.venv/bin/python tools/jacobian_fisher_importance_allocator.py \
  --curves-json <curves.json> \
  --importance-json <importance.json> \
  --byte-budget 150000 \
  --output-json <manifest.json>
```

## Evidence Semantics

All manifests are planning-only:

```text
evidence_grade = [CPU-planning empirical/prediction jacobian-fisher-importance allocator]
evidence_semantics = cpu_importance_weighted_quantization_allocation_no_score_no_dispatch
score_claim = false
promotion_eligible = false
rank_or_kill_eligible = false
ready_for_exact_eval_dispatch = false
dispatch_attempted = false
score_affecting_payload_changed = false
charged_bits_changed = false
downstream_selection_can_change_charged_bits = true
```

Default blockers:

- `cpu_planning_allocator_not_score_authority`
- `requires_cuda_pixel_jacobian_or_fisher_pullback`
- `requires_importance_artifact_custody_and_calibration`
- `requires_byte_closed_archive_rebuild`
- `requires_static_pre_submission_compliance`
- `requires_exact_cuda_auth_eval_before_score_claim`

## Tests

Focused tests added in
`src/tac/tests/test_jacobian_fisher_importance_allocator.py` cover:

- monotonic tightening as target distortion gets stricter;
- high-importance tensors receiving tighter `K` under distortion targets;
- high-importance tensors receiving tighter `K` under byte budgets;
- per-weight importance reduction to tensor-order allocator weights;
- invalid curves, non-finite/negative values, dual objectives, and infeasible
  budgets failing closed;
- output metadata blocking promotion/ranking;
- CLI JSON smoke path.

Verification run:

```text
.venv/bin/python -m pytest src/tac/tests/test_jacobian_fisher_importance_allocator.py -q
# 7 passed in 0.13s

.venv/bin/python -m pytest \
  src/tac/tests/test_optimization_lagrangian_per_tensor_allocation.py \
  src/tac/tests/test_jacobian_weighted_selected_k.py \
  src/tac/tests/test_beta_fisher_lossy_weights.py -q
# 30 passed in 5.50s
```

## Exact CUDA Integration Blockers

This primitive becomes useful for score work only after:

1. CUDA pixel/scorer pullback exists for the exact deterministic decoder path.
2. Per-weight or per-channel Jacobian/Fisher buffers are reduced to the same
   tensor order as the byte/error curves.
3. Top selected perturbations are calibrated with byte-level finite differences
   through a byte-closed archive path.
4. A builder consumes the selected `K` / precision values and records old/new
   archive SHA-256, changed payload SHA-256, no-op proof, and runtime closure.
5. Static pre-submission compliance passes.
6. A dispatch claim is opened, exact CUDA auth eval runs, and the result is
   adversarially reviewed before any score/rank/promotion status changes.

Until then, this is empirical/prediction planning evidence only.
