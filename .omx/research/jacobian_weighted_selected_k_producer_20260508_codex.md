# Jacobian-weighted selected-K producer - 2026-05-08

Scope: PR106/Jacobian-weighted selected-K producer design-to-implementation.
No GPU dispatch was launched. No score, promotion, ranking, kill, or family
status change is supported by this CPU planning scaffold.

## Surfaces Added

- `src/tac/optimization/jacobian_weighted_selected_k.py`
  - Loads a future JSON importance manifest with per-tensor or per-channel
    Jacobian/scorer-pullback values.
  - Fails closed unless metadata records CUDA device provenance and carries no
    diagnostic, proxy, smoke, stub, synthetic, or advisory markers.
  - Reduces per-channel importance to phase-1 per-tensor scalars, then calls
    `JacobianWeightedAllocator`.
  - Emits `weighted_k_allocations[].selected_Ks` in the same schema consumed by
    the no-dead-K builder.
- `tools/build_pr106_jacobian_weighted_selected_ks.py`
  - Thin PR106 wrapper around the reusable module.
  - Parses the PR106 public-frontier archive into tensor streams, uses the
    existing PR106 joint byte encoder, and writes a planning manifest under
    `reports/raw/pr106_jacobian_weighted_selected_ks/` by default.

## Evidence Semantics

Manifest outputs are tagged:

```text
evidence_grade = [CPU-planning jacobian-weighted selected-K producer]
evidence_semantics = cpu_jacobian_importance_selected_k_producer_no_score_no_dispatch
score_claim = false
promotion_eligible = false
rank_or_kill_eligible = false
ready_for_exact_eval_dispatch = false
dispatch_attempted = false
score_affecting_payload_changed = false
charged_bits_changed = false
downstream_selected_Ks_can_change_charged_bits = true
```

Default blockers retained on producer output:

- `selected_Ks_cpu_planning_not_score_authority`
- `requires_byte_closed_no_dead_K_archive_rebuild`
- `requires_static_archive_preflight`
- `requires_exact_cuda_auth_eval_before_score_claim`

## Gate Behavior

The producer rejects:

- CPU/MPS/missing-device importance metadata;
- metadata marked diagnostic/proxy/smoke/stub/synthetic/advisory;
- all-zero importance;
- uniform importance by default;
- missing tensor coverage;
- malformed per-channel or per-tensor values.

The CUDA gate only validates the provenance declared by the future importance
manifest. The selected-K allocation itself remains CPU-safe and does not load
the scorer.

## Tests

Focused tests cover:

- `weighted_k_allocations[].selected_Ks` schema emission;
- per-channel-to-per-tensor reduction;
- CUDA metadata gate failure on CPU input;
- diagnostic/proxy metadata rejection;
- uniform and missing-tensor rejection.

## Next Exact CUDA Gate

Once a certified non-diagnostic PR106 Jacobian importance manifest exists:

1. Run `tools/build_pr106_jacobian_weighted_selected_ks.py` to emit selected
   Ks.
2. Rebuild a byte-closed no-dead-K archive from the selected-K row.
3. Run static pre-submission compliance on the exact archive/runtime packet.
4. Claim the lane and run exact CUDA auth eval.
5. Only after structured CUDA adjudication and adversarial result review may
   any score, promotion, rank, kill, or family-status decision change.
