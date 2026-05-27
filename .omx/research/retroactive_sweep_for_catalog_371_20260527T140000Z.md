# Retroactive sweep for Catalog #371 (check_no_orphan_auto_trigger_stub_with_satisfied_condition)

**Per Catalog #348** (event-driven retroactive verdict-taint sweep): every new
STRICT preflight gate landing ships this 4-field sweep memo so historical
KILL / DEFER / FALSIFY verdicts whose evidence basis the new gate invalidates
are re-evaluated.

## 1. Bug-class symptom signature

A no-op `auto_*` / `recalibrate_*` / `refresh_*` / `trigger_*` function that
returns WITHOUT acting even when its OWN documented trigger condition is
already satisfied. Canonical instance: `tac.canonical_equations.registry.
auto_recalibrate_from_continual_learning_posterior` hardcoded
`equations_recalibrated=0,  # stub; auto-refit comes in a follow-on landing`
and no-op'd on every equation, including the 11 (of 64) whose stored
`predicted_vs_empirical_residual` summary no longer matched their own landed
`EmpiricalAnchor` rows — the "stale-prior orphan" failure mode.

## 2. Pre-fix window

The stub shipped with the canonical-equations registry landing (Catalog #344,
2026-05-19) and remained a no-op through 2026-05-27. During that window any
agent reading an equation's `predicted_vs_empirical_residual` or
`is_well_calibrated` consumed a stale summary that the recalibrator was
supposed to keep current but never did.

## 3. Historical KILL / DEFER / FALSIFY verdict search

Searched `.omx/research/` and the canonical-equation registry for verdicts that
depended on a canonical-equation residual summary or `is_well_calibrated` flag:

- **canonical equation #2** `hinton_kl_distill_enables_qat_catalyst_composition_savings_v1`:
  3 anchors, 2nd `IMPLEMENTATION_LEVEL_FALSIFIED` (Catalog #307) + 3rd
  `PARTIAL_CONFIRMATION`. Stored summary carried a stale `nan` axis
  (`paired_qat_with_without_distill_anchor`) + a synthetic
  `t3_council_binding_revision_landing_axis: 0.0` that did NOT correspond to
  any landed anchor. The IMPLEMENTATION-LEVEL falsification verdict per
  Catalog #307 is PRESERVED (paradigm intact, alpha=0.15 closed-form lift
  falsified at implementation level toward empirical ~0). The refit makes the
  residual summary evidence-faithful; it does NOT change the verdict.
- **10 sister equations** (`arith_coded_cls_stream`, `cascade_a_fec10_hybrid`,
  `hfv2_sparse_pair_sidecar`, `hinton_distilled_scorer_surrogate`,
  `markov_context_selector`, `master_gradient_null_space_byte_fraction`,
  `mlx_pytorch_conv2d_fp64`, `mlx_pytorch_conv2d_kahan`,
  `mlx_pytorch_drift_vs_training_depth_z6`,
  `procedural_codebook_from_seed_compression_savings`,
  `residual_hybrid_boosting`): all had stale summaries refit to match their own
  anchors. NONE carried a KILL/FALSIFY verdict that the stale summary
  invalidated — the verdicts were anchor-level and the anchors themselves are
  preserved; only the per-axis SUMMARY drifted. No verdict re-evaluation
  required.

## 4. Per-finding RE-EVAL priority

| Finding | RE-EVAL priority | Disposition |
|---|---|---|
| eq#2 alpha=0.15 IMPLEMENTATION-LEVEL falsification (Catalog #307) | NONE | Verdict PRESERVED; refit only re-summarizes residuals; paradigm intact per Forbidden-premature-KILL |
| 10 sister equations stale summaries | NONE | Anchor-level verdicts unaffected; summaries refit to evidence-faithful; idempotent |
| `mlx_cuda_bidirectional_drift` + `daubechies_multi_scale_wavelet` NaN-pending sentinels | NONE | 1 anchor each → trigger NOT satisfied → LEGITIMATE-DEFERRED (awaiting paired-dispatch anchor); refit correctly leaves them |

**Conclusion:** no historical KILL/DEFER/FALSIFY verdict was tainted by the
stale-prior orphan-stub. The fix is summary-coherence only (re-derives the
residual SUMMARY from already-landed anchors; never synthesizes a measurement
per Catalog #287/#323). The verdict-preservation discipline per CLAUDE.md
"Forbidden premature KILL" + Catalog #307 holds: eq#2's implementation-level
falsification stands and the paradigm remains intact.
