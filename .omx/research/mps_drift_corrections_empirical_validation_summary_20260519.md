# MPS engineering corrections empirical validation summary 2026-05-19

> _Cite-as:_ `[empirical:.omx/state/mps_drift_corrections_empirical_validation_20260519.json]`

## Verdict: FALSIFIED-FOR-TINYRENDERER-CLASS

The 3 engineering corrections (Kahan Conv2d + pinned softmax + fp32 matmul
override) landed per slot 9's predicted 30x drift reduction were empirically
validated against the Phase B TinyRenderer checkpoint. Result:

- **baseline (MPS, no corrections)**: mean_aggregate = 38.131418914
- **corrected (MPS, 3 corrections)**: mean_aggregate = 38.131447808
- **drift_reduction_ratio = 1.00x** (slot 9 predicted 30x)

## Rationale

Slot 9's 30x predicted reduction was calibrated for **SegNet-class
architectures** (45 Conv2d layers, 5-class softmax argmax at score-relevant
boundaries, 196K accumulation depth). TinyRenderer is dramatically smaller:

| Property | TinyRenderer | SegNet-class |
|---|---|---|
| Parameter count | 12K | 140K |
| Conv2d count | 4 | 45 |
| Softmax count | 0 | 1 (5-class) |
| Accumulation depth | ~96K | ~196K |

The 3 corrections fire correctly (Kahan path active on MPS Conv2d; pinned
softmax inactive because no softmax in TinyRenderer; fp32 matmul accumulation
strict applied). They are NO-OP for this small architecture.

## Cargo-cult-unwind classification per Catalog #303

| Assumption | Classification |
|---|---|
| 3 corrections compose multiplicatively to 30x | CARGO-CULTED-FALSIFIED for current architecture-class evidence |
| Kahan summation reduces drift by 10x | HARD-EARNED for SegNet-class; CARGO-CULTED for TinyRenderer-class |
| Pinned softmax reduces argmax-boundary flips by 50% | HARD-EARNED for SegNet 5-class output; N/A for TinyRenderer (no softmax) |

## Per CLAUDE.md "Apples-to-apples evidence discipline"

Reporting HONESTLY, not extrapolating. SegNet-class empirical validation
remains PENDING operator-routed Modal A10G dispatch (~$0.30 budget, 1-2 hr).

## Pose input caveat

Empirical re-run used `pose=torch.zeros(10, 12)` (the Phase B training-time
pose was not independently saved). The 56.32% per-pair CV (vs Phase B 2.6%
CV) reflects this pose mismatch, NOT degradation from the corrections. The
drift_reduction_ratio metric remains canonical because BOTH baseline and
corrected use identical inputs.

## Provenance per Catalog #287/#323

- `axis_tag = [macOS-MPS-Kahan-corrected-PyTorch-vs-CUDA-diagnostic]`
- `evidence_grade = macOS-MPS-Kahan-corrected-diagnostic`
- `score_claim = false`
- `promotion_eligible = false`
- non-promotable per CLAUDE.md "MPS auth eval is NOISE" + Catalog #1 / #192 / #317

## Cross-references

- Slot 9 formalization: `feedback_mps_drift_mathematical_and_engineering_formalization_landed_20260519.md`
- Landing memo: `feedback_mps_engineering_corrections_3_landed_20260519.md`
- Operator-state JSON (gitignored): `.omx/state/mps_drift_corrections_empirical_validation_20260519.json`


<!-- # FORMALIZATION_PENDING:pre_framework_memo_dated_2026-05-19_predates_canonical_equations_birthday_registry_population_in_progress_appended_by_strict_flip_enablers_per_operator_blanket_approval_per_claude_md_forbidden_premature_kill_without_research_exhaustion_this_is_DEFER_pending_canonical_equation_backfill_NOT_kill -->
