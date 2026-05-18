# Codex routing directive: canonical Phase 1 Fisher-precondition helper package
# Date: 2026-05-18
# Operator: closure campaign master memo `closure_campaign_pursue_and_confirm_master_20260518.md` op-routable OPR-CLOSE-2 batch
# Authority: `.omx/research/phase_1_fisher_precondition_canonical_helper_design_memo_20260518.md` (THE design memo; 129.3 KB)
# Sister: closure_campaign_pursue_and_confirm_master_20260518.md §11.2 TOP-5 (OPR-CLOSE-2)
# Per CLAUDE.md "Subagent coherence-by-default" + Catalog #245 4-layer pattern + Catalog #290 + #294 + #303 + #305 + #296

## CANONICAL POINTERS (read FIRST)

1. `/Users/adpena/Projects/pact/CLAUDE.md` (FULL; especially Catalog #125 6-hook + #245 4-layer + #131 fcntl-locked JSONL + #270 dispatch optimization protocol)
2. `/Users/adpena/Projects/pact/AGENTS.md`
3. `.omx/research/phase_1_fisher_precondition_canonical_helper_design_memo_20260518.md` (THE AUTHORITY design memo with canonical-vs-unique decision per layer + 9-dim checklist + cargo-cult audit + observability surface + Dykstra-feasibility predicted-band; council: T2 sextet + Amari memorial + Boyd + MacKay + Carmack)
4. `.omx/research/closure_campaign_pursue_and_confirm_master_20260518.md` (parent coordinator memo)
5. `.omx/research/cross_stack_synthesis_9_design_landings_unified_framework_20260518.md` §4 FISHER row in 9×9 matrix (CONSUMES → RIEM/POSEAXIS/FLOOR; consumed by VENN; SUB with TROP)
6. `src/tac/master_gradient.py` + `tools/extract_master_gradient.py` (the upstream OP-AUDIT-1 dependency; Fisher diagonal computed from master-gradient fp64 sensitivity)
7. `src/tac/probe_outcomes_ledger.py` (sister canonical 4-layer helper pattern per Catalog #313)
8. `src/tac/deploy/modal/call_id_ledger.py` (sister canonical 4-layer helper pattern per Catalog #245)

## STRATEGIC CONTEXT

The Phase 1 Fisher-precondition design memo specifies a canonical helper that:
- Computes the empirical Fisher diagonal F_ii from per-pair master-gradient fp64 sensitivity (OP-AUDIT-1 dependency)
- Classifies the Fisher conditioning verdict: `validated_well_conditioned` / `validated_near_singular_requires_kfac` / `invalid_input`
- Anchors on PR101_lc_v2 master-gradient `f174192aeadf...` per Catalog #316 frontier
- Predicts Phase 1 ALONE ΔS `[-0.015, -0.005]`; CASCADE `[-0.060, -0.019]` (parent §11 aggregate matrix; Catalog #287 axis tag `[prediction]`)
- Composes per cross-stack §4: CONSUMES from VENN per cell (Fisher diagonal cell-by-cell); FEEDS RIEM (Riemannian-Newton step direction); FEEDS POSEAXIS (pose-axis Fisher-orthogonal projection); FEEDS FLOOR (Fisher curvature → floor distance)
- C6 IBPS extension diagnoses the 22× miss anchor per `feedback_meta_fix_catalog_324_predicted_band_post_training_validation_required_landed_20260517.md`

This directive routes the full 4-layer canonical helper build to Codex.

## OP-1: Layer 1 — canonical helper module

**Target**: `src/tac/phase_1_fisher_precondition/__init__.py` + sister modules

**Public API** (per design memo §6 canonical interface specification):

```python
# src/tac/phase_1_fisher_precondition/__init__.py
from .fisher_diagonal import compute_fisher_diagonal, FisherDiagonal
from .conditioning import classify_fisher_conditioning_verdict, FisherConditioningVerdict
from .orthogonal_projection import fisher_orthogonal_projection
from .anchor_writer import append_fisher_conditioning_anchor, load_fisher_conditioning_anchors_strict
from .cell_extension import compute_fisher_diagonal_per_venn_cell  # consumes VENN

__all__ = [
    "compute_fisher_diagonal", "FisherDiagonal",
    "classify_fisher_conditioning_verdict", "FisherConditioningVerdict",
    "fisher_orthogonal_projection",
    "append_fisher_conditioning_anchor", "load_fisher_conditioning_anchors_strict",
    "compute_fisher_diagonal_per_venn_cell",
]
```

**Canonical dataclasses + enums** (frozen per design memo §5):

```python
from enum import Enum

class FisherConditioningVerdict(Enum):
    """Per design memo §5 + Catalog #296 Dykstra-feasibility predicted-band check."""
    VALIDATED_WELL_CONDITIONED = "validated_well_conditioned"
    """Fisher diagonal has condition number < 1e8 across all per-pair bytes;
    Newton step direction is numerically stable; Phase 1 PROCEED."""

    VALIDATED_NEAR_SINGULAR_REQUIRES_KFAC = "validated_near_singular_requires_kfac"
    """Fisher diagonal condition number ≥ 1e8 on a subset of cells;
    full Newton requires K-FAC factored approximation per Martens-Grosse;
    Phase 1 PROCEED with K-FAC fallback."""

    INVALID_INPUT = "invalid_input"
    """master-gradient anchor missing OR non-positive Fisher diagonal entries;
    Phase 1 BLOCKED pending OP-AUDIT-1 anchor completion."""


@dataclass(frozen=True)
class FisherDiagonal:
    """fp64 per-byte Fisher diagonal anchored on master-gradient anchor."""
    archive_sha256: str  # canonical 64-char hex
    fisher_values_per_byte: tuple[float, ...]  # one fp64 value per archive byte
    condition_number: float
    captured_at_utc: str
    master_gradient_anchor_id: str  # foreign key to master_gradient_anchors.jsonl
    provenance: dict  # canonical Provenance per Catalog #323
```

**Implementation requirements** (per design memo §6 + §10 9-dim checklist):

- `compute_fisher_diagonal(archive_sha256, *, master_gradient_anchor_path) -> FisherDiagonal` deterministic given master-gradient anchor
- `classify_fisher_conditioning_verdict(fisher_diagonal) -> FisherConditioningVerdict` per design memo §5 thresholds
- `fisher_orthogonal_projection(direction_vector, fisher_diagonal) -> ndarray` returns Fisher-orthogonal component (used by POSEAXIS hook)
- `append_fisher_conditioning_anchor(fisher_diagonal, verdict)` writes to fcntl-locked `.omx/state/fisher_conditioning_anchors.jsonl` per Catalog #131
- `load_fisher_conditioning_anchors_strict(...)` raises `FisherConditioningAnchorCorruptError` on parse failure per Catalog #138

**Canonical-vs-unique decision per layer** (per Catalog #290 + design memo):

| Layer | Decision | Rationale |
|---|---|---|
| master-gradient anchor consumption | ADOPT canonical (`tac.master_gradient.load_anchor_for_archive`) | OP-AUDIT-1 dependency |
| fp64 numerics | ADOPT canonical (numpy float64) | Standard |
| fcntl-locked JSONL persistence | ADOPT canonical (Catalog #131 helper) | Cross-substrate discipline |
| Provenance | ADOPT canonical (Catalog #323) | Cross-substrate discipline |
| Fisher diagonal formula | FORK_BECAUSE_PRINCIPLED_MISMATCH | F_ii = (∂L/∂θ_i)^2 averaged over training samples; unique to Fisher info |
| Conditioning verdict thresholds | FORK_BECAUSE_PRINCIPLED_MISMATCH | Per Amari memorial seat: condition number < 1e8 is hard-earned threshold from natural gradient literature |
| K-FAC fallback | DEFERRED_PENDING_PHASE_2 | Out-of-scope for Phase 1; Riemannian-Newton META consumer |
| Per-Venn-cell extension | ADOPT canonical (`tac.canonical_n_set_venn_classification.classify_archive_per_cell`) | Consumes VENN |

## OP-2: Layer 2 — CLI tool

**Target**: `tools/canonical_phase_1_fisher_precondition_cli.py`

**CLI subcommands**:

```bash
# Compute Fisher diagonal for an archive
.venv/bin/python tools/canonical_phase_1_fisher_precondition_cli.py compute \
    --archive-sha 6bae0201... --master-gradient-anchor .omx/state/master_gradient_anchors.jsonl \
    --output .omx/state/fisher_conditioning_anchors/anchor_<sha[:12]>_<utc>.json

# Classify conditioning verdict
.venv/bin/python tools/canonical_phase_1_fisher_precondition_cli.py classify \
    --archive-sha 6bae0201... --format json

# Audit all anchors per Catalog #305 observability surface
.venv/bin/python tools/canonical_phase_1_fisher_precondition_cli.py audit \
    --summary
```

## OP-3: Layer 3 — STRICT preflight gate

**Catalog #N**: claim transactionally via `.venv/bin/python tools/claim_catalog_number.py claim --commit-via-serializer --reason "FISHER canonical helper STRICT gate"`.

**Gate function**: `check_canonical_phase_1_fisher_precondition_use(strict, verbose) -> list`

**Refuses**: hand-rolled Fisher diagonal computation (`hand_rolled_fisher_diagonal`, `inline_fisher_loop`, `custom_conditioning_threshold`) outside `tac.phase_1_fisher_precondition`. Acceptance: canonical import OR same-line `# CANONICAL_FISHER_BYPASS_OK:<rationale>` waiver.

**Wire-in**: warn-only at landing per Strict-flip atomicity rule.

**Tests**: ≥45 tests covering live-repo regression + positive/negative + waiver semantics + strict-mode + edge cases.

## OP-4: Layer 4 — integration wire-ins per Catalog #125 6-hook

**Hook #1 sensitivity-map**: Fisher diagonal → per-axis weight contribution:
```python
# src/tac/sensitivity_map/fisher_contribution.py
def add_fisher_diagonal_to_axis_weights(axis_weights, fisher_diagonal):
    """High-Fisher cells → high axis weight."""
    ...
```

**Hook #2 Pareto constraint**: Fisher-orthogonal projection extends Pareto-feasibility set:
```python
# src/tac/pareto_fisher_orthogonal_constraint.py
def add_fisher_orthogonal_pareto_constraint(pareto_solver, fisher_diagonal):
    """Fisher-orthogonal step direction restricted to Pareto-feasible polytope."""
    ...
```

**Hook #3 bit-allocator**: Fisher curvature per byte → bit allocation:
```python
# src/tac/bit_allocator/fisher_curvature.py
def allocate_bits_per_fisher_curvature(fisher_diagonal, total_budget_bytes):
    """High-Fisher bytes get tier-1 bits."""
    ...
```

**Hook #4 cathedral autopilot**: extend `tools/cathedral_autopilot_autonomous_loop.py`:
```python
def adjust_predicted_delta_for_fisher_conditioning_verdict(predicted_delta, archive_sha256):
    """VALIDATED_WELL_CONDITIONED → +0.02 reward; VALIDATED_NEAR_SINGULAR → +0.005; INVALID_INPUT → 0 (no adjustment)."""
    ...
```

**Hook #5 continual-learning**: `append_fisher_conditioning_anchor` → `.omx/state/fisher_conditioning_anchors.jsonl` per Catalog #131.

**Hook #6 probe-disambiguator**: per Contrarian's VETO in design memo §12 — `tools/probe_phase_1_fisher_byte_mutation_smoke.py` proves Fisher-orthogonal null-space property empirically:
```bash
# Byte-mutation smoke: mutate a Fisher-null-space byte; verify score unchanged
.venv/bin/python tools/probe_phase_1_fisher_byte_mutation_smoke.py \
    --archive-sha 6bae0201... --null-space-byte-index 1234
```

## OP-5: Operator-facing audit tool per Catalog #305

**Target**: `tools/audit_canonical_phase_1_fisher_precondition_compliance.py`

Shows per-archive conditioning verdict + condition number distribution + Fisher-orthogonal subspace dimension.

## OP-6: Memory entry

`feedback_canonical_phase_1_fisher_precondition_package_landed_<utc>.md` with all 6 Catalog #294 / #303 / #305 / #296 / #290 / #125 sections.

## OP-7: canonical_task_status row emission

Emit 7 task rows (OP_1 through OP_7) per closure verifier dependency.

## C6 IBPS extension (per design memo §8)

The Fisher helper extends to diagnose the C6 IBPS 22× miss anchor:
```python
# Diagnostic call against C6 IBPS Modal smoke archive
fisher = compute_fisher_diagonal(c6_ibps_archive_sha256, ...)
verdict = classify_fisher_conditioning_verdict(fisher)
# Expected: VALIDATED_NEAR_SINGULAR_REQUIRES_KFAC (per design memo predicted)
# Reactivates C6 IBPS Phase 2 with K-FAC fallback per CLAUDE.md "Forbidden premature KILL"
```

## Discipline checklist

Same as VENN directive § "Discipline checklist".

Begin.
