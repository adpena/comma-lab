# Codex routing directive: canonical Tropical d_seg solver helper package
# Date: 2026-05-18
# Operator: closure campaign master memo `closure_campaign_pursue_and_confirm_master_20260518.md` op-routable OPR-CLOSE-2 batch
# Authority: `.omx/research/tropical_d_seg_solver_design_memo_20260518.md` (THE design memo; 117.4 KB)
# Sister: closure_campaign_pursue_and_confirm_master_20260518.md §11.2 TOP-5 (OPR-CLOSE-2)
# Per CLAUDE.md "Subagent coherence-by-default" + Catalog #245 4-layer pattern + Catalog #290 + #294 + #303 + #305 + #296

## CANONICAL POINTERS (read FIRST)

1. `/Users/adpena/Projects/pact/CLAUDE.md`
2. `/Users/adpena/Projects/pact/AGENTS.md`
3. `.omx/research/tropical_d_seg_solver_design_memo_20260518.md` (THE AUTHORITY; council T2 sextet + Maragos + Akian + Boyd + Mallat + van_den_Oord + Filler)
4. `.omx/research/closure_campaign_pursue_and_confirm_master_20260518.md` (parent coordinator memo)
5. `.omx/research/cross_stack_synthesis_9_design_landings_unified_framework_20260518.md` §4 TROP row in 9×9 matrix (CONSUMES POSEAXIS/FLOOR; ADD with VENN/CCREZ; SUB with FISHER/RIEM; ORTHO with Z8/TT5L2)

## STRATEGIC CONTEXT

The Tropical d_seg solver design memo specifies a canonical helper that:
- Detects per-pixel SegNet class boundaries via Mallat wavelet-multiscale boundary detection (per Mallat seat)
- Computes tropical polynomial faithfulness probe (per Maragos + Akian seats)
- Predicts Phase 1 ALONE ΔS `[-0.010, -0.002]`; CASCADE Phase 1-4 aggregate `[-0.040, -0.012]` (Catalog #287 axis tag `[prediction]`)
- Phases: Phase 1 (boundary detection) / Phase 2 (faithfulness probe) / Phase 3 (composition) / Phase 4 (RIEM tropical-fallback integration)
- Composes per cross-stack §4: CONSUMES from VENN per cell; FEEDS RIEM tropical-fallback; FEEDS POSEAXIS pose-axis boundary detection; FEEDS FLOOR boundary-distance floor metric
- Phase 2-3 require paired smoke (~$5-10) per design memo cost envelope

## OP-1: Layer 1 — canonical helper module

**Target**: `src/tac/tropical_d_seg_solver/__init__.py` + sister modules

**Public API**:

```python
# src/tac/tropical_d_seg_solver/__init__.py
from .boundary_detector import detect_segnet_boundary_per_pixel, BoundaryMap
from .mallat_wavelet import compute_mallat_multiscale_boundary, MallatBoundaryProfile
from .tropical_polynomial import TropicalPolynomial, tropical_polynomial_degree
from .faithfulness_probe import probe_tropical_faithfulness, FaithfulnessVerdict
from .anchor_writer import append_tropical_d_seg_anchor, load_tropical_d_seg_anchors_strict
from .phase_dispatcher import dispatch_phase_1_boundary_only, dispatch_phase_2_faithfulness, dispatch_phase_3_composition, dispatch_phase_4_riem_fallback

__all__ = [
    "detect_segnet_boundary_per_pixel", "BoundaryMap",
    "compute_mallat_multiscale_boundary", "MallatBoundaryProfile",
    "TropicalPolynomial", "tropical_polynomial_degree",
    "probe_tropical_faithfulness", "FaithfulnessVerdict",
    "append_tropical_d_seg_anchor", "load_tropical_d_seg_anchors_strict",
    "dispatch_phase_1_boundary_only", "dispatch_phase_2_faithfulness", "dispatch_phase_3_composition", "dispatch_phase_4_riem_fallback",
]
```

**Canonical dataclasses + enums**:

```python
class FaithfulnessVerdict(Enum):
    FAITHFUL_HIGH_RECALL = "faithful_high_recall"
    """Tropical polynomial recall ≥ ε per design memo §5 threshold; composition-α validated."""

    FAITHFUL_PARTIAL = "faithful_partial"
    """Recall ∈ [ε/2, ε); Phase 3 composition may proceed with reduced confidence."""

    NON_FAITHFUL = "non_faithful"
    """Recall < ε/2; tropical approximation insufficient; fallback to RIEM Phase 2."""

    INVALID_INPUT = "invalid_input"
    """Boundary map missing OR SegNet anchor incompatible."""


@dataclass(frozen=True)
class BoundaryMap:
    """Per-pixel SegNet class boundary map per design memo §5."""
    archive_sha256: str
    boundary_per_pixel: tuple[tuple[int, ...], ...]  # 2D (H × W) of class IDs
    boundary_density: float  # fraction of pixels at class boundary
    captured_at_utc: str
    mallat_anchor_id: str  # foreign key to Mallat multiscale profile
    provenance: dict


@dataclass(frozen=True)
class TropicalPolynomial:
    coefficients: tuple[float, ...]  # tropical (max-plus) coefficients
    monomials: tuple[tuple[int, ...], ...]  # exponent tuples
    degree: int  # max total exponent across monomials
```

**Implementation requirements** (per design memo §6):

- `detect_segnet_boundary_per_pixel(archive_sha256, *, scorer_archive) -> BoundaryMap`
- `compute_mallat_multiscale_boundary(boundary_map, *, num_scales=5) -> MallatBoundaryProfile`
- `tropical_polynomial_degree(p: TropicalPolynomial) -> int`
- `probe_tropical_faithfulness(p, boundary_map, *, epsilon=0.05) -> FaithfulnessVerdict`
- `dispatch_phase_<N>` functions sequencing through 4 phases
- `append_tropical_d_seg_anchor` writes fcntl-locked JSONL per Catalog #131

**Canonical-vs-unique decision per layer** (per Catalog #290):

| Layer | Decision | Rationale |
|---|---|---|
| SegNet anchor consumption | ADOPT canonical (`tac.master_gradient` SegNet variant) | Standard contract |
| Mallat wavelet primitives | ADOPT canonical (PyWavelets) | Standard library |
| Tropical polynomial primitives | FORK_BECAUSE_PRINCIPLED_MISMATCH | Tropical algebra is unique to this design; no canonical impl exists; Maragos + Akian seats validate |
| Faithfulness probe formula | FORK_BECAUSE_PRINCIPLED_MISMATCH | Per Mallat seat: recall threshold per design memo §5 |
| Phase dispatcher | FORK_BECAUSE_PRINCIPLED_MISMATCH | 4-phase sequencing is design-memo-specific |
| fcntl-locked JSONL | ADOPT canonical (Catalog #131) | Cross-substrate discipline |

## OP-2: Layer 2 — CLI tool

**Target**: `tools/canonical_tropical_d_seg_solver_cli.py`

```bash
# Phase 1: detect boundary
.venv/bin/python tools/canonical_tropical_d_seg_solver_cli.py phase-1-boundary \
    --archive-sha 6bae0201... --output .omx/state/tropical_d_seg_anchors/...

# Phase 2: faithfulness probe (requires paired smoke ~$5-10)
.venv/bin/python tools/canonical_tropical_d_seg_solver_cli.py phase-2-faithfulness \
    --archive-sha 6bae0201... --paired-smoke-anchor ...

# Audit all phase anchors
.venv/bin/python tools/canonical_tropical_d_seg_solver_cli.py audit --summary
```

## OP-3: Layer 3 — STRICT preflight gate

**Catalog #N**: claim transactionally.

**Gate function**: `check_canonical_tropical_d_seg_solver_use(strict, verbose) -> list`

**Refuses**: hand-rolled tropical operations (`hand_rolled_tropical_polynomial`, `inline_max_plus_op`, `custom_boundary_detector`) outside `tac.tropical_d_seg_solver`. Acceptance: canonical import OR same-line `# CANONICAL_TROPICAL_BYPASS_OK:<rationale>` waiver.

**Tests**: ≥40 tests.

## OP-4: Layer 4 — integration wire-ins per Catalog #125 6-hook

**Hook #1 sensitivity-map**: per-pixel boundary density → axis weight:
```python
def add_per_pixel_boundary_density_to_axis_weights(axis_weights, boundary_map):
    """High-boundary-density pixels → high axis weight."""
    ...
```

**Hook #2 Pareto**: tropical polynomial degree constraint:
```python
def add_tropical_polynomial_degree_pareto_constraint(pareto_solver, poly):
    """Polynomial degree ≤ design-memo-§5-max contributes Pareto constraint."""
    ...
```

**Hook #3 bit-allocator**: per-byte boundary impact:
```python
def allocate_bits_per_byte_boundary_impact(boundary_map, total_budget_bytes):
    """Bytes at boundary get tier-1 allocation."""
    ...
```

**Hook #4 cathedral autopilot**: `adjust_predicted_delta_for_tropical_phase_eligibility` (already declared in design memo §3 Tier-1 #4):
```python
def adjust_predicted_delta_for_tropical_phase_eligibility(predicted_delta, archive_sha256):
    """Phase 1 verdict FAITHFUL_HIGH_RECALL → +0.010 reward; FAITHFUL_PARTIAL → +0.003; NON_FAITHFUL → 0."""
    ...
```

**Hook #5 continual-learning**: `append_tropical_d_seg_anchor` → `.omx/state/tropical_d_seg_anchors.jsonl`.

**Hook #6 probe-disambiguator**: `tools/probe_tropical_vs_riemannian_newton_disambiguator.py` (shared with RIEM Hook #6; canonical disambiguator per cross-stack §4 SUB relationship).

## OP-5: Operator-facing audit tool per Catalog #305

**Target**: `tools/audit_canonical_tropical_d_seg_solver_compliance.py`

## OP-6: Memory entry

## OP-7: canonical_task_status row emission

## Phase 2-3 paired smoke dependency

Per design memo cost envelope: Phase 2 + Phase 3 require ~$5-10 paired Modal/Lightning smoke. Operator-gated via Catalog #325 per-substrate symposium per CLAUDE.md "Substrate MUST be at OPTIMAL FORM before paid empirical dispatch" non-negotiable.

## Discipline checklist

Same as sister directives.

Begin.
