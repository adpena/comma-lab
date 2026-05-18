# Codex routing directive: canonical Riemannian-Newton META-substrate canonical helper package
# Date: 2026-05-18
# Operator: closure campaign master memo `closure_campaign_pursue_and_confirm_master_20260518.md` op-routable OPR-CLOSE-2 batch
# Authority: `.omx/research/riemannian_newton_substrate_engineering_design_memo_20260518.md` (THE design memo; 119.0 KB)
# Sister: closure_campaign_pursue_and_confirm_master_20260518.md §11.2 TOP-5 (OPR-CLOSE-2)
# Per CLAUDE.md "Subagent coherence-by-default" + Catalog #245 4-layer pattern + Catalog #290 + #294 + #303 + #305 + #296

## CANONICAL POINTERS (read FIRST)

1. `/Users/adpena/Projects/pact/CLAUDE.md` (FULL; especially Catalog #125 6-hook + #245 4-layer + #131 fcntl-locked JSONL + #240 recipe-vs-trainer-state)
2. `/Users/adpena/Projects/pact/AGENTS.md`
3. `.omx/research/riemannian_newton_substrate_engineering_design_memo_20260518.md` (THE AUTHORITY; council T2 sextet + MacKay + Hafner + Boyd + Carmack + Hotz)
4. `.omx/research/closure_campaign_pursue_and_confirm_master_20260518.md` (parent coordinator memo)
5. `.omx/research/cross_stack_synthesis_9_design_landings_unified_framework_20260518.md` §4 RIEM row in 9×9 matrix (CONSUMES FISHER/Z8/TT5L2/POSEAXIS/FLOOR; ADD with VENN/CCREZ; SUB with TROP/FISHER)
6. `src/tac/substrates/_shared/trainer_skeleton.py` (the META-substrate inheritance target)
7. `src/tac/phase_1_fisher_precondition/` (sister upstream dependency per FISHER directive)
8. Geomstats library reference (per design memo §6 — Stiefel-manifold primitives)

## STRATEGIC CONTEXT

The Riemannian-Newton META-canonical-helper design memo specifies a meta-substrate inheritance pattern:
- Stiefel-manifold parameterization for substrate weight tensors (per Boumal-Absil-Mahony trust-region per design memo §6)
- Symplectic-EMA flow extending CLAUDE.md "EMA — NON-NEGOTIABLE" canonical 0.997 decay
- Tropical-Newton fallback at SegNet boundaries (composes with TROP per cross-stack §4 SUB relationship)
- Predicts ΔS `[-0.025, -0.008]` per archive; aggregate across 4 archives `[-0.105, -0.034]` under HIGH-orthogonality (Catalog #287 axis tag `[prediction]`)
- META: inherits into per-substrate trainer skeleton; each substrate inherits Riemannian-Newton step direction WITHOUT per-substrate reimplementation
- Phase 2 dependency: Phase 1 Fisher-precondition (per FISHER directive) lands first; RIEM consumes Fisher diagonal for natural-gradient step direction
- Empirical validation target: Z6-v2 Wave 2 full-mode (per CLAUDE.md "PER-SUBSTRATE OPTIMAL FORM via adversarial grand council symposium")

## OP-1: Layer 1 — canonical helper module

**Target**: `src/tac/riemannian_newton_substrate_engineering/__init__.py` + sister modules

**Public API**:

```python
# src/tac/riemannian_newton_substrate_engineering/__init__.py
from .stiefel_manifold import StiefelManifold, retract_to_manifold, parallel_transport
from .symplectic_ema import SymplecticEMA, update_symplectic_shadow
from .newton_step import compute_riemannian_newton_step, RiemannianNewtonStep
from .trust_region import boumal_absil_mahony_trust_region_constrain
from .tropical_fallback import detect_segnet_boundary_zone, tropical_newton_fallback
from .anchor_writer import append_riemannian_newton_convergence_anchor, load_convergence_anchors_strict
from .meta_substrate import RiemannianNewtonMetaSubstrateMixin  # mixin for trainer_skeleton

__all__ = [
    "StiefelManifold", "retract_to_manifold", "parallel_transport",
    "SymplecticEMA", "update_symplectic_shadow",
    "compute_riemannian_newton_step", "RiemannianNewtonStep",
    "boumal_absil_mahony_trust_region_constrain",
    "detect_segnet_boundary_zone", "tropical_newton_fallback",
    "append_riemannian_newton_convergence_anchor", "load_convergence_anchors_strict",
    "RiemannianNewtonMetaSubstrateMixin",
]
```

**Canonical dataclasses + enums**:

```python
from enum import Enum

class RiemannianNewtonConvergenceVerdict(Enum):
    CONVERGED_NEWTON = "converged_newton"
    """Pure Riemannian-Newton step converged in <100 iterations on Stiefel manifold."""

    CONVERGED_TROPICAL_FALLBACK = "converged_tropical_fallback"
    """Hit SegNet boundary zone; tropical-Newton fallback converged.
    Sister of TROP Phase 2."""

    DIVERGED_REQUIRES_KFAC = "diverged_requires_kfac"
    """Hessian near-singular; requires K-FAC factored approximation
    (per FISHER directive Phase 2 K-FAC fallback)."""

    INVALID_INPUT = "invalid_input"
    """Fisher diagonal anchor missing OR substrate not yet in Stiefel parameterization."""


@dataclass(frozen=True)
class RiemannianNewtonStep:
    direction: tuple[float, ...]  # tangent space step direction
    step_magnitude: float
    is_tropical_fallback: bool
    convergence_iterations: int
    archive_sha256: str  # canonical 64-char hex per Catalog #316
    captured_at_utc: str
    fisher_anchor_id: str  # foreign key per FISHER directive
    provenance: dict  # canonical Provenance per Catalog #323


class RiemannianNewtonMetaSubstrateMixin:
    """Mixin for tac.substrates._shared.trainer_skeleton.

    Each substrate inherits to get Riemannian-Newton step direction
    WITHOUT per-substrate reimplementation per CLAUDE.md
    'UNIQUE-AND-COMPLETE-PER-METHOD operating mode' canonical-helper-when-serves cascade.
    """

    def riemannian_newton_step(self, fisher_diagonal, current_weights):
        """Compute Stiefel-manifold step direction."""
        ...

    def symplectic_ema_update(self, current_weights, shadow_weights):
        """Update EMA shadow on manifold."""
        ...
```

**Implementation requirements** (per design memo §6):

- `StiefelManifold(dim_n, dim_p)` parameterization
- `retract_to_manifold(W)` projects R^(n×p) onto Stiefel
- `parallel_transport(v, from_W, to_W)` tangent vector transport
- `compute_riemannian_newton_step(loss_fn, current_weights, fisher_diagonal) -> RiemannianNewtonStep`
- `boumal_absil_mahony_trust_region_constrain(step, manifold)` enforces convex feasibility
- `detect_segnet_boundary_zone(weights) -> bool` heuristic for tropical fallback
- `tropical_newton_fallback(weights, loss_fn) -> RiemannianNewtonStep` consumes TROP package
- `append_riemannian_newton_convergence_anchor(step, verdict)` writes to fcntl-locked `.omx/state/riemannian_newton_convergence.jsonl` per Catalog #131

**Canonical-vs-unique decision per layer** (per Catalog #290):

| Layer | Decision | Rationale |
|---|---|---|
| Stiefel-manifold primitives | ADOPT canonical (Geomstats library) | Per Boumal-Absil-Mahony reference impl |
| Fisher diagonal | ADOPT canonical (`tac.phase_1_fisher_precondition`) | Upstream FISHER dependency |
| Symplectic-EMA | FORK_BECAUSE_PRINCIPLED_MISMATCH | Standard EMA is geometric; symplectic-EMA preserves Stiefel volume form per Hafner DreamerV3 insight |
| Trust-region constraint | ADOPT canonical (Boumal-Absil-Mahony scheme) | Standard convex optimization |
| Tropical fallback | ADOPT canonical (`tac.tropical_d_seg_solver`) | TROP Phase 2 sister |
| fcntl-locked JSONL | ADOPT canonical (Catalog #131) | Cross-substrate discipline |
| META-substrate inheritance | FORK_BECAUSE_PRINCIPLED_MISMATCH | Mixin pattern is unique to this design — no canonical inheritance contract exists |

## OP-2: Layer 2 — CLI tool

**Target**: `tools/canonical_riemannian_newton_meta_substrate_cli.py`

```bash
# Compute Riemannian-Newton step for an archive
.venv/bin/python tools/canonical_riemannian_newton_meta_substrate_cli.py compute-step \
    --archive-sha 6bae0201... --fisher-anchor .omx/state/fisher_conditioning_anchors.jsonl \
    --output .omx/state/riemannian_newton_convergence.jsonl

# Audit convergence verdicts
.venv/bin/python tools/canonical_riemannian_newton_meta_substrate_cli.py audit --summary
```

## OP-3: Layer 3 — STRICT preflight gate

**Catalog #N**: claim transactionally.

**Gate function**: `check_canonical_riemannian_newton_meta_substrate_use(strict, verbose) -> list`

**Refuses**: hand-rolled Newton step (`hand_rolled_newton_step`, `inline_stiefel_retract`, `custom_trust_region`) outside `tac.riemannian_newton_substrate_engineering`. Acceptance: canonical import OR same-line `# CANONICAL_RIEMANNIAN_NEWTON_BYPASS_OK:<rationale>` waiver.

**Tests**: ≥50 tests including Geomstats integration tests + Stiefel-manifold retract round-trip + symplectic-EMA volume preservation + Boumal-Absil-Mahony trust-region convergence + tropical fallback handoff to TROP package.

## OP-4: Layer 4 — integration wire-ins per Catalog #125 6-hook

**Hook #1 sensitivity-map**: Riemannian metric on Stiefel manifold → axis weight:
```python
# src/tac/sensitivity_map/riemannian_metric_contribution.py
def add_riemannian_metric_to_axis_weights(axis_weights, stiefel_metric):
    """Manifold curvature → axis weight reweighting."""
    ...
```

**Hook #2 Pareto constraint**: Boumal-Absil-Mahony trust-region adds Pareto constraint:
```python
def add_boumal_absil_mahony_pareto_constraint(pareto_solver, manifold):
    """Trust-region radius constrains Pareto-feasibility set."""
    ...
```

**Hook #3 bit-allocator**: N/A_WITH_RATIONALE — canonical helper consumed by per-substrate trainer, not bit-allocator. Per Catalog #125 explicit rationale: "META-substrate inheritance pattern operates at trainer skeleton layer, NOT at archive bit allocation layer."

**Hook #4 cathedral autopilot**: extend `tools/cathedral_autopilot_autonomous_loop.py`:
```python
def adjust_predicted_delta_for_riemannian_newton_convergence_verdict(predicted_delta, archive_sha256):
    """CONVERGED_NEWTON → +0.025 reward; CONVERGED_TROPICAL_FALLBACK → +0.010; DIVERGED → 0."""
    ...
```

**Hook #5 continual-learning**: `append_riemannian_newton_convergence_anchor` → `.omx/state/riemannian_newton_convergence.jsonl`.

**Hook #6 probe-disambiguator**: `tools/probe_tropical_vs_riemannian_newton_disambiguator.py` — same probe as TROP directive Hook #6; the probe disambiguates which family fits an archive.

## OP-5: Operator-facing audit tool per Catalog #305

**Target**: `tools/audit_canonical_riemannian_newton_meta_substrate_compliance.py`

## OP-6: Memory entry per Catalog #294 / #303 / #305 / #296 / #290 / #125

## OP-7: canonical_task_status row emission

## META-substrate inheritance wiring (per design memo §7)

Trainer skeleton extension:
```python
# src/tac/substrates/_shared/trainer_skeleton.py
class BaseSubstrateTrainer(RiemannianNewtonMetaSubstrateMixin):
    """All substrate trainers inherit; each gets Riemannian-Newton step direction for free."""
    ...
```

Per CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD" canonical-helper-when-serves cascade: substrate inheriting the mixin GETS Riemannian-Newton; substrate that explicitly opts out via `# RIEMANNIAN_NEWTON_META_DISABLED_OK:<rationale>` keeps Euclidean step direction.

## Empirical validation target

Z6-v2 Wave 2 full-mode anchor per CLAUDE.md "PER-SUBSTRATE OPTIMAL FORM" Catalog #325 6-step contract. Predicted Phase 2 ΔS `[-0.025, -0.008]` on Z6-v2 archive validates Riemannian-Newton meta-substrate inheritance benefits.

## Discipline checklist

Same as sister directives.

Begin.
