# SPDX-License-Identifier: MIT
"""ddm_lp2 P3 — the (e1) SOLVE-SEEDED-BIRTHS harness (BUILD-ONLY; does NOT run).

The convocation's derive-original rung-2 component (gc12 §3 rung-2 / §6 P3). For each still-erased
super-nucleus Lane component at the rung-1 endpoint:

    1. extract the COVERING tokens (the local token neighborhood over the component's pixel support),
    2. run a BOUNDED LOCAL token solve (GN/CG on covering tokens ONLY, margin objective through R) via
       an INJECTED solver — the real fd1/j2 providers are
       ``ddm_family_d_gn_description.FamilyDGaussNewtonEngineV1.propose`` (fd1: damped GN/CG on the
       measured secant HVP, active-set restricted to the covering tokens) over the
       ``direct_description_joint_descent.DirectDescriptionJointDescentMLXModule.loss_and_grad`` (j2:
       exact CE+hinge margin), and/or ``coupled_margin_levelset.solve_active_set_kkt`` (the margin-
       through-R active-set KKT engine). REUSE — do not fork the GN/CG/KKT math,
    3. record the per-component solve RESIDUAL = the local-token REACHABILITY measurement (rg3
       zero-support analog; gc12 (e1) PLP),
    4. verify the seeded init (#208 rare-class-protected + #532 rendered-init verification) BEFORE
       marking the component for the reconcile tail,
    5. carry the PREREGISTERED FALSIFIER: seeded-birth survival < 50% of seeded ΔS through the
       reconcile window ⇒ (e1) CLOSES at formulation scope (local-solve-seed on this vehicle).

BUILD-ONLY: the scorer (``HardOracle``) and the GN/QP solver (``LocalTokenSolver``) are INJECTED
(b2b stub pattern), so this module is scorer-free and does NOT run a real solve or a real reconcile
tail. MAIN wires the real fd1/j2 solver + real scorer + real endpoint components at fire time. This
harness stays ON the render manifold (fp1's receiver tax does not apply — it re-renders from seeds,
does not composite a flat field), SPENDS token bytes (pa1r-favorable direction), and is NOT nv1's
null-snap (opposite sign; scope-checked).

Pointer 0.1910828242 [contest-CPU] UNMOVED. Apparatus — no score claim.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

import numpy as np

SCHEMA = "ddm_lp2_e1_seeding_harness.v1"
CLASS_NAME = "Lane"
LANE_CLASS_INDEX = 1  # comma10k canonical order [Road, Lane, Undrivable, Movable, MyCar]

# PREREGISTERED FALSIFIER threshold (ANCHOR: gc12 §3 rung-2 (e1) — "survival < ~50% of seeded ΔS at
# tail end -> seeding closes at formulation scope"). Not a tuned knob; the convocation's committed line.
SURVIVAL_FALSIFIER_FRACTION = 0.50

# #315 nucleus law: only super-nucleus (>5px) components are seed targets (sub-nucleus = GT flicker).
DEFAULT_NUCLEUS_THRESHOLD_PX = 5


class E1SeedingError(ValueError):
    """Raised (fail-closed) on malformed component / grid / verification input."""


# --- injected-dependency Protocols (the b2b stub seam) --------------------------------------------


@dataclass(frozen=True)
class LocalSeedSolveResult:
    """Output of one bounded local token solve over a component's covering tokens."""

    component_id: int
    status: str  # "SEED_ACCEPTED" | "STALLED_UNREACHABLE" | "REJECTED"
    token_indices: tuple[int, ...]
    token_deltas: tuple[float, ...]
    solve_residual: float  # remaining margin debt through R after the local solve = reachability
    baseline_key: float  # scorer key (e.g. flip count) before the seed
    candidate_key: float  # scorer key after the seed
    reachable: bool  # candidate_key strictly improved over baseline_key


class HardOracle(Protocol):
    """Scorer-metric callback: token deltas -> a scalar admission key (lower = better; e.g. flips).

    Real provider = the frozen CPU-torch SegNet through R (authority; NEVER MPS), matching
    ``boundary_coordinate_joint_solve.solve_joint_boundary_candidate``'s
    ``hard_oracle: Callable[[np.ndarray], HardOracleEvaluation]`` seam.
    """

    def __call__(self, token_deltas: np.ndarray) -> float: ...


class LocalTokenSolver(Protocol):
    """Bounded local GN/QP token solve over a component's covering tokens.

    Real provider = ``solve_joint_boundary_candidate`` (corrected active-set QP + exact uint8
    realization + hard admission) restricted to the covering token columns, OR the fd2 QDBS terminal
    solver. REUSE — do not fork the GN/QP math.
    """

    def __call__(
        self,
        *,
        component_id: int,
        token_indices: np.ndarray,
        hard_oracle: HardOracle,
        max_iterations: int,
    ) -> LocalSeedSolveResult: ...


# --- data model -----------------------------------------------------------------------------------


@dataclass(frozen=True)
class ErasedComponent:
    """A still-erased super-nucleus Lane component at the rung-1 endpoint (MAIN provides from the
    endpoint realized argmax vs gt_n600 — one scorer pass; QA91 method_note)."""

    component_id: int
    size_px: int
    pixel_support: tuple[tuple[int, int], ...]  # (row, col) integer pixels of the component's support
    class_index: int = LANE_CLASS_INDEX

    def is_super_nucleus(self, threshold_px: int = DEFAULT_NUCLEUS_THRESHOLD_PX) -> bool:
        return self.size_px > threshold_px


@dataclass(frozen=True)
class TokenGridGeometry:
    """The token grid <-> image pixel geometry for covering-token extraction."""

    image_height: int
    image_width: int
    token_rows: int
    token_cols: int

    @property
    def n_tokens(self) -> int:
        return self.token_rows * self.token_cols

    def pixel_to_token(self, row: int, col: int) -> int:
        if not (0 <= row < self.image_height and 0 <= col < self.image_width):
            raise E1SeedingError(f"pixel ({row},{col}) outside image {self.image_height}x{self.image_width}")
        tr = (row * self.token_rows) // self.image_height
        tc = (col * self.token_cols) // self.image_width
        return tr * self.token_cols + tc


@dataclass(frozen=True)
class CoveringTokenSet:
    component_id: int
    token_indices: tuple[int, ...]


@dataclass(frozen=True)
class SeedInitVerification:
    """#208 rare-class-protected + #532 rendered-init verification of the SEEDED init."""

    lane_channel_live: bool  # #208: Lane head channel must not be dead after seeding
    dead_classes: tuple[int, ...]  # #532: any class whose rendered mass collapsed to ~0
    per_class_mass: tuple[float, ...]
    gt_class_priors: tuple[float, ...]
    max_relative_mass_error: float
    passed: bool


@dataclass(frozen=True)
class SurvivalVerdict:
    """The preregistered (e1) falsifier evaluated AFTER a real reconcile tail (MAIN runs the tail)."""

    seeded_delta_s: float
    survived_delta_s: float
    survival_fraction: float
    threshold_fraction: float
    e1_closes_at_formulation_scope: bool  # True => falsifier FIRED, (e1) closes


@dataclass(frozen=True)
class E1SeedingReport:
    schema: str
    class_name: str
    n_components_in: int
    n_super_nucleus: int
    n_seed_accepted: int
    n_stalled_unreachable: int
    n_rejected: int
    reachable_fraction: float  # rg3-analog: fraction of super-nucleus components locally reachable
    solve_results: tuple[LocalSeedSolveResult, ...]
    init_verification: SeedInitVerification | None
    marked_for_reconcile: tuple[int, ...]  # component ids seeded AND init-verified
    survival_falsifier_fraction: float
    provenance: dict[str, str]
    score_claim: bool = False
    evidence_axis: str = "[apparatus — build-only harness, injected stub solver/scorer]"
    pointer: str = "0.1910828242 [contest-CPU] UNMOVED"
    caveats: tuple[str, ...] = field(default_factory=tuple)


_PROVENANCE: dict[str, str] = {
    "solve_residual": "DERIVED (remaining margin debt through R after local solve = token reachability, rg3-analog)",
    "reachable_fraction": "DERIVED (super-nucleus components whose local solve strictly improved the scorer key)",
    "survival_falsifier_fraction": (
        "ANCHOR: gc12 §3 rung-2 (e1) preregistered falsifier — survival <50% of seeded ΔS ⇒ (e1) closes "
        "at formulation scope (local-solve-seed on this vehicle)"
    ),
    "init_verification": "#208 rare-class-protected (Lane channel live) + #532 rendered-init verification",
    "solver": (
        "INJECTED (real fd1/j2 = ddm_family_d_gn_description.FamilyDGaussNewtonEngineV1.propose over "
        "direct_description_joint_descent.DirectDescriptionJointDescentMLXModule.loss_and_grad, and/or "
        "coupled_margin_levelset.solve_active_set_kkt; REUSE, active-set restricted to covering tokens)"
    ),
    "hard_oracle": "INJECTED (real = frozen CPU-torch SegNet through R; authority, NEVER MPS)",
    "lane_class_index": "comma10k canonical order -> Lane=1 (NOT luma sort; CLAUDE.md class-order law)",
}


# --- covering-token extraction (real, deterministic geometry) -------------------------------------


def extract_covering_tokens(
    component: ErasedComponent,
    geometry: TokenGridGeometry,
    *,
    dilation_cells: int = 0,
) -> CoveringTokenSet:
    """Map a component's pixel support to its covering token cells (+ optional 1-ring dilation).

    LOCAL by construction: only the token cells the component touches (plus ``dilation_cells`` rings)
    enter the solve — the bounded-neighborhood requirement (gc12 §3 (e1)).
    """

    if not component.pixel_support:
        raise E1SeedingError(f"component {component.component_id} has empty pixel_support")
    if dilation_cells < 0:
        raise E1SeedingError("dilation_cells must be >= 0")
    base_cells: set[tuple[int, int]] = set()
    for row, col in component.pixel_support:
        tok = geometry.pixel_to_token(int(row), int(col))
        base_cells.add((tok // geometry.token_cols, tok % geometry.token_cols))
    cells = set(base_cells)
    for _ in range(dilation_cells):
        ring: set[tuple[int, int]] = set()
        for tr, tc in cells:
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    nr, nc = tr + dr, tc + dc
                    if 0 <= nr < geometry.token_rows and 0 <= nc < geometry.token_cols:
                        ring.add((nr, nc))
        cells |= ring
    indices = sorted(tr * geometry.token_cols + tc for tr, tc in cells)
    return CoveringTokenSet(component_id=component.component_id, token_indices=tuple(indices))


# --- init verification (#208 / #532) --------------------------------------------------------------


def verify_seed_init(
    per_class_mass: Sequence[float],
    gt_class_priors: Sequence[float],
    *,
    lane_class_index: int = LANE_CLASS_INDEX,
    dead_mass_eps: float = 1e-6,
    max_relative_mass_error: float = 5.0,
) -> SeedInitVerification:
    """#208 rare-class-protected + #532 rendered-init verification of the SEEDED init.

    Renders the seeded init's per-class mass fractions and checks (a) the Lane channel is LIVE (#208:
    a dead Lane channel kills the birthed components — the exact fp1 catch), (b) no class collapsed to
    ~0 (#532), (c) per-class mass is within a relative band of the GT priors. Fail-closed on shape.
    """

    mass = np.asarray(per_class_mass, dtype=np.float64)
    priors = np.asarray(gt_class_priors, dtype=np.float64)
    if mass.shape != priors.shape or mass.ndim != 1:
        raise E1SeedingError("per_class_mass and gt_class_priors must be equal-length 1-D vectors")
    if not (0 <= lane_class_index < mass.size):
        raise E1SeedingError(f"lane_class_index {lane_class_index} out of range {mass.size}")
    dead = tuple(int(i) for i in np.where(mass <= dead_mass_eps)[0])
    lane_live = bool(mass[lane_class_index] > dead_mass_eps)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel_err = np.where(priors > 0, np.abs(mass - priors) / priors, 0.0)
    max_rel = float(rel_err.max()) if rel_err.size else 0.0
    passed = bool(lane_live and not dead and max_rel <= max_relative_mass_error)
    return SeedInitVerification(
        lane_channel_live=lane_live,
        dead_classes=dead,
        per_class_mass=tuple(float(m) for m in mass),
        gt_class_priors=tuple(float(p) for p in priors),
        max_relative_mass_error=max_rel,
        passed=passed,
    )


# --- the harness (build-only; injected solver/oracle) ---------------------------------------------


def run_local_seed_solves(
    components: Sequence[ErasedComponent],
    geometry: TokenGridGeometry,
    solver: LocalTokenSolver,
    hard_oracle: HardOracle,
    *,
    nucleus_threshold_px: int = DEFAULT_NUCLEUS_THRESHOLD_PX,
    dilation_cells: int = 1,
    max_iterations: int = 128,
) -> list[LocalSeedSolveResult]:
    """Run one bounded local solve per SUPER-NUCLEUS component (sub-nucleus skipped per #315)."""

    results: list[LocalSeedSolveResult] = []
    for comp in components:
        if not comp.is_super_nucleus(nucleus_threshold_px):
            continue  # sub-nucleus = GT flicker; not a seed target
        covering = extract_covering_tokens(comp, geometry, dilation_cells=dilation_cells)
        if not covering.token_indices:
            raise E1SeedingError(f"component {comp.component_id} produced no covering tokens")
        res = solver(
            component_id=comp.component_id,
            token_indices=np.asarray(covering.token_indices, dtype=np.int64),
            hard_oracle=hard_oracle,
            max_iterations=max_iterations,
        )
        results.append(res)
    return results


def assemble_seeding_report(
    components: Sequence[ErasedComponent],
    solve_results: Sequence[LocalSeedSolveResult],
    init_verification: SeedInitVerification | None,
    *,
    nucleus_threshold_px: int = DEFAULT_NUCLEUS_THRESHOLD_PX,
) -> E1SeedingReport:
    """Aggregate solve results into the typed report; only init-verified accepted seeds are marked."""

    n_super = sum(1 for c in components if c.is_super_nucleus(nucleus_threshold_px))
    accepted = [r for r in solve_results if r.status == "SEED_ACCEPTED"]
    stalled = [r for r in solve_results if r.status == "STALLED_UNREACHABLE"]
    rejected = [r for r in solve_results if r.status == "REJECTED"]
    reachable = [r for r in solve_results if r.reachable]
    reachable_fraction = (len(reachable) / n_super) if n_super else 0.0
    init_ok = init_verification is not None and init_verification.passed
    marked = tuple(r.component_id for r in accepted) if init_ok else ()
    caveats: list[str] = []
    if init_verification is None:
        caveats.append("init_verification NOT run — no components marked for reconcile (fail-closed)")
    elif not init_verification.passed:
        caveats.append(
            "#208/#532 init verification FAILED (dead/collapsed class) — 0 marked; abort per fp1 catch"
        )
    caveats.append("BUILD-ONLY: no real solve/reconcile ran; solver+oracle are stubs (MAIN wires real fd1/j2 + SegNet)")
    return E1SeedingReport(
        schema=SCHEMA,
        class_name=CLASS_NAME,
        n_components_in=len(components),
        n_super_nucleus=n_super,
        n_seed_accepted=len(accepted),
        n_stalled_unreachable=len(stalled),
        n_rejected=len(rejected),
        reachable_fraction=reachable_fraction,
        solve_results=tuple(solve_results),
        init_verification=init_verification,
        marked_for_reconcile=marked,
        survival_falsifier_fraction=SURVIVAL_FALSIFIER_FRACTION,
        provenance=dict(_PROVENANCE),
        caveats=tuple(caveats),
    )


def evaluate_survival_falsifier(
    seeded_delta_s: float,
    survived_delta_s: float,
    *,
    threshold_fraction: float = SURVIVAL_FALSIFIER_FRACTION,
) -> SurvivalVerdict:
    """The preregistered (e1) falsifier (MAIN calls this AFTER a real reconcile tail runs).

    survival_fraction = survived ΔS / seeded ΔS. If < threshold ⇒ the erasure force re-erased the
    seeds ⇒ (e1) CLOSES at formulation scope (local-solve-seed on this vehicle).
    """

    if not (0.0 < threshold_fraction < 1.0):
        raise E1SeedingError("threshold_fraction must be in (0, 1)")
    if seeded_delta_s <= 0.0:
        raise E1SeedingError("seeded_delta_s must be positive (there must be a seed to survive)")
    survival_fraction = survived_delta_s / seeded_delta_s
    closes = bool(survival_fraction < threshold_fraction)
    return SurvivalVerdict(
        seeded_delta_s=seeded_delta_s,
        survived_delta_s=survived_delta_s,
        survival_fraction=survival_fraction,
        threshold_fraction=threshold_fraction,
        e1_closes_at_formulation_scope=closes,
    )


def report_to_row(report: E1SeedingReport) -> dict[str, Any]:
    return asdict(report)


# --- reference stubs (for tests + as the b2b wiring template MAIN replaces) ------------------------


@dataclass(frozen=True)
class StubLocalTokenSolver:
    """Deterministic stub solver: seeds accept iff the covering-token count is within a band.

    Purely for plumbing tests + as the interface template MAIN swaps for the real fd1/j2 solver.
    """

    accept_min_tokens: int = 1
    accept_max_tokens: int = 10_000
    seed_delta: float = 0.5

    def __call__(
        self,
        *,
        component_id: int,
        token_indices: np.ndarray,
        hard_oracle: HardOracle,
        max_iterations: int,
    ) -> LocalSeedSolveResult:
        n = int(token_indices.size)
        deltas = np.full(n, self.seed_delta, dtype=np.float64)
        baseline_key = float(hard_oracle(np.zeros(n, dtype=np.float64)))
        candidate_key = float(hard_oracle(deltas))
        reachable = candidate_key < baseline_key
        if self.accept_min_tokens <= n <= self.accept_max_tokens and reachable:
            status = "SEED_ACCEPTED"
        elif reachable:
            status = "REJECTED"
        else:
            status = "STALLED_UNREACHABLE"
        return LocalSeedSolveResult(
            component_id=component_id,
            status=status,
            token_indices=tuple(int(i) for i in token_indices.tolist()),
            token_deltas=tuple(float(d) for d in deltas.tolist()),
            solve_residual=max(0.0, candidate_key),
            baseline_key=baseline_key,
            candidate_key=candidate_key,
            reachable=reachable,
        )


def stub_hard_oracle(token_deltas: np.ndarray) -> float:
    """Deterministic stub scorer: key decreases as the seed magnitude grows (a paintable component)."""

    return float(100.0 - np.abs(np.asarray(token_deltas, dtype=np.float64)).sum())
