# SPDX-License-Identifier: MIT
"""Lattice-aware trust control for receiver-realized iterative solves.

This module deliberately does not know about a scorer, archive, or contest
objective.  A caller linearizes its measured receiver at the current accepted
integer state, uses the helpers here to create bounded Babai candidates, and
then supplies fresh hard measurements.  Model reduction and hard acceptance
remain separate authorities.

The trust update follows the classical ratio

``rho = (hard_before - hard_after) / (model_before - model_after)``.

Negative ``rho`` always hard-shrinks.  No value of ``rho`` can accept a step;
only a caller-provided admissible hard objective improvement can do that.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from tac.optimization.coupled_margin_levelset import babai_nearest_plane


class IterativeRealizedTrustError(ValueError):
    """Malformed trust policy, model, lattice state, or hard measurement."""


class TrustUpdate(StrEnum):
    HARD_SHRINK_NEGATIVE_RHO = "HARD_SHRINK_NEGATIVE_RHO"
    SHRINK_MODEL_INVALID = "SHRINK_MODEL_INVALID"
    SHRINK_REJECTED = "SHRINK_REJECTED"
    GROW_ACCEPTED_HIGH_RHO = "GROW_ACCEPTED_HIGH_RHO"
    HOLD_ACCEPTED = "HOLD_ACCEPTED"


class TemperingStatus(StrEnum):
    HARD_IMPROVEMENT = "HARD_IMPROVEMENT"
    NO_HARD_IMPROVEMENT = "NO_HARD_IMPROVEMENT"
    N_A_DEGENERATE_ENERGY_SPREAD = "N_A_DEGENERATE_ENERGY_SPREAD"
    N_A_NO_MOVABLE_COORDINATES = "N_A_NO_MOVABLE_COORDINATES"


class TemplateBasis(StrEnum):
    """Counted template subspaces used by the bounded-collateral probe."""

    ROWBAND_1X1_CONTROL = "1x1_rowband_control"
    CONTEXTUAL_2X2 = "2x2_contextual"
    BOUNDARY_NORMAL_2X2 = "boundary_normal_2x2"


@dataclass(frozen=True, slots=True)
class TrustRegionPolicy:
    minimum_radius: float = 0.5
    maximum_radius: float = 16.0
    shrink_factor: float = 0.5
    negative_rho_factor: float = 0.25
    growth_factor: float = 2.0
    growth_rho: float = 0.75
    poor_rho: float = 0.25

    def __post_init__(self) -> None:
        values = (
            self.minimum_radius,
            self.maximum_radius,
            self.shrink_factor,
            self.negative_rho_factor,
            self.growth_factor,
            self.growth_rho,
            self.poor_rho,
        )
        if not all(math.isfinite(float(value)) for value in values):
            raise IterativeRealizedTrustError("trust policy values must be finite")
        if not 0.0 < self.minimum_radius <= self.maximum_radius:
            raise IterativeRealizedTrustError("trust radius bounds are invalid")
        if not 0.0 < self.negative_rho_factor < self.shrink_factor < 1.0:
            raise IterativeRealizedTrustError("trust shrink factors are invalid")
        if self.growth_factor <= 1.0 or not 0.0 <= self.poor_rho < self.growth_rho:
            raise IterativeRealizedTrustError("trust growth thresholds are invalid")


@dataclass(frozen=True, slots=True)
class LatticeCandidate:
    candidate_id: str
    scale: float
    integer_step: np.ndarray
    continuous_step: np.ndarray
    quadratic_error: float
    covering_bound: float
    clipped_coordinate_count: int

    def __post_init__(self) -> None:
        integer = np.asarray(self.integer_step, dtype=np.int64)
        continuous = np.asarray(self.continuous_step, dtype=np.float64)
        if integer.ndim != 1 or continuous.shape != integer.shape:
            raise IterativeRealizedTrustError("candidate step geometry differs")
        if not np.isfinite(continuous).all() or not np.any(integer):
            raise IterativeRealizedTrustError("candidate step must be finite and nonzero")
        object.__setattr__(self, "integer_step", _immutable(integer))
        object.__setattr__(self, "continuous_step", _immutable(continuous))


@dataclass(frozen=True, slots=True)
class HardCandidate:
    candidate_id: str
    hard_objective: float
    d_seg: float
    archive_bytes: int
    admissible: bool
    predicted_reduction: float
    realized_model_reduction: float
    integer_step: np.ndarray

    def __post_init__(self) -> None:
        scalars = (
            self.hard_objective,
            self.d_seg,
            self.predicted_reduction,
            self.realized_model_reduction,
        )
        if not all(math.isfinite(float(value)) for value in scalars):
            raise IterativeRealizedTrustError("hard candidate scalars must be finite")
        if self.archive_bytes < 0:
            raise IterativeRealizedTrustError("archive bytes must be nonnegative")
        step = np.asarray(self.integer_step, dtype=np.int64)
        if step.ndim != 1:
            raise IterativeRealizedTrustError("hard candidate step must be one-dimensional")
        object.__setattr__(self, "integer_step", _immutable(step))


@dataclass(frozen=True, slots=True)
class RealizedSelection:
    selected: HardCandidate | None
    accepted: bool
    baseline_objective: float
    objective_improvement: float
    rho: float | None
    evaluated_count: int
    admissible_count: int


@dataclass(frozen=True, slots=True)
class TrustDecision:
    old_radius: float
    new_radius: float
    update: TrustUpdate
    rho: float | None


@dataclass(frozen=True, slots=True)
class TemperingTerminal:
    replica: int
    state: np.ndarray
    cheap_energy: float
    hard_key: tuple[float, ...]

    def __post_init__(self) -> None:
        state = np.asarray(self.state, dtype=np.int64)
        if state.ndim != 1 or not np.isfinite(float(self.cheap_energy)):
            raise IterativeRealizedTrustError("tempering terminal is malformed")
        object.__setattr__(self, "state", _immutable(state))


@dataclass(frozen=True, slots=True)
class TemperingResult:
    status: TemperingStatus
    temperatures: tuple[float, ...]
    terminals: tuple[TemperingTerminal, ...]
    proposals: int
    cheap_accepts: int
    swaps: int
    selected_replica: int | None


@dataclass(frozen=True, slots=True)
class BasisProjection:
    """Integer-preserving map from a counted latent basis to full receiver DOF."""

    basis: TemplateBasis
    matrix: np.ndarray
    current: np.ndarray
    dof_labels: tuple[str, ...]
    template_latent_count: int
    boundary_axes: tuple[str, ...]

    def __post_init__(self) -> None:
        matrix = np.asarray(self.matrix, dtype=np.float64)
        current = np.asarray(self.current, dtype=np.int64)
        if matrix.ndim != 2 or current.shape != (matrix.shape[1],):
            raise IterativeRealizedTrustError("basis projection geometry differs")
        if len(self.dof_labels) != current.size or not 0 <= self.template_latent_count <= current.size:
            raise IterativeRealizedTrustError("basis projection labels/partition differ")
        if not np.isfinite(matrix).all() or not np.all((matrix == 0.0) | (matrix == 1.0)):
            raise IterativeRealizedTrustError("basis projection must be a finite binary map")
        if np.any(np.sum(matrix, axis=1) != 1.0):
            raise IterativeRealizedTrustError("every full receiver DOF must have one latent owner")
        object.__setattr__(self, "matrix", _immutable(matrix))
        object.__setattr__(self, "current", _immutable(current))

    def lift_step(self, latent_step: np.ndarray) -> np.ndarray:
        """Lift one integer latent step without rounding or information loss."""

        step = np.asarray(latent_step, dtype=np.int64)
        if step.shape != self.current.shape:
            raise IterativeRealizedTrustError("latent step geometry differs from basis")
        lifted = self.matrix @ step
        rounded = np.rint(lifted).astype(np.int64)
        if not np.array_equal(lifted, rounded.astype(np.float64)):
            raise IterativeRealizedTrustError("basis lift did not preserve the integer lattice")
        return rounded


def build_template_basis_projection(
    template_values_u8: np.ndarray,
    compensation_rgb_i8: np.ndarray,
    *,
    basis: TemplateBasis | str,
    boundary_axes: Sequence[str] = (),
) -> BasisProjection:
    """Build the exact counted map for one preregistered template basis.

    The full receiver always consumes ``T x 2 x 2 x RGB`` patches followed by
    sparse RGB compensation.  The row-band control repeats one RGB triple over
    each patch.  The boundary-normal basis owns two RGB triples split across a
    measured horizontal or vertical normal.  The contextual basis owns all
    four patch cells.  All maps are binary, so integer latent steps remain
    exact integer receiver steps.
    """

    values = np.asarray(template_values_u8, dtype=np.int64)
    compensation = np.asarray(compensation_rgb_i8, dtype=np.int64)
    selected = TemplateBasis(basis)
    if values.ndim != 4 or values.shape[1:] != (2, 2, 3):
        raise IterativeRealizedTrustError("template values must have T x 2 x 2 x RGB geometry")
    if compensation.ndim != 2 or compensation.shape[1] != 3:
        raise IterativeRealizedTrustError("compensation must have support x RGB geometry")
    if np.any(values < 0) or np.any(values > 255) or np.any(compensation < -127) or np.any(compensation > 127):
        raise IterativeRealizedTrustError("basis origin is outside counted integer bounds")
    template_count = int(values.shape[0])
    axes = tuple(str(value) for value in boundary_axes)
    if selected is TemplateBasis.BOUNDARY_NORMAL_2X2:
        if len(axes) != template_count or any(value not in {"x", "y"} for value in axes):
            raise IterativeRealizedTrustError("boundary-normal basis needs one x/y axis per template")
    elif axes:
        raise IterativeRealizedTrustError("boundary axes are only valid for the boundary-normal basis")

    full_template_count = int(values.size)
    full_count = full_template_count + int(compensation.size)
    columns: list[np.ndarray] = []
    current: list[int] = []
    labels: list[str] = []

    def add_template_latent(owner_rows: Sequence[int], value: int, label: str) -> None:
        column = np.zeros(full_count, dtype=np.float64)
        column[np.asarray(owner_rows, dtype=np.int64)] = 1.0
        columns.append(column)
        current.append(int(value))
        labels.append(label)

    for template in range(template_count):
        if selected is TemplateBasis.CONTEXTUAL_2X2:
            for patch_y in range(2):
                for patch_x in range(2):
                    for channel in range(3):
                        row = int(np.ravel_multi_index((template, patch_y, patch_x, channel), values.shape))
                        add_template_latent(
                            (row,),
                            int(values[template, patch_y, patch_x, channel]),
                            f"template:{template}:y{patch_y}:x{patch_x}:c{channel}",
                        )
        elif selected is TemplateBasis.ROWBAND_1X1_CONTROL:
            if not np.all(values[template] == values[template, 0, 0]):
                raise IterativeRealizedTrustError("1x1 control origin is outside its equality subspace")
            for channel in range(3):
                rows = [
                    int(np.ravel_multi_index((template, patch_y, patch_x, channel), values.shape))
                    for patch_y in range(2)
                    for patch_x in range(2)
                ]
                add_template_latent(
                    rows,
                    int(values[template, 0, 0, channel]),
                    f"template:{template}:uniform:c{channel}",
                )
        else:
            axis = axes[template]
            for side in range(2):
                coordinates = (
                    [(patch_y, side) for patch_y in range(2)]
                    if axis == "x"
                    else [(side, patch_x) for patch_x in range(2)]
                )
                for channel in range(3):
                    owned = [
                        int(np.ravel_multi_index((template, patch_y, patch_x, channel), values.shape))
                        for patch_y, patch_x in coordinates
                    ]
                    samples = [values.reshape(-1)[row] for row in owned]
                    if any(int(sample) != int(samples[0]) for sample in samples[1:]):
                        raise IterativeRealizedTrustError(
                            "boundary-normal origin is outside its side-equality subspace"
                        )
                    add_template_latent(
                        owned,
                        int(samples[0]),
                        f"template:{template}:normal_{axis}:side{side}:c{channel}",
                    )
    template_latent_count = len(columns)
    for support in range(compensation.shape[0]):
        for channel in range(3):
            row = full_template_count + support * 3 + channel
            column = np.zeros(full_count, dtype=np.float64)
            column[row] = 1.0
            columns.append(column)
            current.append(int(compensation[support, channel]))
            labels.append(f"sparse_comp:{support}:c{channel}")
    matrix = np.stack(columns, axis=1)
    return BasisProjection(
        selected,
        matrix,
        np.asarray(current, dtype=np.int64),
        tuple(labels),
        template_latent_count,
        axes,
    )


def categorical_fisher_trace_from_margin(margin: np.ndarray) -> np.ndarray:
    """Binary top-two Fisher trace ``0.5*sech(m/2)^2``.

    The margin is a frozen-head surrogate for the full categorical Fisher.
    Clipping only prevents overflow and leaves the tie-tight regime unchanged.
    """

    value = np.asarray(margin, dtype=np.float64)
    if value.ndim != 1 or not np.isfinite(value).all():
        raise IterativeRealizedTrustError("margin must be a finite vector")
    clipped = np.clip(value * 0.5, -40.0, 40.0)
    return 0.5 / np.square(np.cosh(clipped))


def fisher_margin_debt(margin: np.ndarray, required: np.ndarray) -> float:
    """Fisher-weighted signed-margin hinge used only as model currency."""

    value = np.asarray(margin, dtype=np.float64)
    target = np.asarray(required, dtype=np.float64)
    if value.ndim != 1 or target.shape != value.shape or not np.isfinite(target).all():
        raise IterativeRealizedTrustError("margin debt geometry differs")
    deficit = np.maximum(target - value, 0.0)
    # Preserve a nonzero floor away from ties so a target crossing remains in
    # the model objective while still ranking tie-tight cells highest.
    weight = 1.0e-6 + categorical_fisher_trace_from_margin(value)
    return float(weight @ deficit)


def quantized_babai_candidates(
    continuous_step: np.ndarray,
    hessian: np.ndarray,
    *,
    current: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    trust_radius: float | np.ndarray,
    scales: Sequence[float],
    maximum_candidates: int,
) -> tuple[LatticeCandidate, ...]:
    """Return unique, nonzero, box-safe nearest-plane candidates.

    Every continuous scaled proposal is clipped to the current trust box,
    Babai-projected, then clipped again to the physical integer bounds.  The
    returned step is therefore the only state a caller is allowed to measure.
    """

    step = np.asarray(continuous_step, dtype=np.float64)
    metric = np.asarray(hessian, dtype=np.float64)
    origin = np.asarray(current, dtype=np.int64)
    lo = np.asarray(lower, dtype=np.int64)
    hi = np.asarray(upper, dtype=np.int64)
    if (
        step.ndim != 1
        or metric.shape != (step.size, step.size)
        or origin.shape != step.shape
        or lo.shape != step.shape
        or hi.shape != step.shape
        or not np.isfinite(step).all()
        or not np.isfinite(metric).all()
        or np.any(lo > origin)
        or np.any(origin > hi)
    ):
        raise IterativeRealizedTrustError("Babai candidate inputs are malformed")
    cap = int(maximum_candidates)
    if isinstance(maximum_candidates, bool) or cap <= 0 or cap != maximum_candidates:
        raise IterativeRealizedTrustError("maximum_candidates must be a positive integer")
    radius = np.asarray(trust_radius, dtype=np.float64)
    if radius.ndim == 0:
        radius = np.full(step.size, float(radius), dtype=np.float64)
    if radius.shape != step.shape or not np.isfinite(radius).all() or np.any(radius <= 0.0):
        raise IterativeRealizedTrustError("trust radius must be positive per coordinate")
    ordered_scales = []
    for raw in scales:
        scale = float(raw)
        if not math.isfinite(scale) or scale <= 0.0:
            raise IterativeRealizedTrustError("candidate scales must be finite and positive")
        if scale not in ordered_scales:
            ordered_scales.append(scale)
    if not ordered_scales:
        raise IterativeRealizedTrustError("at least one candidate scale is required")

    rows: list[LatticeCandidate] = []
    seen: set[bytes] = set()
    for scale in ordered_scales:
        continuous = np.clip(scale * step, -radius, radius)
        projection = babai_nearest_plane(continuous, metric)
        raw_integer = projection.integer_step
        bounded_state = np.clip(origin + raw_integer, lo, hi)
        integer = bounded_state - origin
        if not np.any(integer):
            continue
        key = np.ascontiguousarray(integer, dtype="<i8").tobytes()
        if key in seen:
            continue
        seen.add(key)
        error = integer.astype(np.float64) - continuous
        rows.append(
            LatticeCandidate(
                candidate_id=f"babai_{len(rows):02d}_scale_{scale:.8g}",
                scale=scale,
                integer_step=integer,
                continuous_step=continuous,
                quadratic_error=float(error @ metric @ error),
                covering_bound=projection.covering_bound,
                clipped_coordinate_count=int(np.count_nonzero(raw_integer != integer)),
            )
        )
        if len(rows) == cap:
            break
    return tuple(rows)


def ranked_prefix_sign_candidates(
    continuous_step: np.ndarray,
    *,
    current: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    trust_radii: Sequence[float],
) -> tuple[LatticeCandidate, ...]:
    """Matched-budget model-disabled descent controls on the same lattice.

    The sign is inherited from the caller's clipped first-order step.  Stable
    magnitude ranking determines nested coordinate prefixes, and each prefix
    uses the corresponding integer trust radius.  No QP/KKT model is queried.
    Requiring enough nonzero coordinates prevents duplicate exact-scorer calls
    from masquerading as an equal-budget comparison.
    """

    step = np.asarray(continuous_step, dtype=np.float64)
    origin = np.asarray(current, dtype=np.int64)
    lo = np.asarray(lower, dtype=np.int64)
    hi = np.asarray(upper, dtype=np.int64)
    radii = tuple(float(value) for value in trust_radii)
    if (
        step.ndim != 1
        or origin.shape != step.shape
        or lo.shape != step.shape
        or hi.shape != step.shape
        or not np.isfinite(step).all()
        or np.any(lo > origin)
        or np.any(origin > hi)
        or not radii
        or any(not math.isfinite(value) or value < 1.0 for value in radii)
    ):
        raise IterativeRealizedTrustError("ranked-prefix descent inputs are malformed")
    ranked = np.flatnonzero(step)
    ranked = ranked[np.argsort(-np.abs(step[ranked]), kind="stable")]
    if ranked.size < len(radii):
        raise IterativeRealizedTrustError("equal-budget descent needs one nonzero coordinate per exact call")
    rows = []
    seen: set[bytes] = set()
    for index, radius in enumerate(radii):
        prefix = max(index + 1, math.ceil(ranked.size * (index + 1) / len(radii)))
        amplitude = max(1, math.floor(radius))
        integer = np.zeros(step.size, dtype=np.int64)
        selected = ranked[:prefix]
        integer[selected] = np.sign(step[selected]).astype(np.int64) * amplitude
        bounded = np.clip(origin + integer, lo, hi)
        integer = bounded - origin
        if not np.any(integer):
            raise IterativeRealizedTrustError("ranked-prefix descent collapsed at the physical box")
        digest = np.ascontiguousarray(integer, dtype="<i8").tobytes()
        if digest in seen:
            raise IterativeRealizedTrustError("ranked-prefix descent produced a duplicate exact call")
        seen.add(digest)
        continuous = integer.astype(np.float64)
        rows.append(
            LatticeCandidate(
                candidate_id=f"ranked_prefix_sign_{index:02d}_r{radius:g}",
                scale=radius,
                integer_step=integer,
                continuous_step=continuous,
                quadratic_error=0.0,
                covering_bound=0.25 * integer.size,
                clipped_coordinate_count=int(np.count_nonzero(bounded == lo) + np.count_nonzero(bounded == hi)),
            )
        )
    return tuple(rows)


def select_realized_improvement(
    baseline_objective: float,
    candidates: Sequence[HardCandidate],
    *,
    tolerance: float = 0.0,
) -> RealizedSelection:
    """Select the best admissible hard improvement; never proxy-accept."""

    baseline = float(baseline_objective)
    tol = float(tolerance)
    if not math.isfinite(baseline) or not math.isfinite(tol) or tol < 0.0:
        raise IterativeRealizedTrustError("hard selection baseline/tolerance is invalid")
    rows = tuple(candidates)
    admissible = tuple(row for row in rows if row.admissible)
    improving = tuple(row for row in admissible if row.hard_objective < baseline - tol)
    if not improving:
        return RealizedSelection(None, False, baseline, 0.0, None, len(rows), len(admissible))
    selected = min(
        improving,
        key=lambda row: (row.hard_objective, row.archive_bytes, row.d_seg, row.candidate_id),
    )
    predicted = selected.predicted_reduction
    rho = selected.realized_model_reduction / predicted if predicted > 0.0 else None
    if rho is not None and not math.isfinite(rho):
        rho = None
    return RealizedSelection(
        selected,
        True,
        baseline,
        baseline - selected.hard_objective,
        rho,
        len(rows),
        len(admissible),
    )


def update_trust_radius(
    radius: float,
    *,
    rho: float | None,
    accepted: bool,
    policy: TrustRegionPolicy,
) -> TrustDecision:
    """Update a scalar lattice radius without conferring acceptance."""

    old = float(radius)
    if not math.isfinite(old) or not policy.minimum_radius <= old <= policy.maximum_radius:
        raise IterativeRealizedTrustError("current trust radius is outside policy bounds")
    ratio = None if rho is None else float(rho)
    if ratio is not None and not math.isfinite(ratio):
        ratio = None
    if ratio is not None and ratio < 0.0:
        new = old * policy.negative_rho_factor
        update = TrustUpdate.HARD_SHRINK_NEGATIVE_RHO
    elif ratio is None:
        new = old * policy.shrink_factor
        update = TrustUpdate.SHRINK_MODEL_INVALID
    elif not accepted or ratio < policy.poor_rho:
        new = old * policy.shrink_factor
        update = TrustUpdate.SHRINK_REJECTED
    elif ratio >= policy.growth_rho:
        new = old * policy.growth_factor
        update = TrustUpdate.GROW_ACCEPTED_HIGH_RHO
    else:
        new = old
        update = TrustUpdate.HOLD_ACCEPTED
    bounded = min(policy.maximum_radius, max(policy.minimum_radius, new))
    return TrustDecision(old, bounded, update, ratio)


def summarize_validity_curve(rows: Sequence[dict[str, float | int | None]]) -> tuple[dict[str, object], ...]:
    """Aggregate raw predicted/realized reductions by lattice step quantum."""

    groups: dict[float, list[tuple[float, float, float | None]]] = {}
    for row in rows:
        quantum = float(row["lattice_quanta"])
        predicted = float(row["predicted_reduction"])
        realized = float(row["realized_reduction"])
        raw_rho = row.get("rho")
        rho = None if raw_rho is None else float(raw_rho)
        if not all(math.isfinite(value) for value in (quantum, predicted, realized)) or quantum <= 0.0:
            raise IterativeRealizedTrustError("validity-curve row is malformed")
        if rho is not None and not math.isfinite(rho):
            rho = None
        groups.setdefault(quantum, []).append((predicted, realized, rho))
    result = []
    for quantum in sorted(groups):
        values = groups[quantum]
        ratios = np.asarray([value[2] for value in values if value[2] is not None], dtype=np.float64)
        result.append(
            {
                "lattice_quanta": quantum,
                "candidate_count": len(values),
                "rho_count": int(ratios.size),
                "rho_median": None if not ratios.size else float(np.median(ratios)),
                "rho_mean": None if not ratios.size else float(np.mean(ratios)),
                "negative_rho_count": int(np.count_nonzero(ratios < 0.0)),
                "positive_realized_count": sum(value[1] > 0.0 for value in values),
                "predicted_reduction_mean": float(np.mean([value[0] for value in values])),
                "realized_reduction_mean": float(np.mean([value[1] for value in values])),
            }
        )
    return tuple(result)


def bounded_parallel_tempering(
    initial: np.ndarray,
    *,
    lower: np.ndarray,
    upper: np.ndarray,
    coordinates: Sequence[int],
    cheap_energy: Callable[[np.ndarray], float],
    hard_key: Callable[[np.ndarray], tuple[float, ...]],
    seed: int,
    sweeps: int = 16,
    temperature_multipliers: Sequence[float] = (0.25, 0.5, 1.0, 2.0),
) -> TemperingResult:
    """Bounded deterministic PT over integer unit moves.

    Temperatures are derived from the robust spread of local cheap-energy
    probes.  Cheap energy controls only Metropolis traversal.  ``hard_key`` is
    evaluated for the initial state and unique terminals and is the sole final
    ranking authority.
    """

    origin = np.asarray(initial, dtype=np.int64)
    lo = np.asarray(lower, dtype=np.int64)
    hi = np.asarray(upper, dtype=np.int64)
    if origin.ndim != 1 or lo.shape != origin.shape or hi.shape != origin.shape:
        raise IterativeRealizedTrustError("tempering state geometry differs")
    if np.any(origin < lo) or np.any(origin > hi):
        raise IterativeRealizedTrustError("tempering initial state is outside bounds")
    if isinstance(sweeps, bool) or int(sweeps) != sweeps or sweeps <= 0:
        raise IterativeRealizedTrustError("tempering sweeps must be a positive integer")
    coords = tuple(sorted({int(value) for value in coordinates}))
    if any(value < 0 or value >= origin.size for value in coords):
        raise IterativeRealizedTrustError("tempering coordinate is out of range")
    movable = tuple(value for value in coords if lo[value] < origin[value] or origin[value] < hi[value])
    if not movable:
        return TemperingResult(TemperingStatus.N_A_NO_MOVABLE_COORDINATES, (), (), 0, 0, 0, None)
    base = float(cheap_energy(origin.copy()))
    if not math.isfinite(base):
        raise IterativeRealizedTrustError("tempering initial cheap energy is nonfinite")
    local_deltas = []
    seeds = []
    for coordinate in movable:
        for delta in (-1, 1):
            candidate = origin.copy()
            candidate[coordinate] = np.clip(candidate[coordinate] + delta, lo[coordinate], hi[coordinate])
            if np.array_equal(candidate, origin):
                continue
            energy = float(cheap_energy(candidate.copy()))
            if not math.isfinite(energy):
                raise IterativeRealizedTrustError("tempering cheap energy is nonfinite")
            local_deltas.append(abs(energy - base))
            seeds.append((energy, coordinate, delta, candidate))
    positive = np.asarray([value for value in local_deltas if value > 0.0], dtype=np.float64)
    if not positive.size:
        return TemperingResult(TemperingStatus.N_A_DEGENERATE_ENERGY_SPREAD, (), (), 0, 0, 0, None)
    spread = float(np.median(positive))
    multipliers = tuple(float(value) for value in temperature_multipliers)
    if not multipliers or any(not math.isfinite(value) or value <= 0.0 for value in multipliers):
        raise IterativeRealizedTrustError("tempering multipliers must be finite and positive")
    temperatures = tuple(spread * value for value in multipliers)
    rng = np.random.default_rng(int(seed))
    ranked_seeds = sorted(seeds, key=lambda value: (value[0], value[1], value[2]))
    states = [origin.copy() for _ in temperatures]
    energies = [base for _ in temperatures]
    for replica in range(1, len(states)):
        energy, _coordinate, _delta, candidate = ranked_seeds[(replica - 1) % len(ranked_seeds)]
        states[replica] = candidate.copy()
        energies[replica] = energy

    proposals = 0
    accepts = 0
    swaps = 0
    for _ in range(int(sweeps)):
        for replica, temperature in enumerate(temperatures):
            coordinate = movable[int(rng.integers(0, len(movable)))]
            delta = -1 if int(rng.integers(0, 2)) == 0 else 1
            candidate = states[replica].copy()
            candidate[coordinate] = np.clip(candidate[coordinate] + delta, lo[coordinate], hi[coordinate])
            proposals += 1
            if np.array_equal(candidate, states[replica]):
                continue
            energy = float(cheap_energy(candidate.copy()))
            if not math.isfinite(energy):
                raise IterativeRealizedTrustError("tempering proposal energy is nonfinite")
            exponent = min(0.0, (energies[replica] - energy) / temperature)
            if float(rng.random()) < math.exp(exponent):
                states[replica] = candidate
                energies[replica] = energy
                accepts += 1
        parity = _ % 2
        for left in range(parity, len(states) - 1, 2):
            right = left + 1
            exponent = min(
                0.0,
                (1.0 / temperatures[left] - 1.0 / temperatures[right]) * (energies[left] - energies[right]),
            )
            if float(rng.random()) < math.exp(exponent):
                states[left], states[right] = states[right], states[left]
                energies[left], energies[right] = energies[right], energies[left]
                swaps += 1

    baseline_key = tuple(float(value) for value in hard_key(origin.copy()))
    terminals = []
    seen: set[bytes] = set()
    for replica, (state, energy) in enumerate(zip(states, energies, strict=True)):
        digest = np.ascontiguousarray(state, dtype="<i8").tobytes()
        if digest in seen:
            continue
        seen.add(digest)
        key = tuple(float(value) for value in hard_key(state.copy()))
        if not key or not all(math.isfinite(value) for value in key):
            raise IterativeRealizedTrustError("tempering hard key is malformed")
        terminals.append(TemperingTerminal(replica, state, energy, key))
    winner = min(terminals, key=lambda row: (row.hard_key, row.replica))
    improved = winner.hard_key < baseline_key
    return TemperingResult(
        TemperingStatus.HARD_IMPROVEMENT if improved else TemperingStatus.NO_HARD_IMPROVEMENT,
        temperatures,
        tuple(terminals),
        proposals,
        accepts,
        swaps,
        winner.replica if improved else None,
    )


def _immutable(value: np.ndarray) -> np.ndarray:
    out = np.array(value, copy=True)
    out.setflags(write=False)
    return out


__all__ = [
    "BasisProjection",
    "HardCandidate",
    "IterativeRealizedTrustError",
    "LatticeCandidate",
    "RealizedSelection",
    "TemperingResult",
    "TemperingStatus",
    "TemplateBasis",
    "TrustDecision",
    "TrustRegionPolicy",
    "TrustUpdate",
    "bounded_parallel_tempering",
    "build_template_basis_projection",
    "categorical_fisher_trace_from_margin",
    "fisher_margin_debt",
    "quantized_babai_candidates",
    "ranked_prefix_sign_candidates",
    "select_realized_improvement",
    "summarize_validity_curve",
    "update_trust_radius",
]
