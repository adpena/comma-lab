# SPDX-License-Identifier: MIT
"""Honest restricted-master pricing and exact-replay beam selection for DDM.

The linear relaxation in this module is a proposal surface only.  A column's
``linear_objective_delta`` must come from a receiver-realized singleton
measurement, but overlapping columns need not compose additively.  Therefore
the beam selector calls the supplied exact-replay oracle for every explored
set, and only those replayed sets may become selected budget rows.

This split is deliberate:

* the restricted master provides byte/conflict duals and reduced costs;
* exact replay decides nonlinear set value after paint, uint8, R, and scorer;
* the three-round falsifier refuses to fire on incomplete or proxy evidence.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np


class DDMColumnGenerationError(ValueError):
    """Fail-closed malformed column-generation evidence."""


@dataclass(frozen=True, slots=True)
class PricedColumn:
    """One real-coder-priced column with a measured singleton objective delta."""

    column_id: str
    family: str
    real_coder_bytes: int
    linear_objective_delta: float
    conflict_keys: tuple[str, ...] = ()
    dependencies: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.column_id or not self.family:
            raise DDMColumnGenerationError("column_id and family must be nonempty")
        if (
            isinstance(self.real_coder_bytes, bool)
            or not isinstance(self.real_coder_bytes, int)
            or self.real_coder_bytes <= 0
        ):
            raise DDMColumnGenerationError("real_coder_bytes must be a positive integer")
        if not math.isfinite(float(self.linear_objective_delta)):
            raise DDMColumnGenerationError("linear_objective_delta must be finite")
        if len(set(self.conflict_keys)) != len(self.conflict_keys):
            raise DDMColumnGenerationError("conflict_keys must be unique per column")
        if len(set(self.dependencies)) != len(self.dependencies):
            raise DDMColumnGenerationError("dependencies must be unique per column")
        if self.column_id in self.dependencies:
            raise DDMColumnGenerationError("a column cannot depend on itself")


@dataclass(frozen=True, slots=True)
class RestrictedMasterDuals:
    """HiGHS-compatible duals for byte and conflict ``<=`` constraints."""

    byte_marginal: float
    conflict_marginals: Mapping[str, float]
    objective: float
    selected_fraction_by_id: Mapping[str, float]

    def __post_init__(self) -> None:
        values = (
            self.byte_marginal,
            self.objective,
            *self.conflict_marginals.values(),
            *self.selected_fraction_by_id.values(),
        )
        if not all(math.isfinite(float(value)) for value in values):
            raise DDMColumnGenerationError("restricted-master dual record must be finite")
        if self.byte_marginal > 1.0e-9 or any(
            float(value) > 1.0e-9 for value in self.conflict_marginals.values()
        ):
            raise DDMColumnGenerationError("dual marginals for <= constraints must be nonpositive")


@dataclass(frozen=True, slots=True)
class ReducedCostRow:
    column_id: str
    family: str
    real_coder_bytes: int
    linear_objective_delta: float
    reduced_cost: float
    negative_reduced_cost: bool


@dataclass(frozen=True, slots=True)
class ExactReplay:
    """Exact replay result for one selected column set."""

    column_ids: tuple[str, ...]
    archive_bytes: int
    d_seg: float
    d_pose: float
    objective: float
    archive_sha256: str
    receiver_closed: bool
    scorer_replayed: bool

    def __post_init__(self) -> None:
        if tuple(sorted(set(self.column_ids))) != self.column_ids:
            raise DDMColumnGenerationError("exact replay column_ids must be sorted and unique")
        if (
            isinstance(self.archive_bytes, bool)
            or not isinstance(self.archive_bytes, int)
            or self.archive_bytes <= 0
        ):
            raise DDMColumnGenerationError("exact replay archive_bytes must be positive")
        if not all(
            math.isfinite(float(value)) and float(value) >= 0.0
            for value in (self.d_seg, self.d_pose, self.objective)
        ):
            raise DDMColumnGenerationError("exact replay metrics must be finite and nonnegative")
        if len(self.archive_sha256) != 64 or any(ch not in "0123456789abcdef" for ch in self.archive_sha256):
            raise DDMColumnGenerationError("exact replay archive_sha256 must be lowercase SHA-256")
        if not self.receiver_closed or not self.scorer_replayed:
            raise DDMColumnGenerationError("exact replay must close receiver and scorer custody")


def solve_restricted_master_lp(
    columns: Sequence[PricedColumn],
    *,
    added_byte_budget: int,
) -> RestrictedMasterDuals:
    """Solve the additive LP relaxation and return its actual constraint duals.

    The objective coefficients are measured singleton deltas.  They are not
    promoted to a nonlinear set-value claim; the LP is used only for pricing.
    """

    if (
        isinstance(added_byte_budget, bool)
        or not isinstance(added_byte_budget, int)
        or added_byte_budget < 0
    ):
        raise DDMColumnGenerationError("added_byte_budget must be a nonnegative integer")
    ordered = _validated_columns(columns)
    if not ordered:
        return RestrictedMasterDuals(0.0, {}, 0.0, {})

    try:
        from scipy.optimize import linprog
    except ImportError as exc:  # pragma: no cover - repository runtime includes scipy
        raise DDMColumnGenerationError("scipy is required for restricted-master dual custody") from exc

    conflict_keys = tuple(sorted({key for column in ordered for key in column.conflict_keys}))
    rows: list[list[float]] = [
        [float(column.real_coder_bytes) for column in ordered],
    ]
    rhs: list[float] = [float(added_byte_budget)]
    for key in conflict_keys:
        rows.append([1.0 if key in column.conflict_keys else 0.0 for column in ordered])
        rhs.append(1.0)
    result = linprog(
        c=np.asarray([column.linear_objective_delta for column in ordered], dtype=np.float64),
        A_ub=np.asarray(rows, dtype=np.float64),
        b_ub=np.asarray(rhs, dtype=np.float64),
        bounds=[(0.0, 1.0)] * len(ordered),
        method="highs",
    )
    if not result.success or result.x is None or result.fun is None:
        raise DDMColumnGenerationError(f"restricted-master LP failed: {result.message}")
    marginals = np.asarray(result.ineqlin.marginals, dtype=np.float64)
    if marginals.shape != (1 + len(conflict_keys),):
        raise DDMColumnGenerationError("restricted-master dual cardinality differs")
    duals = RestrictedMasterDuals(
        byte_marginal=float(marginals[0]),
        conflict_marginals={
            key: float(marginals[index + 1]) for index, key in enumerate(conflict_keys)
        },
        objective=float(result.fun),
        selected_fraction_by_id={
            column.column_id: float(result.x[index]) for index, column in enumerate(ordered)
        },
    )
    _verify_lp_kkt(ordered, duals, added_byte_budget=added_byte_budget)
    return duals


def price_columns(
    columns: Sequence[PricedColumn],
    *,
    duals: RestrictedMasterDuals,
    tolerance: float = 1.0e-12,
) -> tuple[ReducedCostRow, ...]:
    """Price columns by ``c_j - a_j^T y`` using actual LP marginals."""

    if not math.isfinite(float(tolerance)) or tolerance < 0.0:
        raise DDMColumnGenerationError("pricing tolerance must be finite and nonnegative")
    rows: list[ReducedCostRow] = []
    for column in _validated_columns(columns):
        penalty = float(column.real_coder_bytes) * float(duals.byte_marginal)
        penalty += sum(float(duals.conflict_marginals.get(key, 0.0)) for key in column.conflict_keys)
        reduced = float(column.linear_objective_delta) - penalty
        rows.append(
            ReducedCostRow(
                column_id=column.column_id,
                family=column.family,
                real_coder_bytes=column.real_coder_bytes,
                linear_objective_delta=float(column.linear_objective_delta),
                reduced_cost=reduced,
                negative_reduced_cost=reduced < -float(tolerance),
            )
        )
    return tuple(sorted(rows, key=lambda row: (row.reduced_cost, row.column_id)))


def exact_replay_beam_select(
    columns: Sequence[PricedColumn],
    *,
    base_archive_bytes: int,
    added_byte_budget: int,
    replay: Callable[[tuple[str, ...]], ExactReplay],
    beam_width: int = 32,
) -> tuple[ExactReplay, tuple[ExactReplay, ...]]:
    """Conflict/dependency-aware beam whose every explored state is replayed.

    Returns ``(best, all_replays)``.  The empty set is always replayed, so the
    caller cannot silently substitute a memo control row.
    """

    if (
        isinstance(base_archive_bytes, bool)
        or not isinstance(base_archive_bytes, int)
        or base_archive_bytes <= 0
    ):
        raise DDMColumnGenerationError("base_archive_bytes must be positive")
    if (
        isinstance(added_byte_budget, bool)
        or not isinstance(added_byte_budget, int)
        or added_byte_budget < 0
    ):
        raise DDMColumnGenerationError("added_byte_budget must be nonnegative")
    if isinstance(beam_width, bool) or not isinstance(beam_width, int) or beam_width < 1:
        raise DDMColumnGenerationError("beam_width must be positive")
    ordered = _validated_columns(columns)
    by_id = {column.column_id: column for column in ordered}
    for column in ordered:
        missing = sorted(set(column.dependencies) - set(by_id))
        if missing:
            raise DDMColumnGenerationError(f"{column.column_id} has missing dependencies: {missing}")
    _refuse_dependency_cycles(by_id)

    replayed: dict[tuple[str, ...], ExactReplay] = {}

    def evaluate(ids: Iterable[str]) -> ExactReplay:
        key = tuple(sorted(set(ids)))
        if key not in replayed:
            row = replay(key)
            if row.column_ids != key:
                raise DDMColumnGenerationError("replay returned a different column set")
            replayed[key] = row
        return replayed[key]

    empty = evaluate(())
    beam: list[tuple[str, ...]] = [()]
    best = empty
    for _depth in range(len(ordered)):
        expanded: set[tuple[str, ...]] = set(beam)
        for state in beam:
            state_set = set(state)
            occupied = {
                key for column_id in state for key in by_id[column_id].conflict_keys
            }
            for column in ordered:
                if column.column_id in state_set:
                    continue
                if not set(column.dependencies).issubset(state_set):
                    continue
                if occupied.intersection(column.conflict_keys):
                    continue
                candidate = tuple(sorted((*state, column.column_id)))
                row = evaluate(candidate)
                if row.archive_bytes <= base_archive_bytes + added_byte_budget:
                    expanded.add(candidate)
                    if _replay_key(row) < _replay_key(best):
                        best = row
        ranked = sorted(expanded, key=lambda ids: _replay_key(evaluate(ids)))
        next_beam = ranked[:beam_width]
        if next_beam == beam:
            break
        beam = next_beam
    return best, tuple(sorted(replayed.values(), key=lambda row: (len(row.column_ids), row.column_ids)))


def generated_vocabulary_falsifier(
    *,
    pricing_rounds: Sequence[Mapping[str, Any]],
    equal_byte_rows: Sequence[Mapping[str, Any]],
    v12_d_seg: float,
    expected_added_byte_budgets: Sequence[int] = (16_384, 49_152, 98_304, 147_456),
) -> bool:
    """Return true only for the exact preregistered three-round conjunction."""

    try:
        control_d_seg = float(v12_d_seg)
    except (TypeError, ValueError):
        return False
    if not math.isfinite(control_d_seg) or control_d_seg < 0.0:
        return False
    budgets = tuple(expected_added_byte_budgets)
    if (
        len(pricing_rounds) != 3
        or len(equal_byte_rows) != 4
        or len(budgets) != 4
        or any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in budgets)
    ):
        return False
    for expected, row in enumerate(pricing_rounds, start=1):
        if (
            row.get("round") != expected
            or row.get("complete") is not True
            or row.get("exact_pricing") is not True
            or row.get("negative_reduced_cost_count") != 0
        ):
            return False
    for budget, row in zip(budgets, equal_byte_rows, strict=True):
        generated = row.get("generated_vocabulary")
        if not isinstance(generated, Mapping):
            return False
        try:
            candidate_d_seg = float(generated.get("d_seg"))
        except (TypeError, ValueError):
            return False
        if (
            row.get("added_byte_budget") != budget
            or row.get("exact_replay_complete") is not True
            or row.get("global_selector") not in {"beam_width_32", "conflict_miqp"}
            or not math.isfinite(candidate_d_seg)
            or candidate_d_seg < control_d_seg
        ):
            return False
    return True


def _validated_columns(columns: Sequence[PricedColumn]) -> tuple[PricedColumn, ...]:
    ordered = tuple(sorted(columns, key=lambda column: column.column_id))
    ids = [column.column_id for column in ordered]
    if len(set(ids)) != len(ids):
        raise DDMColumnGenerationError("column IDs must be unique")
    return ordered


def _refuse_dependency_cycles(by_id: Mapping[str, PricedColumn]) -> None:
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(column_id: str) -> None:
        if column_id in visiting:
            raise DDMColumnGenerationError("column dependency graph contains a cycle")
        if column_id in visited:
            return
        visiting.add(column_id)
        for dependency in by_id[column_id].dependencies:
            visit(dependency)
        visiting.remove(column_id)
        visited.add(column_id)

    for column_id in sorted(by_id):
        visit(column_id)


def _replay_key(row: ExactReplay) -> tuple[float, int, tuple[str, ...]]:
    return float(row.objective), int(row.archive_bytes), row.column_ids


def _verify_lp_kkt(
    columns: Sequence[PricedColumn],
    duals: RestrictedMasterDuals,
    *,
    added_byte_budget: int,
) -> None:
    priced = price_columns(columns, duals=duals, tolerance=1.0e-8)
    by_id = {row.column_id: row for row in priced}
    for column in columns:
        fraction = float(duals.selected_fraction_by_id[column.column_id])
        reduced = float(by_id[column.column_id].reduced_cost)
        if 1.0e-7 < fraction < 1.0 - 1.0e-7 and abs(reduced) > 1.0e-7:
            raise DDMColumnGenerationError("restricted-master interior variable violates stationarity")
        if fraction <= 1.0e-7 and reduced < -1.0e-7:
            raise DDMColumnGenerationError("restricted-master lower-bound KKT violation")
        if fraction >= 1.0 - 1.0e-7 and reduced > 1.0e-7:
            raise DDMColumnGenerationError("restricted-master upper-bound KKT violation")
    used = sum(
        column.real_coder_bytes * duals.selected_fraction_by_id[column.column_id]
        for column in columns
    )
    if used < added_byte_budget - 1.0e-7 and abs(duals.byte_marginal) > 1.0e-7:
        raise DDMColumnGenerationError("nonbinding byte row has a nonzero dual")
    for conflict_key, marginal in duals.conflict_marginals.items():
        conflict_use = sum(
            duals.selected_fraction_by_id[column.column_id]
            for column in columns
            if conflict_key in column.conflict_keys
        )
        if conflict_use < 1.0 - 1.0e-7 and abs(float(marginal)) > 1.0e-7:
            raise DDMColumnGenerationError("nonbinding conflict row has a nonzero dual")


__all__ = [
    "DDMColumnGenerationError",
    "ExactReplay",
    "PricedColumn",
    "ReducedCostRow",
    "RestrictedMasterDuals",
    "exact_replay_beam_select",
    "generated_vocabulary_falsifier",
    "price_columns",
    "solve_restricted_master_lp",
]
