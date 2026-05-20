# SPDX-License-Identifier: MIT
"""Per-axis bit allocator (seg / pose / rate; Catalog #125 hook #3).

Canonical bit-allocator helper that allocates a total bit budget across
the THREE canonical contest scorer axes per upstream
``upstream/evaluate.py``:

    final_score = 100 * seg + sqrt(10 * pose) + 25 * archive_bytes / 37545489

Axes (canonical names + canonical scorer coefficients per CLAUDE.md
"SegNet vs PoseNet importance — operating-point dependent"):

* ``seg``  (SegNet distortion; coefficient 100)
* ``pose`` (PoseNet distortion; sqrt(10 * pose) — operating-point dependent
            marginal sensitivity)
* ``rate`` (archive bytes; coefficient 25 / 37545489)

Per CLAUDE.md "SegNet vs PoseNet importance — operating-point dependent":
the marginal value of bits-per-axis FLIPS as a function of the current
operating point. At PR106-frontier (pose_avg ~ 3.4e-5) pose marginal is
2.71x SegNet's; at old 1.x scores SegNet was 77x more important. The
canonical operating-point-aware allocation requires the caller to provide
per-axis sensitivity priors that capture the current operating point.

Per CLAUDE.md "Subagent coherence-by-default" non-negotiable, this module:

* hook #1 sensitivity-map = ACTIVE (consumes per-axis sensitivity)
* hook #2 Pareto constraint = ACTIVE (total_bits is the Pareto bound)
* hook #3 bit-allocator = **PRIMARY**
* hook #4 cathedral autopilot dispatch = ACTIVE via downstream consumers
* hook #5 continual-learning posterior = ACTIVE (per-axis sensitivity
  updates from canonical equations registry)
* hook #6 probe-disambiguator = ACTIVE (UNIFORM vs SENSITIVITY_WEIGHTED
  vs SCORER_FORMULA_WEIGHTED enum exposes the choice)

Per Catalog #323 every PerAxisBitAllocationPlan carries canonical
Provenance with ``evidence_grade=PREDICTED`` + ``score_claim=False`` +
``axis_tag="[predicted]"``.
"""
from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum

from tac.provenance import (
    Provenance,
    ProvenanceEvidenceGrade,
    ProvenanceKind,
    build_provenance_for_predicted,
)


CANONICAL_MODEL_ID_PER_AXIS = "tac.bit_allocator.per_axis.v1"
"""Canonical model identifier for Provenance attribution."""

CANONICAL_SCORER_AXES = ("seg", "pose", "rate")
"""Canonical contest scorer axes per upstream/evaluate.py."""

CANONICAL_SCORER_COEFFICIENTS_FORMULA_WEIGHTED = {
    "seg": 100.0,
    "pose": math.sqrt(10.0),  # marginal at unit pose; operating-point-dependent
    "rate": 25.0,
}
"""Canonical baseline coefficients for SCORER_FORMULA_WEIGHTED strategy.

NOTE: these are LITERAL scorer coefficients NOT operating-point-corrected
marginal sensitivities. Operating-point-aware strategies should use
SENSITIVITY_WEIGHTED with caller-provided per-axis priors. Per CLAUDE.md
"SegNet vs PoseNet importance — operating-point dependent" the marginal
values FLIP across the score landscape; the bare scorer coefficients are
a baseline allocation prior, not a substitute for empirical priors.
"""


class PerAxisAllocationStrategy(Enum):
    """Canonical per-axis allocation strategy (Catalog #125 hook #6).

    Members:
        UNIFORM: every axis gets equal bits (baseline; ignores priors).
        SENSITIVITY_WEIGHTED: bits_a ∝ caller-provided sensitivity_a
            (operating-point-aware; canonical default per CLAUDE.md).
        SCORER_FORMULA_WEIGHTED: bits_a ∝ scorer coefficient (seg=100,
            pose=sqrt(10), rate=25); operating-point-INDEPENDENT baseline.
    """

    UNIFORM = "uniform"
    SENSITIVITY_WEIGHTED = "sensitivity_weighted"
    SCORER_FORMULA_WEIGHTED = "scorer_formula_weighted"


class PerAxisAllocationError(ValueError):
    """Raised for an invalid input to :func:`allocate_per_axis`."""


@dataclass(frozen=True)
class PerAxisBitAllocationPlan:
    """Canonical frozen per-axis bit-allocation manifest.

    Attributes:
        bits_per_axis: axis_name -> bit count; sums to total_budget_bits.
            Keys must be from CANONICAL_SCORER_AXES.
        strategy: the :class:`PerAxisAllocationStrategy` used
        total_budget_bits: input Pareto constraint
        axes: tuple of axis names in canonical order (subset of CANONICAL_SCORER_AXES)
        provenance: Catalog #323 canonical Provenance object
        score_claim: ALWAYS False
        promotion_eligible: ALWAYS False
        axis_tag: ALWAYS "[predicted]"
        notes: optional diagnostic dict
    """

    bits_per_axis: Mapping[str, int]
    strategy: PerAxisAllocationStrategy
    total_budget_bits: int
    axes: tuple[str, ...]
    provenance: Provenance
    score_claim: bool = False
    promotion_eligible: bool = False
    axis_tag: str = "[predicted]"
    notes: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Catalog #323 non-promotable invariants + per-axis-name canonical contract."""
        if self.score_claim is not False:
            raise PerAxisAllocationError(
                "PerAxisBitAllocationPlan.score_claim must be False (Catalog #323)"
            )
        if self.promotion_eligible is not False:
            raise PerAxisAllocationError(
                "PerAxisBitAllocationPlan.promotion_eligible must be False"
            )
        if self.axis_tag != "[predicted]":
            raise PerAxisAllocationError(
                "PerAxisBitAllocationPlan.axis_tag must be '[predicted]'"
            )
        if self.provenance.evidence_grade != ProvenanceEvidenceGrade.PREDICTED:
            raise PerAxisAllocationError(
                "provenance.evidence_grade must be PREDICTED"
            )
        if self.provenance.artifact_kind != ProvenanceKind.PREDICTED_FROM_MODEL:
            raise PerAxisAllocationError(
                "provenance.artifact_kind must be PREDICTED_FROM_MODEL"
            )
        if self.total_budget_bits < 0:
            raise PerAxisAllocationError(
                f"total_budget_bits must be non-negative, got {self.total_budget_bits}"
            )
        if len(self.axes) == 0:
            raise PerAxisAllocationError(
                "axes must contain at least one canonical axis"
            )
        canonical_set = set(CANONICAL_SCORER_AXES)
        for axis in self.axes:
            if axis not in canonical_set:
                raise PerAxisAllocationError(
                    f"axis {axis!r} not in CANONICAL_SCORER_AXES={CANONICAL_SCORER_AXES}"
                )
        if len(set(self.axes)) != len(self.axes):
            raise PerAxisAllocationError(
                f"axes must be unique, got {self.axes!r}"
            )
        if set(self.bits_per_axis.keys()) != set(self.axes):
            raise PerAxisAllocationError(
                f"bits_per_axis keys {set(self.bits_per_axis.keys())!r} != "
                f"axes {set(self.axes)!r}"
            )
        actual_sum = sum(self.bits_per_axis.values())
        if actual_sum != self.total_budget_bits:
            raise PerAxisAllocationError(
                f"sum(bits_per_axis)={actual_sum} != total_budget_bits "
                f"{self.total_budget_bits}"
            )
        for axis_name, bits in self.bits_per_axis.items():
            if not isinstance(bits, int) or isinstance(bits, bool):
                raise PerAxisAllocationError(
                    f"bits[{axis_name!r}] must be int, got {type(bits).__name__}"
                )
            if bits < 0:
                raise PerAxisAllocationError(
                    f"bits[{axis_name!r}] is negative: {bits}"
                )

    def as_dict(self) -> dict[str, object]:
        """JSON-serializable view."""
        return {
            "bits_per_axis": {str(k): int(v) for k, v in self.bits_per_axis.items()},
            "strategy": self.strategy.value,
            "total_budget_bits": int(self.total_budget_bits),
            "axes": list(self.axes),
            "provenance": _provenance_to_jsonable(self.provenance),
            "score_claim": False,
            "promotion_eligible": False,
            "axis_tag": "[predicted]",
            "notes": dict(self.notes),
        }


def _provenance_to_jsonable(prov: Provenance) -> dict[str, object]:
    """Render Catalog #323 Provenance as a JSON-serializable dict."""
    return {
        "artifact_kind": prov.artifact_kind.name,
        "source_path": prov.source_path,
        "source_sha256": prov.source_sha256,
        "measurement_axis": prov.measurement_axis,
        "hardware_substrate": prov.hardware_substrate,
        "evidence_grade": prov.evidence_grade.name,
        "promotion_eligible": bool(prov.promotion_eligible),
        "score_claim_valid": bool(prov.score_claim_valid),
        "captured_at_utc": prov.captured_at_utc,
        "canonical_helper_invocation": prov.canonical_helper_invocation,
    }


def _normalize_sensitivity_per_axis(
    sensitivity_per_axis: Mapping[str, float],
    axes: tuple[str, ...],
) -> tuple[float, ...]:
    """Validate + return tuple of sensitivities in canonical axes order.

    Missing axes get sensitivity 0.0. Each axis name must be in
    CANONICAL_SCORER_AXES. Negative or non-finite raise.
    """
    canonical_set = set(CANONICAL_SCORER_AXES)
    out = [0.0] * len(axes)
    seen: set[str] = set()
    for raw_key, raw_val in sensitivity_per_axis.items():
        if not isinstance(raw_key, str):
            raise PerAxisAllocationError(
                f"axis name must be str, got {type(raw_key).__name__}"
            )
        if raw_key not in canonical_set:
            raise PerAxisAllocationError(
                f"axis {raw_key!r} not in CANONICAL_SCORER_AXES={CANONICAL_SCORER_AXES}"
            )
        if raw_key not in axes:
            raise PerAxisAllocationError(
                f"axis {raw_key!r} not in scope axes={axes!r}"
            )
        if raw_key in seen:
            raise PerAxisAllocationError(
                f"axis {raw_key!r} appears multiple times"
            )
        seen.add(raw_key)
        if isinstance(raw_val, bool):
            raise PerAxisAllocationError(
                f"sensitivity[{raw_key!r}] must be numeric, got bool"
            )
        try:
            s = float(raw_val)
        except (TypeError, ValueError) as exc:
            raise PerAxisAllocationError(
                f"sensitivity[{raw_key!r}] must be numeric"
            ) from exc
        if not math.isfinite(s):
            raise PerAxisAllocationError(
                f"sensitivity[{raw_key!r}] must be finite, got {s!r}"
            )
        if s < 0.0:
            raise PerAxisAllocationError(
                f"sensitivity[{raw_key!r}] must be non-negative, got {s!r}"
            )
        idx = axes.index(raw_key)
        out[idx] = s
    return tuple(out)


def _compute_weights(
    strategy: PerAxisAllocationStrategy,
    axes: tuple[str, ...],
    sensitivities: tuple[float, ...] | None,
) -> tuple[float, ...]:
    if strategy is PerAxisAllocationStrategy.UNIFORM:
        return tuple(1.0 for _ in axes)
    if strategy is PerAxisAllocationStrategy.SENSITIVITY_WEIGHTED:
        if sensitivities is None:
            raise PerAxisAllocationError(
                "SENSITIVITY_WEIGHTED requires sensitivity_per_axis"
            )
        return sensitivities
    if strategy is PerAxisAllocationStrategy.SCORER_FORMULA_WEIGHTED:
        return tuple(
            CANONICAL_SCORER_COEFFICIENTS_FORMULA_WEIGHTED[axis] for axis in axes
        )
    raise PerAxisAllocationError(f"unknown strategy: {strategy!r}")


def _largest_remainder(
    total_bits: int, weights: tuple[float, ...]
) -> tuple[int, ...]:
    """Hamilton largest-remainder rounding for deterministic integer allocation."""
    n = len(weights)
    if n == 0:
        return ()
    total_weight = sum(weights)
    if total_weight <= 0.0:
        base = total_bits // n
        remainder = total_bits - base * n
        out = [base] * n
        for i in range(remainder):
            out[i] += 1
        return tuple(out)
    raw = [w * total_bits / total_weight for w in weights]
    floors = [math.floor(r) for r in raw]
    remainder = total_bits - sum(floors)
    fracs = sorted(
        ((raw[i] - floors[i], i) for i in range(n)),
        key=lambda pair: (-pair[0], pair[1]),
    )
    for k in range(remainder):
        floors[fracs[k][1]] += 1
    return tuple(floors)


def _build_inputs_sha256(
    axes: tuple[str, ...],
    sensitivities: tuple[float, ...] | None,
    strategy: PerAxisAllocationStrategy,
    total_budget_bits: int,
    archive_sha256: str | None,
) -> str:
    """Deterministic sha256 of the inputs."""
    hasher = hashlib.sha256()
    hasher.update(b"tac.bit_allocator.per_axis.v1\n")
    hasher.update(f"strategy={strategy.value}\n".encode())
    hasher.update(f"total_budget_bits={int(total_budget_bits)}\n".encode())
    hasher.update(f"axes={','.join(axes)}\n".encode())
    hasher.update(f"archive_sha256={archive_sha256 or ''}\n".encode())
    if sensitivities is None:
        hasher.update(b"sensitivities=None\n")
    else:
        for axis, s in zip(axes, sensitivities, strict=True):
            hasher.update(f"{axis}:{s!r}\n".encode())
    return hasher.hexdigest()


def allocate_per_axis(
    total_budget_bits: int,
    *,
    strategy: PerAxisAllocationStrategy | str = PerAxisAllocationStrategy.UNIFORM,
    sensitivity_per_axis: Mapping[str, float] | None = None,
    axes: tuple[str, ...] = CANONICAL_SCORER_AXES,
    archive_sha256: str | None = None,
    captured_at_utc: str | None = None,
) -> PerAxisBitAllocationPlan:
    """Allocate bits across seg/pose/rate axes via strategy-weighted budget split.

    Args:
        total_budget_bits: non-negative integer.
        strategy: :class:`PerAxisAllocationStrategy` or canonical string.
            SENSITIVITY_WEIGHTED REQUIRES non-None sensitivity_per_axis.
        sensitivity_per_axis: axis_name -> non-negative finite sensitivity.
            Required by SENSITIVITY_WEIGHTED; ignored by UNIFORM and
            SCORER_FORMULA_WEIGHTED.
        axes: subset of CANONICAL_SCORER_AXES; canonical default is all 3.
        archive_sha256: optional contest archive sha256 for Provenance.
        captured_at_utc: optional ISO-8601 timestamp.

    Returns:
        Frozen :class:`PerAxisBitAllocationPlan` with canonical Provenance.

    Raises:
        PerAxisAllocationError: on malformed input or invariant violation.
    """
    if isinstance(total_budget_bits, bool) or not isinstance(total_budget_bits, int):
        raise PerAxisAllocationError(
            f"total_budget_bits must be int, got {type(total_budget_bits).__name__}"
        )
    if total_budget_bits < 0:
        raise PerAxisAllocationError(
            f"total_budget_bits must be non-negative, got {total_budget_bits}"
        )

    if isinstance(strategy, str):
        try:
            strategy_enum = PerAxisAllocationStrategy(strategy)
        except ValueError as exc:
            raise PerAxisAllocationError(
                f"unknown strategy string: {strategy!r}"
            ) from exc
    elif isinstance(strategy, PerAxisAllocationStrategy):
        strategy_enum = strategy
    else:
        raise PerAxisAllocationError(
            f"strategy must be PerAxisAllocationStrategy or str, "
            f"got {type(strategy).__name__}"
        )

    if not isinstance(axes, tuple):
        raise PerAxisAllocationError(
            f"axes must be a tuple of canonical axis names, got {type(axes).__name__}"
        )
    if len(axes) == 0:
        raise PerAxisAllocationError("axes must contain at least one axis")
    canonical_set = set(CANONICAL_SCORER_AXES)
    for axis in axes:
        if not isinstance(axis, str):
            raise PerAxisAllocationError(
                f"axis name must be str, got {type(axis).__name__}"
            )
        if axis not in canonical_set:
            raise PerAxisAllocationError(
                f"axis {axis!r} not in CANONICAL_SCORER_AXES={CANONICAL_SCORER_AXES}"
            )
    if len(set(axes)) != len(axes):
        raise PerAxisAllocationError(
            f"axes must be unique, got {axes!r}"
        )

    sensitivities: tuple[float, ...] | None = None
    if sensitivity_per_axis is not None:
        sensitivities = _normalize_sensitivity_per_axis(sensitivity_per_axis, axes)
    elif strategy_enum is PerAxisAllocationStrategy.SENSITIVITY_WEIGHTED:
        raise PerAxisAllocationError(
            "SENSITIVITY_WEIGHTED requires sensitivity_per_axis"
        )

    weights = _compute_weights(strategy_enum, axes, sensitivities)
    allocations = _largest_remainder(total_budget_bits, weights)
    bits_per_axis = {axes[i]: int(allocations[i]) for i in range(len(axes))}

    inputs_sha = _build_inputs_sha256(
        axes, sensitivities, strategy_enum, total_budget_bits, archive_sha256
    )
    if captured_at_utc is None:
        captured_at_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")
    provenance = build_provenance_for_predicted(
        model_id=CANONICAL_MODEL_ID_PER_AXIS,
        inputs_sha256=inputs_sha,
        measurement_axis="[predicted]",
        hardware_substrate="unknown",
        captured_at_utc=captured_at_utc,
    )

    notes = {
        "model_id": CANONICAL_MODEL_ID_PER_AXIS,
        "strategy": strategy_enum.value,
        "axes": list(axes),
        "sensitivity_provided": sensitivities is not None,
        "weights": list(weights),
        "archive_sha256_prefix": (
            archive_sha256[:12] if isinstance(archive_sha256, str) else None
        ),
        "axis_tag": "[predicted]",
        "score_claim": False,
        "promotion_eligible": False,
        "canonical_scorer_axes_pinned": list(CANONICAL_SCORER_AXES),
        "operating_point_aware": (
            strategy_enum is PerAxisAllocationStrategy.SENSITIVITY_WEIGHTED
        ),
    }

    return PerAxisBitAllocationPlan(
        bits_per_axis=bits_per_axis,
        strategy=strategy_enum,
        total_budget_bits=int(total_budget_bits),
        axes=axes,
        provenance=provenance,
        score_claim=False,
        promotion_eligible=False,
        axis_tag="[predicted]",
        notes=notes,
    )


__all__ = (
    "CANONICAL_MODEL_ID_PER_AXIS",
    "CANONICAL_SCORER_AXES",
    "CANONICAL_SCORER_COEFFICIENTS_FORMULA_WEIGHTED",
    "PerAxisAllocationError",
    "PerAxisAllocationStrategy",
    "PerAxisBitAllocationPlan",
    "allocate_per_axis",
)
