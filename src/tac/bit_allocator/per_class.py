# SPDX-License-Identifier: MIT
"""Per-SegNet-class bit allocator (canonical-prior weighted; Catalog #125 hook #3).

Canonical bit-allocator helper that consumes per-class priors (e.g.
SegNet-class chroma priority from NSCS06 v7 cargo-cult-unwind methodology
or per-class master-gradient sensitivity from
``tac.cathedral_consumers.per_segnet_class_chroma_consumer``) and
allocates a fixed total bit budget across the 5 SegNet classes
(BACKGROUND / VEHICLE / PERSON / ROAD / TRAFFIC_SIGN, indices 0..4) per
the contest scorer's class taxonomy.

Per the NSCS06 v6→v7 44% improvement (105.15 → 58.89 contest-CUDA) per
canonical-vs-unique unwind: chroma allocation per class (vs global)
unlocked the score; this allocator codifies that prior as a primary
bit-allocation surface.

Per CLAUDE.md "Subagent coherence-by-default" non-negotiable, this module:

* hook #1 sensitivity-map = ACTIVE (consumes per-class priors)
* hook #2 Pareto constraint = ACTIVE (total_bits is the Pareto bound)
* hook #3 bit-allocator = **PRIMARY**
* hook #4 cathedral autopilot dispatch = ACTIVE via
  ``per_segnet_class_chroma_consumer``
* hook #5 continual-learning posterior = ACTIVE (per-class priors update
  on new anchors from canonical equations registry)
* hook #6 probe-disambiguator = ACTIVE (PROPORTIONAL vs SQRT vs UNIFORM
  enum exposes the choice)

Per Catalog #323 every PerClassBitAllocationPlan carries canonical
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


CANONICAL_MODEL_ID_PER_CLASS = "tac.bit_allocator.per_class.v1"
"""Canonical model identifier for Provenance attribution."""

SEGNET_CLASS_COUNT = 5
"""Canonical SegNet class count per upstream scorer (BG/VEHICLE/PERSON/ROAD/SIGN)."""

CANONICAL_SEGNET_CLASS_NAMES = (
    "background",
    "vehicle",
    "person",
    "road",
    "traffic_sign",
)
"""Canonical per-index class names for diagnostic purposes (notes only)."""


class PerClassAllocationStrategy(Enum):
    """Canonical per-class allocation strategy (Catalog #125 hook #6).

    Members:
        UNIFORM: every class gets equal bits; baseline.
        PROPORTIONAL: bits_c ∝ prior_c (most aggressive).
        SQRT: bits_c ∝ sqrt(prior_c); concavity preserves non-zero bits per
            class when budget >= N (canonical for variance reduction).
    """

    UNIFORM = "uniform"
    PROPORTIONAL = "proportional"
    SQRT = "sqrt"


class PerClassAllocationError(ValueError):
    """Raised for an invalid input to :func:`allocate_per_class`."""


@dataclass(frozen=True)
class PerClassBitAllocationPlan:
    """Canonical frozen per-class bit-allocation manifest.

    Per Catalog #323 every score-claim-adjacent payload carries Provenance.
    The bit-allocator output is observability-only (PREDICTED grade).

    Attributes:
        bits_per_class: class_index (0..N-1) -> bit count; sums to total_budget_bits
        strategy: the :class:`PerClassAllocationStrategy` used
        total_budget_bits: input Pareto constraint
        n_classes: count of classes the allocation covers (default 5 = SegNet)
        provenance: Catalog #323 canonical Provenance object
        score_claim: ALWAYS False
        promotion_eligible: ALWAYS False
        axis_tag: ALWAYS "[predicted]"
        notes: optional diagnostic dict (raw priors, class names)
    """

    bits_per_class: Mapping[int, int]
    strategy: PerClassAllocationStrategy
    total_budget_bits: int
    n_classes: int
    provenance: Provenance
    score_claim: bool = False
    promotion_eligible: bool = False
    axis_tag: str = "[predicted]"
    notes: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Catalog #323 non-promotable invariants."""
        if self.score_claim is not False:
            raise PerClassAllocationError(
                "PerClassBitAllocationPlan.score_claim must be False (Catalog #323)"
            )
        if self.promotion_eligible is not False:
            raise PerClassAllocationError(
                "PerClassBitAllocationPlan.promotion_eligible must be False (Catalog #323)"
            )
        if self.axis_tag != "[predicted]":
            raise PerClassAllocationError(
                "PerClassBitAllocationPlan.axis_tag must be '[predicted]'"
            )
        if self.provenance.evidence_grade != ProvenanceEvidenceGrade.PREDICTED:
            raise PerClassAllocationError(
                "provenance.evidence_grade must be PREDICTED"
            )
        if self.provenance.artifact_kind != ProvenanceKind.PREDICTED_FROM_MODEL:
            raise PerClassAllocationError(
                "provenance.artifact_kind must be PREDICTED_FROM_MODEL"
            )
        if self.total_budget_bits < 0:
            raise PerClassAllocationError(
                f"total_budget_bits must be non-negative, got {self.total_budget_bits}"
            )
        if self.n_classes <= 0:
            raise PerClassAllocationError(
                f"n_classes must be positive, got {self.n_classes}"
            )
        actual_sum = sum(self.bits_per_class.values())
        if actual_sum != self.total_budget_bits:
            raise PerClassAllocationError(
                f"sum(bits_per_class)={actual_sum} != total_budget_bits "
                f"{self.total_budget_bits}"
            )
        if len(self.bits_per_class) != self.n_classes:
            raise PerClassAllocationError(
                f"len(bits_per_class)={len(self.bits_per_class)} != "
                f"n_classes={self.n_classes}"
            )
        for class_idx, bits in self.bits_per_class.items():
            if not isinstance(class_idx, int) or isinstance(class_idx, bool):
                raise PerClassAllocationError(
                    f"class_index must be int, got {type(class_idx).__name__}"
                )
            if class_idx < 0 or class_idx >= self.n_classes:
                raise PerClassAllocationError(
                    f"class_index {class_idx} out of range [0, {self.n_classes})"
                )
            if not isinstance(bits, int) or isinstance(bits, bool):
                raise PerClassAllocationError(
                    f"bits must be int, got {type(bits).__name__}"
                )
            if bits < 0:
                raise PerClassAllocationError(
                    f"bits for class {class_idx} is negative: {bits}"
                )

    def as_dict(self) -> dict[str, object]:
        """JSON-serializable view."""
        return {
            "bits_per_class": {str(k): int(v) for k, v in self.bits_per_class.items()},
            "strategy": self.strategy.value,
            "total_budget_bits": int(self.total_budget_bits),
            "n_classes": int(self.n_classes),
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


def _normalize_priors(
    prior_per_class: Mapping[int, float],
    n_classes: int,
) -> tuple[float, ...]:
    """Validate + return tuple of priors indexed 0..n_classes-1.

    Missing class indices get prior 0.0. Each class_index must be in
    range [0, n_classes). Negative or non-finite priors raise.
    """
    if not prior_per_class:
        raise PerClassAllocationError(
            "prior_per_class must contain at least one class"
        )
    out = [0.0] * n_classes
    seen: set[int] = set()
    for raw_key, raw_val in prior_per_class.items():
        if isinstance(raw_key, bool) or not isinstance(raw_key, int):
            raise PerClassAllocationError(
                f"class_index must be int, got {type(raw_key).__name__}"
            )
        if raw_key < 0 or raw_key >= n_classes:
            raise PerClassAllocationError(
                f"class_index {raw_key} out of range [0, {n_classes})"
            )
        if raw_key in seen:
            raise PerClassAllocationError(
                f"class_index {raw_key} appears multiple times"
            )
        seen.add(raw_key)
        if isinstance(raw_val, bool):
            raise PerClassAllocationError(
                f"prior[{raw_key}] must be numeric, got bool"
            )
        try:
            p = float(raw_val)
        except (TypeError, ValueError) as exc:
            raise PerClassAllocationError(
                f"prior[{raw_key}] must be numeric"
            ) from exc
        if not math.isfinite(p):
            raise PerClassAllocationError(
                f"prior[{raw_key}] must be finite, got {p!r}"
            )
        if p < 0.0:
            raise PerClassAllocationError(
                f"prior[{raw_key}] must be non-negative, got {p!r}"
            )
        out[raw_key] = p
    return tuple(out)


def _compute_weights(
    priors: tuple[float, ...], strategy: PerClassAllocationStrategy
) -> tuple[float, ...]:
    if strategy is PerClassAllocationStrategy.UNIFORM:
        return tuple(1.0 for _ in priors)
    if strategy is PerClassAllocationStrategy.PROPORTIONAL:
        return priors
    if strategy is PerClassAllocationStrategy.SQRT:
        return tuple(math.sqrt(p) for p in priors)
    raise PerClassAllocationError(f"unknown strategy: {strategy!r}")


def _largest_remainder(
    total_bits: int, weights: tuple[float, ...]
) -> tuple[int, ...]:
    """Hamilton largest-remainder rounding for deterministic integer allocation."""
    n = len(weights)
    if n == 0:
        return ()
    total_weight = sum(weights)
    if total_weight <= 0.0:
        # Degenerate: uniform distribution.
        base = total_bits // n
        remainder = total_bits - base * n
        out = [base] * n
        for i in range(remainder):
            out[i] += 1
        return tuple(out)
    raw = [w * total_bits / total_weight for w in weights]
    floors = [math.floor(r) for r in raw]
    remainder = total_bits - sum(floors)
    # Distribute remainder by largest fractional part (tie-break: lower index).
    fracs = sorted(
        ((raw[i] - floors[i], i) for i in range(n)),
        key=lambda pair: (-pair[0], pair[1]),
    )
    for k in range(remainder):
        floors[fracs[k][1]] += 1
    return tuple(floors)


def _build_inputs_sha256(
    priors: tuple[float, ...],
    strategy: PerClassAllocationStrategy,
    total_budget_bits: int,
    n_classes: int,
    archive_sha256: str | None,
) -> str:
    """Deterministic sha256 of the inputs."""
    hasher = hashlib.sha256()
    hasher.update(b"tac.bit_allocator.per_class.v1\n")
    hasher.update(f"strategy={strategy.value}\n".encode())
    hasher.update(f"total_budget_bits={int(total_budget_bits)}\n".encode())
    hasher.update(f"n_classes={int(n_classes)}\n".encode())
    hasher.update(f"archive_sha256={archive_sha256 or ''}\n".encode())
    for i, p in enumerate(priors):
        hasher.update(f"{i}:{p!r}\n".encode())
    return hasher.hexdigest()


def allocate_per_class(
    total_budget_bits: int,
    prior_per_class: Mapping[int, float],
    *,
    strategy: PerClassAllocationStrategy | str = PerClassAllocationStrategy.SQRT,
    n_classes: int = SEGNET_CLASS_COUNT,
    archive_sha256: str | None = None,
    captured_at_utc: str | None = None,
) -> PerClassBitAllocationPlan:
    """Allocate per-SegNet-class bits via strategy-weighted budget split.

    Args:
        total_budget_bits: non-negative integer; total bit budget.
        prior_per_class: class_index (0..N-1) -> non-negative finite prior.
            Missing class indices get prior 0.0. Empty mapping raises.
        strategy: :class:`PerClassAllocationStrategy` or canonical string
            (``"uniform"`` / ``"proportional"`` / ``"sqrt"``). SQRT is the
            canonical default per NSCS06 v6→v7 unwind methodology.
        n_classes: total class count (default 5 = SegNet).
        archive_sha256: optional contest archive sha256 for Provenance.
        captured_at_utc: optional ISO-8601 timestamp.

    Returns:
        Frozen :class:`PerClassBitAllocationPlan` with canonical Provenance.

    Raises:
        PerClassAllocationError: on malformed input or invariant violation.
    """
    if isinstance(total_budget_bits, bool) or not isinstance(total_budget_bits, int):
        raise PerClassAllocationError(
            f"total_budget_bits must be int, got {type(total_budget_bits).__name__}"
        )
    if total_budget_bits < 0:
        raise PerClassAllocationError(
            f"total_budget_bits must be non-negative, got {total_budget_bits}"
        )
    if isinstance(n_classes, bool) or not isinstance(n_classes, int):
        raise PerClassAllocationError(
            f"n_classes must be int, got {type(n_classes).__name__}"
        )
    if n_classes <= 0:
        raise PerClassAllocationError(
            f"n_classes must be positive, got {n_classes}"
        )

    if isinstance(strategy, str):
        try:
            strategy_enum = PerClassAllocationStrategy(strategy)
        except ValueError as exc:
            raise PerClassAllocationError(
                f"unknown strategy string: {strategy!r}"
            ) from exc
    elif isinstance(strategy, PerClassAllocationStrategy):
        strategy_enum = strategy
    else:
        raise PerClassAllocationError(
            f"strategy must be PerClassAllocationStrategy or str, "
            f"got {type(strategy).__name__}"
        )

    priors = _normalize_priors(prior_per_class, n_classes)
    weights = _compute_weights(priors, strategy_enum)
    allocations = _largest_remainder(total_budget_bits, weights)
    bits_per_class = {int(i): int(allocations[i]) for i in range(n_classes)}

    inputs_sha = _build_inputs_sha256(
        priors, strategy_enum, total_budget_bits, n_classes, archive_sha256
    )
    if captured_at_utc is None:
        captured_at_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")
    provenance = build_provenance_for_predicted(
        model_id=CANONICAL_MODEL_ID_PER_CLASS,
        inputs_sha256=inputs_sha,
        measurement_axis="[predicted]",
        hardware_substrate="unknown",
        captured_at_utc=captured_at_utc,
    )

    # Map indexes to canonical SegNet class names when n_classes matches.
    class_names_field: tuple[str, ...] | None = None
    if n_classes == SEGNET_CLASS_COUNT:
        class_names_field = CANONICAL_SEGNET_CLASS_NAMES

    notes = {
        "model_id": CANONICAL_MODEL_ID_PER_CLASS,
        "strategy": strategy_enum.value,
        "n_classes": int(n_classes),
        "class_names": list(class_names_field) if class_names_field else None,
        "priors_sum": float(sum(priors)),
        "weights_sum": float(sum(weights)),
        "archive_sha256_prefix": (
            archive_sha256[:12] if isinstance(archive_sha256, str) else None
        ),
        "axis_tag": "[predicted]",
        "score_claim": False,
        "promotion_eligible": False,
    }

    return PerClassBitAllocationPlan(
        bits_per_class=bits_per_class,
        strategy=strategy_enum,
        total_budget_bits=int(total_budget_bits),
        n_classes=int(n_classes),
        provenance=provenance,
        score_claim=False,
        promotion_eligible=False,
        axis_tag="[predicted]",
        notes=notes,
    )


__all__ = (
    "CANONICAL_MODEL_ID_PER_CLASS",
    "CANONICAL_SEGNET_CLASS_NAMES",
    "PerClassAllocationError",
    "PerClassAllocationStrategy",
    "PerClassBitAllocationPlan",
    "SEGNET_CLASS_COUNT",
    "allocate_per_class",
)
