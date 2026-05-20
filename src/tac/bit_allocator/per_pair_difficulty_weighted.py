# SPDX-License-Identifier: MIT
"""Per-pair difficulty-weighted bit allocator (Catalog #125 hook #3 PRIMARY).

Canonical bit-allocator helper that consumes the per-pair difficulty atlas
emitted by :mod:`tac.cathedral_consumers.per_pair_difficulty_atlas_consumer`
(SLOT MG-3 territory) and distributes a fixed total bit budget across pairs
according to a declared strategy.

The 3 canonical strategies (Catalog #125 hook #6 probe-disambiguator surface):

* ``UNIFORM``: every pair gets the same share (``total_bits / N``); ignores
  difficulty entirely. Baseline against which weighted strategies are scored.
* ``LINEAR``: bits per pair ∝ difficulty (``bits_p ∝ difficulty_p``). Most
  aggressive reweighting; can starve low-difficulty pairs.
* ``SQRT``: bits per pair ∝ ``sqrt(difficulty)``. Concavity ensures lower
  variance + non-zero allocation for every pair (when total bits ≥ N).

Per CLAUDE.md "Subagent coherence-by-default" non-negotiable, this module:

* hook #1 sensitivity-map = ACTIVE (consumes per-pair difficulty)
* hook #2 Pareto constraint = ACTIVE (total_bits is the Pareto bound)
* hook #3 bit-allocator = **PRIMARY**
* hook #4 cathedral autopilot dispatch = ACTIVE via per_pair_difficulty_atlas_consumer
* hook #5 continual-learning posterior = ACTIVE (atlas refresh on new anchor)
* hook #6 probe-disambiguator = ACTIVE (strategy enum exposes the choice)

The return value is a frozen :class:`BitAllocationResult` carrying canonical
Provenance per Catalog #323. It is non-promotable by construction
(``evidence_grade == PREDICTED``, ``score_claim is False``).
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


CANONICAL_MODEL_ID = "tac.bit_allocator.per_pair_difficulty_weighted.v1"
"""Canonical model identifier for Provenance attribution."""


class AllocationStrategy(Enum):
    """Canonical allocation strategy enum (Catalog #125 hook #6).

    Members:
        UNIFORM: bits_p = total_bits / N (baseline; ignores difficulty)
        LINEAR: bits_p ∝ difficulty_p
        SQRT: bits_p ∝ sqrt(difficulty_p)
    """

    UNIFORM = "uniform"
    LINEAR = "linear"
    SQRT = "sqrt"


class BitAllocationStrategyError(ValueError):
    """Raised for an invalid input to :func:`allocate_bits_per_pair`."""


@dataclass(frozen=True)
class BitAllocationResult:
    """Canonical frozen bit-allocation manifest.

    Per Catalog #323 every score-claim-adjacent payload carries Provenance.
    The bit-allocator output is observability-only (PREDICTED grade);
    downstream packet realization + paired-axis auth-eval is required
    before any contest score claim may be made.

    Attributes:
        bits_per_pair: pair_index -> bit count (sums to total_bits)
        strategy: the AllocationStrategy used
        total_bits: the input Pareto constraint
        n_pairs: count of pairs the allocation covers
        provenance: Catalog #323 canonical Provenance object
        score_claim: ALWAYS False (per Catalog #323 invariant)
        promotion_eligible: ALWAYS False (per Catalog #323 invariant)
        axis_tag: ALWAYS "[predicted]" (per Catalog #287/#323)
        notes: optional diagnostic dict (raw weights, normalization info)
    """

    bits_per_pair: Mapping[int, int]
    strategy: AllocationStrategy
    total_bits: int
    n_pairs: int
    provenance: Provenance
    score_claim: bool = False
    promotion_eligible: bool = False
    axis_tag: str = "[predicted]"
    notes: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Catalog #323 non-promotable invariants."""
        if self.score_claim is not False:
            raise BitAllocationStrategyError(
                "BitAllocationResult.score_claim must be False (Catalog #323)"
            )
        if self.promotion_eligible is not False:
            raise BitAllocationStrategyError(
                "BitAllocationResult.promotion_eligible must be False (Catalog #323)"
            )
        if self.axis_tag != "[predicted]":
            raise BitAllocationStrategyError(
                "BitAllocationResult.axis_tag must be '[predicted]' (Catalog #287/#323)"
            )
        if self.provenance.evidence_grade != ProvenanceEvidenceGrade.PREDICTED:
            raise BitAllocationStrategyError(
                "BitAllocationResult.provenance.evidence_grade must be PREDICTED"
            )
        if self.provenance.artifact_kind != ProvenanceKind.PREDICTED_FROM_MODEL:
            raise BitAllocationStrategyError(
                "BitAllocationResult.provenance.artifact_kind must be PREDICTED_FROM_MODEL"
            )
        actual_sum = sum(self.bits_per_pair.values())
        if actual_sum != self.total_bits:
            raise BitAllocationStrategyError(
                f"sum(bits_per_pair) == {actual_sum} != total_bits {self.total_bits}"
            )
        if len(self.bits_per_pair) != self.n_pairs:
            raise BitAllocationStrategyError(
                f"len(bits_per_pair) == {len(self.bits_per_pair)} != n_pairs {self.n_pairs}"
            )
        for pair_idx, bits in self.bits_per_pair.items():
            if not isinstance(pair_idx, int):
                raise BitAllocationStrategyError(
                    f"pair_index must be int, got {type(pair_idx).__name__}"
                )
            if not isinstance(bits, int):
                raise BitAllocationStrategyError(
                    f"bits must be int, got {type(bits).__name__}"
                )
            if bits < 0:
                raise BitAllocationStrategyError(
                    f"bits for pair {pair_idx} is negative: {bits}"
                )

    def as_dict(self) -> dict[str, object]:
        """JSON-serializable view (e.g. for sidecar artifact emission)."""
        return {
            "bits_per_pair": {str(k): int(v) for k, v in self.bits_per_pair.items()},
            "strategy": self.strategy.value,
            "total_bits": int(self.total_bits),
            "n_pairs": int(self.n_pairs),
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
        "contest_archive_zip_path": prov.contest_archive_zip_path,
        "contest_archive_member_name": prov.contest_archive_member_name,
        "rejection_reason": prov.rejection_reason,
    }


def _normalize_difficulty(
    difficulty_per_pair: Mapping[int, float],
) -> tuple[tuple[int, ...], tuple[float, ...]]:
    """Validate + return sorted (pair_index_tuple, difficulty_tuple)."""
    if not difficulty_per_pair:
        raise BitAllocationStrategyError(
            "difficulty_per_pair must contain at least one pair"
        )
    keys: list[int] = []
    values: list[float] = []
    for raw_key, raw_val in difficulty_per_pair.items():
        if isinstance(raw_key, bool) or not isinstance(raw_key, int):
            raise BitAllocationStrategyError(
                f"pair_index must be int, got {type(raw_key).__name__}"
            )
        if isinstance(raw_val, bool):
            raise BitAllocationStrategyError(
                f"difficulty[{raw_key}] must be numeric, got bool"
            )
        try:
            diff = float(raw_val)
        except (TypeError, ValueError) as exc:
            raise BitAllocationStrategyError(
                f"difficulty[{raw_key}] must be numeric"
            ) from exc
        if not math.isfinite(diff):
            raise BitAllocationStrategyError(
                f"difficulty[{raw_key}] must be finite, got {diff!r}"
            )
        if diff < 0.0:
            raise BitAllocationStrategyError(
                f"difficulty[{raw_key}] must be non-negative, got {diff!r}"
            )
        keys.append(raw_key)
        values.append(diff)
    # Sort by pair_index for determinism (also stabilizes hash/inputs sha).
    paired = sorted(zip(keys, values, strict=True), key=lambda p: p[0])
    return tuple(k for k, _ in paired), tuple(v for _, v in paired)


def _compute_raw_weights(
    diffs: tuple[float, ...], strategy: AllocationStrategy
) -> tuple[float, ...]:
    if strategy is AllocationStrategy.UNIFORM:
        return tuple(1.0 for _ in diffs)
    if strategy is AllocationStrategy.LINEAR:
        return diffs
    if strategy is AllocationStrategy.SQRT:
        return tuple(math.sqrt(d) for d in diffs)
    raise BitAllocationStrategyError(f"unknown strategy: {strategy!r}")


def _largest_remainder(
    total_bits: int, weights: tuple[float, ...]
) -> tuple[int, ...]:
    """Largest-remainder rounding for deterministic integer allocation.

    When all weights are zero, fall back to uniform allocation.
    """
    n = len(weights)
    if n == 0:
        return ()
    total_weight = sum(weights)
    if total_weight <= 0.0:
        # Degenerate: equal distribution.
        base = total_bits // n
        remainder = total_bits - base * n
        out = [base] * n
        for i in range(remainder):
            out[i] += 1
        return tuple(out)
    # Largest-remainder method (Hamilton apportionment).
    raw = [w * total_bits / total_weight for w in weights]
    floors = [math.floor(r) for r in raw]
    remainder = total_bits - sum(floors)
    # Distribute remainder by largest fractional part; tie-break by index.
    fracs = sorted(
        ((raw[i] - floors[i], i) for i in range(n)),
        key=lambda pair: (-pair[0], pair[1]),
    )
    for k in range(remainder):
        floors[fracs[k][1]] += 1
    return tuple(floors)


def _build_inputs_sha256(
    pair_keys: tuple[int, ...],
    diffs: tuple[float, ...],
    strategy: AllocationStrategy,
    total_bits: int,
    archive_sha256: str | None,
) -> str:
    """Deterministic sha256 of the inputs (for Provenance attribution)."""
    hasher = hashlib.sha256()
    hasher.update(b"tac.bit_allocator.per_pair_difficulty_weighted.v1\n")
    hasher.update(f"strategy={strategy.value}\n".encode())
    hasher.update(f"total_bits={int(total_bits)}\n".encode())
    hasher.update(f"archive_sha256={archive_sha256 or ''}\n".encode())
    hasher.update(f"n_pairs={len(pair_keys)}\n".encode())
    for key, diff in zip(pair_keys, diffs, strict=True):
        # Use repr to capture full float precision deterministically.
        hasher.update(f"{int(key)}:{diff!r}\n".encode())
    return hasher.hexdigest()


def allocate_bits_per_pair(
    total_bits: int,
    difficulty_per_pair: Mapping[int, float],
    strategy: AllocationStrategy | str = AllocationStrategy.UNIFORM,
    *,
    archive_sha256: str | None = None,
    captured_at_utc: str | None = None,
) -> BitAllocationResult:
    """Allocate ``total_bits`` across pairs by ``difficulty_per_pair`` + strategy.

    Args:
        total_bits: non-negative integer; sum of bits to distribute.
        difficulty_per_pair: pair_index -> non-negative finite difficulty.
            Empty mapping raises BitAllocationStrategyError.
        strategy: AllocationStrategy member OR canonical string
            (``"uniform"`` / ``"linear"`` / ``"sqrt"``).
        archive_sha256: optional contest archive sha256 for Provenance
            attribution (recorded but does NOT promote the allocation).
        captured_at_utc: optional ISO-8601 timestamp; defaults to now.

    Returns:
        Frozen :class:`BitAllocationResult` with sum(bits_per_pair) == total_bits
        and Catalog #323 PREDICTED Provenance.

    Raises:
        BitAllocationStrategyError: on any malformed input or invariant
            violation.

    Notes:
        - Uses largest-remainder (Hamilton) rounding for deterministic
          integer allocation with no drift between strategies.
        - When total_bits == 0, all pairs receive 0 bits.
        - When all difficulties are zero (or strategy=UNIFORM), bits are
          distributed equally; remainder goes to lowest-indexed pairs.
    """
    if isinstance(total_bits, bool) or not isinstance(total_bits, int):
        raise BitAllocationStrategyError(
            f"total_bits must be int, got {type(total_bits).__name__}"
        )
    if total_bits < 0:
        raise BitAllocationStrategyError(
            f"total_bits must be non-negative, got {total_bits}"
        )
    if isinstance(strategy, str):
        try:
            strategy_enum = AllocationStrategy(strategy)
        except ValueError as exc:
            raise BitAllocationStrategyError(
                f"unknown strategy string: {strategy!r}"
            ) from exc
    elif isinstance(strategy, AllocationStrategy):
        strategy_enum = strategy
    else:
        raise BitAllocationStrategyError(
            f"strategy must be AllocationStrategy or str, got {type(strategy).__name__}"
        )

    pair_keys, diffs = _normalize_difficulty(difficulty_per_pair)
    n_pairs = len(pair_keys)
    weights = _compute_raw_weights(diffs, strategy_enum)
    allocations = _largest_remainder(total_bits, weights)
    bits_per_pair = {int(k): int(b) for k, b in zip(pair_keys, allocations, strict=True)}

    inputs_sha = _build_inputs_sha256(
        pair_keys, diffs, strategy_enum, total_bits, archive_sha256
    )
    if captured_at_utc is None:
        captured_at_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")
    provenance = build_provenance_for_predicted(
        model_id=CANONICAL_MODEL_ID,
        inputs_sha256=inputs_sha,
        measurement_axis="[predicted]",
        hardware_substrate="unknown",
        captured_at_utc=captured_at_utc,
    )

    notes = {
        "model_id": CANONICAL_MODEL_ID,
        "strategy": strategy_enum.value,
        "n_pairs": int(n_pairs),
        "total_difficulty_sum": float(sum(diffs)),
        "total_weight_sum": float(sum(weights)),
        "archive_sha256_prefix": (
            archive_sha256[:12] if isinstance(archive_sha256, str) else None
        ),
        "axis_tag": "[predicted]",
        "score_claim": False,
        "promotion_eligible": False,
    }

    return BitAllocationResult(
        bits_per_pair=bits_per_pair,
        strategy=strategy_enum,
        total_bits=int(total_bits),
        n_pairs=int(n_pairs),
        provenance=provenance,
        score_claim=False,
        promotion_eligible=False,
        axis_tag="[predicted]",
        notes=notes,
    )


__all__ = (
    "AllocationStrategy",
    "BitAllocationResult",
    "BitAllocationStrategyError",
    "CANONICAL_MODEL_ID",
    "allocate_bits_per_pair",
)
