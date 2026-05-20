# SPDX-License-Identifier: MIT
"""Per-byte bit allocator (top-K sensitivity-weighted; Catalog #125 hook #3).

Canonical bit-allocator helper that consumes per-byte sensitivity scores
(e.g. from ``tac.master_gradient_consumers`` or
``tac.cathedral_consumers.per_byte_sensitivity_consumer``) and allocates a
total bit budget across byte offsets via TOP_K_BY_SENSITIVITY greedy + a
canonical UNIFORM_BASELINE for benchmarking.

The canonical input prior per CLAUDE.md "Canonical equations + models
registry" is the equation
``per_byte_leverage_uniformly_distributed_v1`` (see
:mod:`tac.canonical_equations.builtins`). That equation establishes the
empirical anchor *top-K leverage scales near-linearly with K (mild Pareto
concentration: top-1% leverage is ~6.4% for PR101)*. This module's
TOP_K_BY_SENSITIVITY method honors that equation by selecting the K most-
sensitive bytes greedily; UNIFORM_BASELINE is the null hypothesis the
equation predicts.

Per CLAUDE.md "Subagent coherence-by-default" non-negotiable, this module:

* hook #1 sensitivity-map = ACTIVE (consumes per-byte sensitivity)
* hook #2 Pareto constraint = ACTIVE (total_bits is the Pareto bound)
* hook #3 bit-allocator = **PRIMARY**
* hook #4 cathedral autopilot dispatch = ACTIVE via
  ``per_byte_sensitivity_consumer``
* hook #5 continual-learning posterior = ACTIVE
  (``per_byte_leverage_uniformly_distributed_v1`` recalibrates on new anchors)
* hook #6 probe-disambiguator = ACTIVE (AllocationMethod enum exposes the choice)

Per Catalog #323 every BitAllocationPlan carries canonical Provenance with
``evidence_grade=PREDICTED`` + ``score_claim=False`` +
``axis_tag="[predicted]"``. The bit-allocator output is observability-only;
realizing it as actual archive bytes + paired-axis Linux x86_64 auth-eval
per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA" is required
before any contest score claim.
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


CANONICAL_MODEL_ID_PER_BYTE = "tac.bit_allocator.per_byte.v1"
"""Canonical model identifier for Provenance attribution."""

CANONICAL_EQUATION_ID = "per_byte_leverage_uniformly_distributed_v1"
"""Cited canonical equation per CLAUDE.md "Canonical equations" non-negotiable."""


class PerByteAllocationMethod(Enum):
    """Canonical per-byte allocation method enum (Catalog #125 hook #6).

    Members:
        TOP_K_BY_SENSITIVITY: greedy assign each of the top-K bytes the
            maximum per-byte bit cap (default 8); residual bits go to the
            top remaining bytes one-by-one until budget exhausts.
        UNIFORM_BASELINE: every byte index in scope gets ``total_bits // N``
            bits; remainder distributed to lowest-index bytes. This is the
            null hypothesis of equation
            ``per_byte_leverage_uniformly_distributed_v1``.
    """

    TOP_K_BY_SENSITIVITY = "top_k_by_sensitivity"
    UNIFORM_BASELINE = "uniform_baseline"


class PerByteAllocationError(ValueError):
    """Raised for an invalid input to :func:`allocate_per_byte`."""


@dataclass(frozen=True)
class PerByteAllocationPlan:
    """Canonical frozen per-byte bit-allocation manifest.

    Per Catalog #323 every score-claim-adjacent payload carries Provenance.
    The bit-allocator output is observability-only (PREDICTED grade).

    Attributes:
        bits_per_byte: byte_offset -> bit count (sums to ``total_budget_bits
            - residual_bits``)
        method: the :class:`PerByteAllocationMethod` used
        total_budget_bits: input Pareto constraint
        residual_bits: bits unspent (constraint slack); always >= 0
        n_bytes_allocated: count of byte offsets in plan (>= top_k for
            TOP_K mode, == n_in_scope for UNIFORM mode)
        n_bytes_in_scope: count of byte offsets supplied in input sensitivity
        per_byte_bit_cap: max bits per byte (default 8)
        canonical_equation_id: ``per_byte_leverage_uniformly_distributed_v1``
        provenance: Catalog #323 canonical Provenance object
        score_claim: ALWAYS False (per Catalog #323 invariant)
        promotion_eligible: ALWAYS False (per Catalog #323 invariant)
        axis_tag: ALWAYS "[predicted]" (per Catalog #287/#323)
        notes: optional diagnostic dict
    """

    bits_per_byte: Mapping[int, int]
    method: PerByteAllocationMethod
    total_budget_bits: int
    residual_bits: int
    n_bytes_allocated: int
    n_bytes_in_scope: int
    per_byte_bit_cap: int
    canonical_equation_id: str
    provenance: Provenance
    score_claim: bool = False
    promotion_eligible: bool = False
    axis_tag: str = "[predicted]"
    notes: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Catalog #323 non-promotable invariants."""
        if self.score_claim is not False:
            raise PerByteAllocationError(
                "PerByteAllocationPlan.score_claim must be False (Catalog #323)"
            )
        if self.promotion_eligible is not False:
            raise PerByteAllocationError(
                "PerByteAllocationPlan.promotion_eligible must be False (Catalog #323)"
            )
        if self.axis_tag != "[predicted]":
            raise PerByteAllocationError(
                "PerByteAllocationPlan.axis_tag must be '[predicted]' (Catalog #287/#323)"
            )
        if self.provenance.evidence_grade != ProvenanceEvidenceGrade.PREDICTED:
            raise PerByteAllocationError(
                "PerByteAllocationPlan.provenance.evidence_grade must be PREDICTED"
            )
        if self.provenance.artifact_kind != ProvenanceKind.PREDICTED_FROM_MODEL:
            raise PerByteAllocationError(
                "PerByteAllocationPlan.provenance.artifact_kind must be PREDICTED_FROM_MODEL"
            )
        if self.residual_bits < 0:
            raise PerByteAllocationError(
                f"residual_bits must be non-negative, got {self.residual_bits}"
            )
        if self.total_budget_bits < 0:
            raise PerByteAllocationError(
                f"total_budget_bits must be non-negative, got {self.total_budget_bits}"
            )
        if self.per_byte_bit_cap <= 0:
            raise PerByteAllocationError(
                f"per_byte_bit_cap must be positive, got {self.per_byte_bit_cap}"
            )
        actual_sum = sum(self.bits_per_byte.values())
        expected_sum = self.total_budget_bits - self.residual_bits
        if actual_sum != expected_sum:
            raise PerByteAllocationError(
                f"sum(bits_per_byte)={actual_sum} != total-residual={expected_sum}"
            )
        if self.n_bytes_allocated != len(self.bits_per_byte):
            raise PerByteAllocationError(
                f"n_bytes_allocated={self.n_bytes_allocated} != "
                f"len(bits_per_byte)={len(self.bits_per_byte)}"
            )
        if self.n_bytes_in_scope < self.n_bytes_allocated:
            raise PerByteAllocationError(
                f"n_bytes_in_scope={self.n_bytes_in_scope} < "
                f"n_bytes_allocated={self.n_bytes_allocated}"
            )
        if self.canonical_equation_id != CANONICAL_EQUATION_ID:
            raise PerByteAllocationError(
                f"canonical_equation_id must be {CANONICAL_EQUATION_ID!r}, "
                f"got {self.canonical_equation_id!r}"
            )
        for byte_offset, bits in self.bits_per_byte.items():
            if not isinstance(byte_offset, int) or isinstance(byte_offset, bool):
                raise PerByteAllocationError(
                    f"byte_offset must be int, got {type(byte_offset).__name__}"
                )
            if byte_offset < 0:
                raise PerByteAllocationError(
                    f"byte_offset must be non-negative, got {byte_offset}"
                )
            if not isinstance(bits, int) or isinstance(bits, bool):
                raise PerByteAllocationError(
                    f"bits must be int, got {type(bits).__name__}"
                )
            if bits < 0:
                raise PerByteAllocationError(
                    f"bits for byte {byte_offset} is negative: {bits}"
                )
            if bits > self.per_byte_bit_cap:
                raise PerByteAllocationError(
                    f"bits for byte {byte_offset}={bits} > cap={self.per_byte_bit_cap}"
                )

    def as_dict(self) -> dict[str, object]:
        """JSON-serializable view (canonical manifest emission)."""
        return {
            "bits_per_byte": {str(k): int(v) for k, v in self.bits_per_byte.items()},
            "method": self.method.value,
            "total_budget_bits": int(self.total_budget_bits),
            "residual_bits": int(self.residual_bits),
            "n_bytes_allocated": int(self.n_bytes_allocated),
            "n_bytes_in_scope": int(self.n_bytes_in_scope),
            "per_byte_bit_cap": int(self.per_byte_bit_cap),
            "canonical_equation_id": self.canonical_equation_id,
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


def _normalize_sensitivity(
    sensitivity_per_byte: Mapping[int, float],
) -> tuple[tuple[int, ...], tuple[float, ...]]:
    """Validate + return (byte_offset_tuple, sensitivity_tuple) sorted by offset."""
    if not sensitivity_per_byte:
        raise PerByteAllocationError(
            "sensitivity_per_byte must contain at least one byte"
        )
    keys: list[int] = []
    values: list[float] = []
    for raw_key, raw_val in sensitivity_per_byte.items():
        if isinstance(raw_key, bool) or not isinstance(raw_key, int):
            raise PerByteAllocationError(
                f"byte_offset must be int, got {type(raw_key).__name__}"
            )
        if raw_key < 0:
            raise PerByteAllocationError(
                f"byte_offset must be non-negative, got {raw_key}"
            )
        if isinstance(raw_val, bool):
            raise PerByteAllocationError(
                f"sensitivity[{raw_key}] must be numeric, got bool"
            )
        try:
            s = float(raw_val)
        except (TypeError, ValueError) as exc:
            raise PerByteAllocationError(
                f"sensitivity[{raw_key}] must be numeric"
            ) from exc
        if not math.isfinite(s):
            raise PerByteAllocationError(
                f"sensitivity[{raw_key}] must be finite, got {s!r}"
            )
        if s < 0.0:
            raise PerByteAllocationError(
                f"sensitivity[{raw_key}] must be non-negative, got {s!r}"
            )
        keys.append(raw_key)
        values.append(s)
    paired = sorted(zip(keys, values, strict=True), key=lambda p: p[0])
    return tuple(k for k, _ in paired), tuple(v for _, v in paired)


def _build_inputs_sha256(
    byte_keys: tuple[int, ...],
    sensitivities: tuple[float, ...],
    method: PerByteAllocationMethod,
    total_budget_bits: int,
    per_byte_bit_cap: int,
    top_k: int,
    archive_sha256: str | None,
) -> str:
    """Deterministic sha256 of the inputs."""
    hasher = hashlib.sha256()
    hasher.update(b"tac.bit_allocator.per_byte.v1\n")
    hasher.update(f"method={method.value}\n".encode())
    hasher.update(f"total_budget_bits={int(total_budget_bits)}\n".encode())
    hasher.update(f"per_byte_bit_cap={int(per_byte_bit_cap)}\n".encode())
    hasher.update(f"top_k={int(top_k)}\n".encode())
    hasher.update(f"archive_sha256={archive_sha256 or ''}\n".encode())
    hasher.update(f"n_bytes={len(byte_keys)}\n".encode())
    for key, sens in zip(byte_keys, sensitivities, strict=True):
        hasher.update(f"{int(key)}:{sens!r}\n".encode())
    return hasher.hexdigest()


def _allocate_uniform(
    byte_keys: tuple[int, ...],
    total_budget_bits: int,
    per_byte_bit_cap: int,
) -> tuple[dict[int, int], int]:
    """UNIFORM_BASELINE allocator.

    Distributes ``total_budget_bits`` equally across all byte offsets in
    scope; remainder routed to lowest-index bytes. Each byte capped at
    ``per_byte_bit_cap``. Returns (bits_per_byte_dict, residual_bits).
    """
    n = len(byte_keys)
    if n == 0:
        return {}, total_budget_bits

    base = total_budget_bits // n
    remainder = total_budget_bits - base * n

    # Cap base at per_byte_bit_cap and route leftover to residual.
    if base > per_byte_bit_cap:
        base = per_byte_bit_cap
        remainder = total_budget_bits - base * n  # may now be > per_byte_bit_cap

    bits = [base] * n
    # Distribute remainder while honoring per-byte cap.
    residual = 0
    for i in range(remainder):
        # remainder is distributed across bytes; if all bytes are at cap,
        # leftover becomes residual.
        idx = i % n
        if bits[idx] < per_byte_bit_cap:
            bits[idx] += 1
        else:
            residual += 1

    bits_per_byte = {int(byte_keys[i]): int(bits[i]) for i in range(n)}
    return bits_per_byte, residual


def _allocate_top_k_by_sensitivity(
    byte_keys: tuple[int, ...],
    sensitivities: tuple[float, ...],
    top_k: int,
    total_budget_bits: int,
    per_byte_bit_cap: int,
) -> tuple[dict[int, int], int]:
    """TOP_K_BY_SENSITIVITY allocator.

    Greedy: assign each of the top-K most-sensitive bytes the maximum
    per-byte bit cap. If budget allows, distribute residual bits to the
    next-most-sensitive bytes one at a time. Returns (bits_per_byte_dict,
    residual_bits).

    Tie-break for equal sensitivity: lower byte_offset wins (determinism).
    """
    n = len(byte_keys)
    if n == 0:
        return {}, total_budget_bits

    # Sort by sensitivity descending (tie-break: byte_offset ascending).
    ranked = sorted(
        zip(byte_keys, sensitivities, strict=True),
        key=lambda pair: (-pair[1], pair[0]),
    )

    bits_per_byte: dict[int, int] = {}
    remaining_budget = total_budget_bits

    # Phase 1: assign per_byte_bit_cap to top_k bytes (greedy).
    k_actual = min(top_k, n)
    for i in range(k_actual):
        byte_offset, _sensitivity = ranked[i]
        assign = min(per_byte_bit_cap, remaining_budget)
        if assign > 0:
            bits_per_byte[int(byte_offset)] = int(assign)
            remaining_budget -= assign

    # Phase 2: residual distribution beyond top_k by sensitivity.
    # Keep filling next-ranked bytes one bit at a time.
    # This handles total_budget_bits > top_k * per_byte_bit_cap.
    idx = k_actual
    while remaining_budget > 0 and idx < n:
        byte_offset, _sensitivity = ranked[idx]
        assign = min(per_byte_bit_cap, remaining_budget)
        if assign > 0:
            bits_per_byte[int(byte_offset)] = int(assign)
            remaining_budget -= assign
        idx += 1

    return bits_per_byte, remaining_budget


def allocate_per_byte(
    total_budget_bits: int,
    sensitivity_per_byte: Mapping[int, float],
    *,
    method: PerByteAllocationMethod | str = PerByteAllocationMethod.TOP_K_BY_SENSITIVITY,
    top_k: int = 32,
    per_byte_bit_cap: int = 8,
    archive_sha256: str | None = None,
    captured_at_utc: str | None = None,
) -> PerByteAllocationPlan:
    """Allocate per-byte bits via top-K sensitivity OR uniform baseline.

    Args:
        total_budget_bits: non-negative integer; total bit budget.
        sensitivity_per_byte: byte_offset -> non-negative finite sensitivity
            (e.g. abs master-gradient magnitude). Empty mapping raises.
        method: :class:`PerByteAllocationMethod` or canonical string
            (``"top_k_by_sensitivity"`` / ``"uniform_baseline"``).
        top_k: only used by TOP_K_BY_SENSITIVITY method; the K most-sensitive
            bytes each receive ``per_byte_bit_cap`` bits before residual
            distribution. Default 32.
        per_byte_bit_cap: maximum bits per byte (default 8 = full byte).
        archive_sha256: optional contest archive sha256 for Provenance.
        captured_at_utc: optional ISO-8601 timestamp.

    Returns:
        Frozen :class:`PerByteAllocationPlan` with canonical Provenance and
        ``canonical_equation_id == "per_byte_leverage_uniformly_distributed_v1"``.

    Raises:
        PerByteAllocationError: on malformed input or invariant violation.
    """
    if isinstance(total_budget_bits, bool) or not isinstance(total_budget_bits, int):
        raise PerByteAllocationError(
            f"total_budget_bits must be int, got {type(total_budget_bits).__name__}"
        )
    if total_budget_bits < 0:
        raise PerByteAllocationError(
            f"total_budget_bits must be non-negative, got {total_budget_bits}"
        )
    if isinstance(per_byte_bit_cap, bool) or not isinstance(per_byte_bit_cap, int):
        raise PerByteAllocationError(
            f"per_byte_bit_cap must be int, got {type(per_byte_bit_cap).__name__}"
        )
    if per_byte_bit_cap <= 0:
        raise PerByteAllocationError(
            f"per_byte_bit_cap must be positive, got {per_byte_bit_cap}"
        )
    if isinstance(top_k, bool) or not isinstance(top_k, int):
        raise PerByteAllocationError(
            f"top_k must be int, got {type(top_k).__name__}"
        )
    if top_k < 0:
        raise PerByteAllocationError(f"top_k must be non-negative, got {top_k}")

    if isinstance(method, str):
        try:
            method_enum = PerByteAllocationMethod(method)
        except ValueError as exc:
            raise PerByteAllocationError(
                f"unknown method string: {method!r}"
            ) from exc
    elif isinstance(method, PerByteAllocationMethod):
        method_enum = method
    else:
        raise PerByteAllocationError(
            f"method must be PerByteAllocationMethod or str, "
            f"got {type(method).__name__}"
        )

    byte_keys, sensitivities = _normalize_sensitivity(sensitivity_per_byte)
    n_in_scope = len(byte_keys)

    if method_enum is PerByteAllocationMethod.UNIFORM_BASELINE:
        bits_per_byte, residual = _allocate_uniform(
            byte_keys, total_budget_bits, per_byte_bit_cap
        )
    elif method_enum is PerByteAllocationMethod.TOP_K_BY_SENSITIVITY:
        bits_per_byte, residual = _allocate_top_k_by_sensitivity(
            byte_keys, sensitivities, top_k, total_budget_bits, per_byte_bit_cap
        )
    else:  # pragma: no cover - defensive
        raise PerByteAllocationError(f"unknown method: {method_enum!r}")

    inputs_sha = _build_inputs_sha256(
        byte_keys,
        sensitivities,
        method_enum,
        total_budget_bits,
        per_byte_bit_cap,
        top_k,
        archive_sha256,
    )
    if captured_at_utc is None:
        captured_at_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")
    provenance = build_provenance_for_predicted(
        model_id=CANONICAL_MODEL_ID_PER_BYTE,
        inputs_sha256=inputs_sha,
        measurement_axis="[predicted]",
        hardware_substrate="unknown",
        captured_at_utc=captured_at_utc,
    )

    notes = {
        "model_id": CANONICAL_MODEL_ID_PER_BYTE,
        "method": method_enum.value,
        "top_k": int(top_k),
        "per_byte_bit_cap": int(per_byte_bit_cap),
        "n_bytes_in_scope": int(n_in_scope),
        "n_bytes_allocated": int(len(bits_per_byte)),
        "archive_sha256_prefix": (
            archive_sha256[:12] if isinstance(archive_sha256, str) else None
        ),
        "canonical_equation_id": CANONICAL_EQUATION_ID,
        "axis_tag": "[predicted]",
        "score_claim": False,
        "promotion_eligible": False,
    }

    return PerByteAllocationPlan(
        bits_per_byte=bits_per_byte,
        method=method_enum,
        total_budget_bits=int(total_budget_bits),
        residual_bits=int(residual),
        n_bytes_allocated=int(len(bits_per_byte)),
        n_bytes_in_scope=int(n_in_scope),
        per_byte_bit_cap=int(per_byte_bit_cap),
        canonical_equation_id=CANONICAL_EQUATION_ID,
        provenance=provenance,
        score_claim=False,
        promotion_eligible=False,
        axis_tag="[predicted]",
        notes=notes,
    )


__all__ = (
    "CANONICAL_EQUATION_ID",
    "CANONICAL_MODEL_ID_PER_BYTE",
    "PerByteAllocationError",
    "PerByteAllocationMethod",
    "PerByteAllocationPlan",
    "allocate_per_byte",
)
