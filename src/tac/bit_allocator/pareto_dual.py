# SPDX-License-Identifier: MIT
"""Pareto-feasibility Lagrangian-dual bit allocator (Catalog #125 hook #2 + #3).

Canonical bit-allocator helper that solves the multi-constraint bit-allocation
problem via Lagrangian dual decomposition + bisection. Per CLAUDE.md
"Meta-Lagrangian/Pareto solver" non-negotiable + Dykstra co-lead on the
inner council quintet pact: "alternating projections onto rate / seg /
pose / archive-size feasible sets compute the achievable Pareto frontier."

The canonical problem statement:

    maximize    sum_i u_i(b_i)                 # element utility
    subject to  sum_i b_i <= total_budget_bits  # primary budget
                b_i in [min_bits, max_bits]     # per-element box

where ``u_i(b_i) = sensitivity_i * log2(1 + b_i)`` is the canonical
concave element-utility (sqrt-rate-distortion in the small-bits regime;
Shannon-source-coding inspired). Concavity guarantees a unique optimum
satisfying KKT, which can be solved by bisecting the Lagrangian
multiplier ``lambda`` such that the budget constraint is met.

The KKT condition for the relaxed problem:

    u_i'(b_i*) = lambda    if min_bits < b_i* < max_bits
    u_i'(b_i*) <= lambda   if b_i* == min_bits
    u_i'(b_i*) >= lambda   if b_i* == max_bits

For ``u_i(b) = s_i * log2(1 + b)``:

    u_i'(b) = s_i / ((1 + b) * ln 2)

So at interior solutions:

    b_i*(lambda) = clamp(s_i / (lambda * ln 2) - 1, min_bits, max_bits)

The allocator bisects ``lambda`` in [eps, max_grad] until the integer-
rounded budget equals (or is the largest <=) ``total_budget_bits``.

The Lagrangian dual surface integrates with
:mod:`tac.findings_lagrangian` at the META Lagrangian level: this
allocator's per-element Lagrangian multiplier is the bit-allocator's
projection of the global meta-Lagrangian per :func:`tac.findings_lagrangian.compute_findings_lagrangian`.
The integration is observability-only at this phase (Phase 1
META-LAGRANGIAN-WIRE-1 wires the canonical invocation into the cathedral
autopilot; this bit-allocator's outputs feed that future loop).

Per CLAUDE.md "Subagent coherence-by-default" non-negotiable, this module:

* hook #1 sensitivity-map = ACTIVE (consumes per-element sensitivity)
* hook #2 Pareto constraint = **PRIMARY** (total_budget_bits + box constraints
  AND the per-element KKT condition that defines the achievable Pareto frontier)
* hook #3 bit-allocator = **PRIMARY**
* hook #4 cathedral autopilot dispatch = ACTIVE (downstream consumers use plan
  to make ranking adjustments via canonical reward calculation)
* hook #5 continual-learning posterior = ACTIVE (Pareto-dual residual feeds
  back into ``tac.findings_lagrangian.posterior_update_from_anchors``)
* hook #6 probe-disambiguator = ACTIVE (DYKSTRA_PROJECTION vs LAGRANGIAN_DUAL
  enum exposes the choice between coordinate descent and bisection)

Per Catalog #323 every ParetoDualBitAllocationPlan carries canonical
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


CANONICAL_MODEL_ID_PARETO_DUAL = "tac.bit_allocator.pareto_dual.v1"
"""Canonical model identifier for Provenance attribution."""

DEFAULT_BISECTION_ITERS = 64
"""Default Lagrangian bisection iteration count (~1e-19 precision)."""

LN_2 = math.log(2.0)


class ParetoDualMethod(Enum):
    """Canonical Pareto-dual allocation method (Catalog #125 hook #6).

    Members:
        LAGRANGIAN_DUAL: bisect the global Lagrangian multiplier such that
            the rounded integer budget equals total_budget_bits. Canonical
            for single-constraint problems.
        DYKSTRA_PROJECTION: alternate projection onto box + budget
            constraints; canonical for multi-constraint Pareto problems
            per Dykstra co-lead inner council pact.
    """

    LAGRANGIAN_DUAL = "lagrangian_dual"
    DYKSTRA_PROJECTION = "dykstra_projection"


class ParetoDualError(ValueError):
    """Raised for an invalid input to :func:`allocate_via_lagrangian_dual`."""


@dataclass(frozen=True)
class ParetoDualBitAllocationPlan:
    """Canonical frozen Pareto-dual bit-allocation manifest.

    Per Catalog #323 every score-claim-adjacent payload carries Provenance.
    The bit-allocator output is observability-only (PREDICTED grade).

    Attributes:
        bits_per_element: element_id -> bit count; sums to <= total_budget_bits.
        method: the :class:`ParetoDualMethod` used.
        total_budget_bits: input Pareto constraint.
        residual_bits: total_budget_bits - sum(bits_per_element); always >= 0.
        min_bits: per-element floor.
        max_bits: per-element ceiling.
        lagrangian_lambda: the KKT dual variable found by bisection (or 0.0
            for DYKSTRA_PROJECTION).
        kkt_residual: the KKT residual at the dual solution; smaller means
            tighter optimality (0 = exact KKT satisfied).
        n_elements: count of elements in scope.
        is_pareto_feasible: True iff sum(bits) <= total_budget_bits AND
            every bit in [min_bits, max_bits]. False indicates an
            infeasibility that the operator must resolve.
        provenance: Catalog #323 canonical Provenance object.
        score_claim: ALWAYS False.
        promotion_eligible: ALWAYS False.
        axis_tag: ALWAYS "[predicted]".
        notes: optional diagnostic dict.
    """

    bits_per_element: Mapping[int, int]
    method: ParetoDualMethod
    total_budget_bits: int
    residual_bits: int
    min_bits: int
    max_bits: int
    lagrangian_lambda: float
    kkt_residual: float
    n_elements: int
    is_pareto_feasible: bool
    provenance: Provenance
    score_claim: bool = False
    promotion_eligible: bool = False
    axis_tag: str = "[predicted]"
    notes: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Catalog #323 non-promotable invariants + KKT contract."""
        if self.score_claim is not False:
            raise ParetoDualError(
                "ParetoDualBitAllocationPlan.score_claim must be False (Catalog #323)"
            )
        if self.promotion_eligible is not False:
            raise ParetoDualError(
                "promotion_eligible must be False (Catalog #323)"
            )
        if self.axis_tag != "[predicted]":
            raise ParetoDualError(
                "axis_tag must be '[predicted]' (Catalog #287/#323)"
            )
        if self.provenance.evidence_grade != ProvenanceEvidenceGrade.PREDICTED:
            raise ParetoDualError(
                "provenance.evidence_grade must be PREDICTED"
            )
        if self.provenance.artifact_kind != ProvenanceKind.PREDICTED_FROM_MODEL:
            raise ParetoDualError(
                "provenance.artifact_kind must be PREDICTED_FROM_MODEL"
            )
        if self.total_budget_bits < 0:
            raise ParetoDualError(
                f"total_budget_bits must be non-negative, got {self.total_budget_bits}"
            )
        if self.residual_bits < 0:
            raise ParetoDualError(
                f"residual_bits must be non-negative, got {self.residual_bits}"
            )
        if self.min_bits < 0 or self.max_bits < self.min_bits:
            raise ParetoDualError(
                f"invalid bit range [{self.min_bits}, {self.max_bits}]"
            )
        if self.lagrangian_lambda < 0.0:
            raise ParetoDualError(
                f"lagrangian_lambda must be non-negative, got {self.lagrangian_lambda}"
            )
        if not math.isfinite(self.kkt_residual) or self.kkt_residual < 0.0:
            raise ParetoDualError(
                f"kkt_residual must be non-negative finite, got {self.kkt_residual}"
            )
        if self.n_elements != len(self.bits_per_element):
            raise ParetoDualError(
                f"n_elements={self.n_elements} != "
                f"len(bits_per_element)={len(self.bits_per_element)}"
            )
        actual_sum = sum(self.bits_per_element.values())
        expected_sum = self.total_budget_bits - self.residual_bits
        if actual_sum != expected_sum:
            raise ParetoDualError(
                f"sum(bits_per_element)={actual_sum} != "
                f"total-residual={expected_sum}"
            )
        for element_id, bits in self.bits_per_element.items():
            if not isinstance(element_id, int) or isinstance(element_id, bool):
                raise ParetoDualError(
                    f"element_id must be int, got {type(element_id).__name__}"
                )
            if not isinstance(bits, int) or isinstance(bits, bool):
                raise ParetoDualError(
                    f"bits must be int, got {type(bits).__name__}"
                )
            if bits < self.min_bits or bits > self.max_bits:
                if self.is_pareto_feasible:
                    raise ParetoDualError(
                        f"bits[{element_id}]={bits} out of "
                        f"[{self.min_bits}, {self.max_bits}] "
                        f"but is_pareto_feasible=True (inconsistent)"
                    )

    def as_dict(self) -> dict[str, object]:
        """JSON-serializable view."""
        return {
            "bits_per_element": {
                str(k): int(v) for k, v in self.bits_per_element.items()
            },
            "method": self.method.value,
            "total_budget_bits": int(self.total_budget_bits),
            "residual_bits": int(self.residual_bits),
            "min_bits": int(self.min_bits),
            "max_bits": int(self.max_bits),
            "lagrangian_lambda": float(self.lagrangian_lambda),
            "kkt_residual": float(self.kkt_residual),
            "n_elements": int(self.n_elements),
            "is_pareto_feasible": bool(self.is_pareto_feasible),
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
    sensitivity_per_element: Mapping[int, float],
) -> tuple[tuple[int, ...], tuple[float, ...]]:
    """Validate + return (element_id_tuple, sensitivity_tuple) sorted by id."""
    if not sensitivity_per_element:
        raise ParetoDualError(
            "sensitivity_per_element must contain at least one element"
        )
    keys: list[int] = []
    values: list[float] = []
    for raw_key, raw_val in sensitivity_per_element.items():
        if isinstance(raw_key, bool) or not isinstance(raw_key, int):
            raise ParetoDualError(
                f"element_id must be int, got {type(raw_key).__name__}"
            )
        if isinstance(raw_val, bool):
            raise ParetoDualError(
                f"sensitivity[{raw_key}] must be numeric, got bool"
            )
        try:
            s = float(raw_val)
        except (TypeError, ValueError) as exc:
            raise ParetoDualError(
                f"sensitivity[{raw_key}] must be numeric"
            ) from exc
        if not math.isfinite(s):
            raise ParetoDualError(
                f"sensitivity[{raw_key}] must be finite, got {s!r}"
            )
        if s < 0.0:
            raise ParetoDualError(
                f"sensitivity[{raw_key}] must be non-negative, got {s!r}"
            )
        keys.append(raw_key)
        values.append(s)
    paired = sorted(zip(keys, values, strict=True), key=lambda p: p[0])
    return tuple(k for k, _ in paired), tuple(v for _, v in paired)


def _bits_for_lambda(
    sensitivities: tuple[float, ...],
    lam: float,
    min_bits: int,
    max_bits: int,
) -> tuple[int, ...]:
    """Compute per-element bits at Lagrangian multiplier lam.

    For u_i(b) = s_i * log2(1+b):
        b_i*(lam) = clamp(s_i / (lam * ln 2) - 1, min_bits, max_bits)
    """
    if lam <= 0.0:
        # Degenerate: every element gets max_bits.
        return tuple(max_bits for _ in sensitivities)
    out = []
    for s in sensitivities:
        if s == 0.0:
            out.append(min_bits)
            continue
        # b* = s / (lam * ln2) - 1
        b_star = s / (lam * LN_2) - 1.0
        b_int = max(min_bits, min(max_bits, int(round(b_star))))
        out.append(b_int)
    return tuple(out)


def _solve_lagrangian_bisection(
    sensitivities: tuple[float, ...],
    total_budget_bits: int,
    min_bits: int,
    max_bits: int,
    bisect_iters: int = DEFAULT_BISECTION_ITERS,
) -> tuple[tuple[int, ...], float, float, int]:
    """Bisect lambda to find KKT optimum subject to budget.

    Returns (bits_tuple, lambda_star, kkt_residual, residual_bits).
    """
    n = len(sensitivities)
    floor_total = n * min_bits
    ceiling_total = n * max_bits

    # Edge: budget too tight.
    if floor_total > total_budget_bits:
        raise ParetoDualError(
            f"infeasible: floor={floor_total} > total_budget_bits={total_budget_bits}"
        )

    # Edge: ceiling fits — every element gets max_bits.
    if ceiling_total <= total_budget_bits:
        bits = tuple(max_bits for _ in sensitivities)
        residual = total_budget_bits - ceiling_total
        # KKT residual: lambda = 0 (no budget pressure). Best satisfied at 0.
        return bits, 0.0, 0.0, residual

    # Edge: all sensitivities zero — uniform floor allocation.
    max_s = max(sensitivities)
    if max_s == 0.0:
        bits = tuple(min_bits for _ in sensitivities)
        residual = total_budget_bits - floor_total
        return bits, 0.0, 0.0, residual

    # Bracket lambda for bisection.
    # lambda_lo: small enough that b_i* >= max_bits for everyone with s>0.
    #   s / (lam * ln2) - 1 >= max_bits  =>  lam <= s / ((max_bits+1) * ln2)
    #   lambda_lo = max_s / ((max_bits+1)*ln2 * 100)  (safety margin)
    lambda_lo = max(max_s / ((max_bits + 1) * LN_2 * 100.0), 1e-30)
    # lambda_hi: large enough that b_i* <= min_bits for everyone.
    #   s / (lam * ln2) - 1 <= min_bits  =>  lam >= s / ((min_bits+1) * ln2)
    lambda_hi = max_s / ((min_bits + 1) * LN_2) * 100.0

    # Grow lambda_lo if needed.
    GROWTH_ITERS = 64
    growth = 0
    while growth < GROWTH_ITERS:
        bits = _bits_for_lambda(sensitivities, lambda_lo, min_bits, max_bits)
        if sum(bits) > total_budget_bits:
            # lambda_lo too small, double it down (smaller lambda = more bits).
            # Actually no — we want lambda_lo to give too-many-bits, lambda_hi
            # to give too-few. If sum > budget at lambda_lo, that's correct
            # (low lambda = high bits). Stop growing.
            break
        # Decrease lambda_lo (more bits).
        lambda_lo /= 10.0
        growth += 1
        if lambda_lo < 1e-40:
            break

    growth = 0
    while growth < GROWTH_ITERS:
        bits = _bits_for_lambda(sensitivities, lambda_hi, min_bits, max_bits)
        if sum(bits) <= total_budget_bits:
            # lambda_hi correct (low bits).
            break
        lambda_hi *= 10.0
        growth += 1
        if lambda_hi > 1e40:
            break

    # Bisection: find largest lambda such that sum(bits) <= total_budget_bits.
    best_bits: tuple[int, ...] | None = None
    best_lambda = lambda_hi
    for _ in range(bisect_iters):
        lam_mid = 0.5 * (lambda_lo + lambda_hi)
        bits = _bits_for_lambda(sensitivities, lam_mid, min_bits, max_bits)
        bit_sum = sum(bits)
        if bit_sum <= total_budget_bits:
            best_bits = bits
            best_lambda = lam_mid
            lambda_hi = lam_mid
        else:
            lambda_lo = lam_mid
        if lambda_hi - lambda_lo < 1e-15 * max(lambda_hi, 1e-30):
            break

    if best_bits is None:
        # No feasible lambda found in bracket — fall back to floor.
        best_bits = tuple(min_bits for _ in sensitivities)
        best_lambda = lambda_hi
    residual = total_budget_bits - sum(best_bits)
    # KKT residual: |lambda - mean(u_i'(b_i*))| at interior elements.
    # We approximate by computing the gradient at the rounded allocation
    # and measuring its variance.
    interior_grads: list[float] = []
    for s, b in zip(sensitivities, best_bits, strict=True):
        if min_bits < b < max_bits and s > 0.0:
            interior_grads.append(s / ((1.0 + b) * LN_2))
    if interior_grads:
        mean_grad = sum(interior_grads) / len(interior_grads)
        kkt_residual = (
            sum((g - mean_grad) ** 2 for g in interior_grads) / len(interior_grads)
        ) ** 0.5
    else:
        kkt_residual = 0.0
    return best_bits, best_lambda, kkt_residual, residual


def _dykstra_projection_allocate(
    sensitivities: tuple[float, ...],
    total_budget_bits: int,
    min_bits: int,
    max_bits: int,
    max_iters: int = 256,
) -> tuple[tuple[int, ...], float, int]:
    """Dykstra alternating projection onto box + budget constraints.

    Real-valued projection (not integer); we project to the
    intersection of [min_bits, max_bits]^n and the half-space
    sum(x) <= total_budget_bits, then round to integers at the end.

    Returns (bits_tuple, kkt_residual, residual_bits).
    """
    n = len(sensitivities)
    floor_total = n * min_bits
    if floor_total > total_budget_bits:
        raise ParetoDualError(
            f"infeasible: floor={floor_total} > total_budget_bits={total_budget_bits}"
        )

    # Warm start: proportional-to-sensitivity (clamped).
    total_s = sum(sensitivities)
    if total_s > 0:
        warm = [s / total_s * total_budget_bits for s in sensitivities]
    else:
        warm = [total_budget_bits / n] * n

    x = warm[:]
    p1 = [0.0] * n  # correction for box projection
    p2 = [0.0] * n  # correction for budget projection

    for _ in range(max_iters):
        # Project onto box [min_bits, max_bits]
        y = [max(min_bits, min(max_bits, xi + p1i)) for xi, p1i in zip(x, p1)]
        p1 = [xi + p1i - yi for xi, p1i, yi in zip(x, p1, y)]

        # Project onto budget half-space (sum <= total)
        z_pre = [yi + p2i for yi, p2i in zip(y, p2)]
        total_pre = sum(z_pre)
        if total_pre <= total_budget_bits:
            z = z_pre
        else:
            overshoot = (total_pre - total_budget_bits) / n
            z = [zi - overshoot for zi in z_pre]
        p2 = [yi + p2i - zi for yi, p2i, zi in zip(y, p2, z)]

        # Convergence check.
        diff_l1 = sum(abs(xn - xo) for xn, xo in zip(z, x))
        x = z
        if diff_l1 < 1e-9:
            break

    # Round to integers honoring budget (Hamilton-style: largest fractional).
    floors = [max(min_bits, min(max_bits, int(math.floor(xi)))) for xi in x]
    cur_sum = sum(floors)
    if cur_sum > total_budget_bits:
        # Trim by decreasing some elements (largest negative slack).
        slack = sorted(
            ((floors[i] - min_bits, i) for i in range(n)),
            key=lambda p: (-p[0], p[1]),
        )
        k = cur_sum - total_budget_bits
        idx = 0
        while k > 0 and idx < len(slack):
            avail, i = slack[idx]
            take = min(avail, k)
            floors[i] -= take
            k -= take
            idx += 1
    elif cur_sum < total_budget_bits:
        # Fill by distributing remainder to largest fractions.
        fracs = sorted(
            ((x[i] - floors[i], i) for i in range(n)),
            key=lambda p: (-p[0], p[1]),
        )
        leftover = total_budget_bits - cur_sum
        # Single sweep: cycle through fracs at most twice (first to take
        # all elements with headroom, second to fill any remaining capacity).
        # If everyone is at max_bits there's no place to put leftover —
        # treat it as residual rather than spinning forever.
        for sweep in range(2):
            if leftover <= 0:
                break
            made_progress = False
            for _frac, i in fracs:
                if leftover <= 0:
                    break
                if floors[i] < max_bits:
                    floors[i] += 1
                    leftover -= 1
                    made_progress = True
            if not made_progress:
                break  # everyone at max_bits; leftover becomes residual

    bits = tuple(int(b) for b in floors)
    residual = total_budget_bits - sum(bits)
    # KKT residual: deviation from real-valued projection target.
    kkt_residual = sum((bi - xi) ** 2 for bi, xi in zip(bits, x)) ** 0.5
    return bits, kkt_residual, residual


def _build_inputs_sha256(
    element_keys: tuple[int, ...],
    sensitivities: tuple[float, ...],
    method: ParetoDualMethod,
    total_budget_bits: int,
    min_bits: int,
    max_bits: int,
    archive_sha256: str | None,
) -> str:
    """Deterministic sha256 of the inputs."""
    hasher = hashlib.sha256()
    hasher.update(b"tac.bit_allocator.pareto_dual.v1\n")
    hasher.update(f"method={method.value}\n".encode())
    hasher.update(f"total_budget_bits={int(total_budget_bits)}\n".encode())
    hasher.update(f"min_bits={int(min_bits)}\n".encode())
    hasher.update(f"max_bits={int(max_bits)}\n".encode())
    hasher.update(f"archive_sha256={archive_sha256 or ''}\n".encode())
    hasher.update(f"n_elements={len(element_keys)}\n".encode())
    for key, s in zip(element_keys, sensitivities, strict=True):
        hasher.update(f"{int(key)}:{s!r}\n".encode())
    return hasher.hexdigest()


def allocate_via_lagrangian_dual(
    total_budget_bits: int,
    sensitivity_per_element: Mapping[int, float],
    *,
    method: ParetoDualMethod | str = ParetoDualMethod.LAGRANGIAN_DUAL,
    min_bits: int = 0,
    max_bits: int = 8,
    bisect_iters: int = DEFAULT_BISECTION_ITERS,
    archive_sha256: str | None = None,
    captured_at_utc: str | None = None,
) -> ParetoDualBitAllocationPlan:
    """Allocate bits via Lagrangian dual decomposition (canonical KKT bisection).

    Solves:

        maximize    sum_i s_i * log2(1 + b_i)
        subject to  sum_i b_i <= total_budget_bits
                    b_i in [min_bits, max_bits]

    via Lagrangian relaxation:

        L(b, lambda) = sum_i s_i * log2(1+b_i) - lambda * (sum_i b_i - total_budget_bits)

    Bisecting lambda for the budget constraint produces the KKT-satisfying
    optimum (concavity guarantees uniqueness).

    The DYKSTRA_PROJECTION method instead alternates projections onto box
    + budget feasibility per Dykstra co-lead canonical inner-council pact.

    Args:
        total_budget_bits: non-negative integer; total bit budget.
        sensitivity_per_element: element_id -> non-negative finite sensitivity.
        method: :class:`ParetoDualMethod` or canonical string.
        min_bits: per-element floor (default 0).
        max_bits: per-element ceiling (default 8 = full byte).
        bisect_iters: number of bisection iterations.
        archive_sha256: optional contest archive sha256 for Provenance.
        captured_at_utc: optional ISO-8601 timestamp.

    Returns:
        Frozen :class:`ParetoDualBitAllocationPlan` with KKT residual + dual
        variable + Pareto-feasibility verdict + canonical Provenance.

    Raises:
        ParetoDualError: on malformed input or infeasibility.
    """
    if isinstance(total_budget_bits, bool) or not isinstance(total_budget_bits, int):
        raise ParetoDualError(
            f"total_budget_bits must be int, got {type(total_budget_bits).__name__}"
        )
    if total_budget_bits < 0:
        raise ParetoDualError(
            f"total_budget_bits must be non-negative, got {total_budget_bits}"
        )
    if isinstance(min_bits, bool) or not isinstance(min_bits, int):
        raise ParetoDualError(
            f"min_bits must be int, got {type(min_bits).__name__}"
        )
    if isinstance(max_bits, bool) or not isinstance(max_bits, int):
        raise ParetoDualError(
            f"max_bits must be int, got {type(max_bits).__name__}"
        )
    if min_bits < 0 or max_bits < min_bits:
        raise ParetoDualError(
            f"invalid bit range [min={min_bits}, max={max_bits}]"
        )
    if isinstance(bisect_iters, bool) or not isinstance(bisect_iters, int):
        raise ParetoDualError(
            f"bisect_iters must be int, got {type(bisect_iters).__name__}"
        )
    if bisect_iters <= 0:
        raise ParetoDualError(
            f"bisect_iters must be positive, got {bisect_iters}"
        )

    if isinstance(method, str):
        try:
            method_enum = ParetoDualMethod(method)
        except ValueError as exc:
            raise ParetoDualError(
                f"unknown method string: {method!r}"
            ) from exc
    elif isinstance(method, ParetoDualMethod):
        method_enum = method
    else:
        raise ParetoDualError(
            f"method must be ParetoDualMethod or str, "
            f"got {type(method).__name__}"
        )

    element_keys, sensitivities = _normalize_sensitivity(sensitivity_per_element)
    n_elements = len(element_keys)

    lagrangian_lambda = 0.0
    if method_enum is ParetoDualMethod.LAGRANGIAN_DUAL:
        bits_tuple, lagrangian_lambda, kkt_residual, residual = _solve_lagrangian_bisection(
            sensitivities, total_budget_bits, min_bits, max_bits, bisect_iters
        )
    elif method_enum is ParetoDualMethod.DYKSTRA_PROJECTION:
        bits_tuple, kkt_residual, residual = _dykstra_projection_allocate(
            sensitivities, total_budget_bits, min_bits, max_bits
        )
    else:  # pragma: no cover - defensive
        raise ParetoDualError(f"unknown method: {method_enum!r}")

    bits_per_element = {
        int(k): int(b) for k, b in zip(element_keys, bits_tuple, strict=True)
    }

    # Pareto feasibility check: sum within budget AND every bit in box.
    is_pareto_feasible = (
        sum(bits_per_element.values()) <= total_budget_bits
        and all(min_bits <= b <= max_bits for b in bits_per_element.values())
    )

    inputs_sha = _build_inputs_sha256(
        element_keys,
        sensitivities,
        method_enum,
        total_budget_bits,
        min_bits,
        max_bits,
        archive_sha256,
    )
    if captured_at_utc is None:
        captured_at_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")
    provenance = build_provenance_for_predicted(
        model_id=CANONICAL_MODEL_ID_PARETO_DUAL,
        inputs_sha256=inputs_sha,
        measurement_axis="[predicted]",
        hardware_substrate="unknown",
        captured_at_utc=captured_at_utc,
    )

    notes = {
        "model_id": CANONICAL_MODEL_ID_PARETO_DUAL,
        "method": method_enum.value,
        "min_bits": int(min_bits),
        "max_bits": int(max_bits),
        "lagrangian_lambda": float(lagrangian_lambda),
        "kkt_residual": float(kkt_residual),
        "bisect_iters": int(bisect_iters),
        "is_pareto_feasible": bool(is_pareto_feasible),
        "archive_sha256_prefix": (
            archive_sha256[:12] if isinstance(archive_sha256, str) else None
        ),
        "axis_tag": "[predicted]",
        "score_claim": False,
        "promotion_eligible": False,
        # Cite the broader META Lagrangian per CLAUDE.md non-negotiable.
        "meta_lagrangian_reference": "tac.findings_lagrangian.compute_findings_lagrangian",
    }

    return ParetoDualBitAllocationPlan(
        bits_per_element=bits_per_element,
        method=method_enum,
        total_budget_bits=int(total_budget_bits),
        residual_bits=int(residual),
        min_bits=int(min_bits),
        max_bits=int(max_bits),
        lagrangian_lambda=float(lagrangian_lambda),
        kkt_residual=float(kkt_residual),
        n_elements=int(n_elements),
        is_pareto_feasible=bool(is_pareto_feasible),
        provenance=provenance,
        score_claim=False,
        promotion_eligible=False,
        axis_tag="[predicted]",
        notes=notes,
    )


__all__ = (
    "CANONICAL_MODEL_ID_PARETO_DUAL",
    "DEFAULT_BISECTION_ITERS",
    "ParetoDualBitAllocationPlan",
    "ParetoDualError",
    "ParetoDualMethod",
    "allocate_via_lagrangian_dual",
)
