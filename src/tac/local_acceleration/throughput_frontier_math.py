# SPDX-License-Identifier: MIT
"""Exact-reduction and certified-precision laws for throughput experiments.

This module is deliberately backend-free.  It supplies the deterministic
NumPy/Python authority used by the Task #494 frontier-math probe; Metal and ANE
implementations remain owned by the throughput-authority ladder.

The central separation is:

* a reordered reduction is bit-stable when its reachable accumulator states
  form a commutative monoid and the final rounding happens once;
* numerical proximity preserves an argmax only when an interval certificate
  separates the reference winner from every competitor; and
* spatial concentration is not compute sparsity until graph-support closure is
  charged.  A global dependency makes the exact active fraction one.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from functools import reduce
from operator import mul
from typing import Any

import numpy as np


def minimum_signed_bits_for_abs_bound(sum_abs_bound: int) -> int:
    """Return the minimum signed width representing ``[-B, B]`` exactly."""

    if (
        isinstance(sum_abs_bound, bool)
        or not isinstance(sum_abs_bound, int)
        or sum_abs_bound < 0
    ):
        raise ValueError("sum_abs_bound must be a non-negative integer")
    if sum_abs_bound == 0:
        return 1
    # Integer ``bit_length`` keeps the width theorem exact even when ``B`` is
    # larger than the faithfully representable range of a binary64 float.
    return sum_abs_bound.bit_length() + 1


def minimum_signed_accumulator_bits(*, max_abs_term: int, fan_in: int) -> int:
    """Return the minimum two's-complement width that cannot overflow.

    For ``|x_i| <= A`` and at most ``n`` terms, every exact partial sum is in
    ``[-nA, nA]``.  A signed ``w``-bit accumulator is therefore safe exactly
    when ``nA <= 2**(w-1)-1``.
    """

    for name, value in (("max_abs_term", max_abs_term), ("fan_in", fan_in)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
    return minimum_signed_bits_for_abs_bound(fan_in * max_abs_term)


def fixed_width_reduction_certificate(
    *, max_abs_term: int, fan_in: int, accumulator_bits: int
) -> dict[str, int | bool]:
    """Certify whether fixed-width integer addition is exact on the domain."""

    if isinstance(accumulator_bits, bool) or not isinstance(accumulator_bits, int):
        raise ValueError("accumulator_bits must be an integer")
    if accumulator_bits < 1:
        raise ValueError("accumulator_bits must be positive")
    minimum = minimum_signed_accumulator_bits(
        max_abs_term=max_abs_term,
        fan_in=fan_in,
    )
    bound = max_abs_term * fan_in
    return {
        "sum_abs_bound": bound,
        "minimum_signed_bits": minimum,
        "accumulator_bits": accumulator_bits,
        "no_overflow": accumulator_bits >= minimum,
        "all_admissible_partial_sums_exactly_represented": accumulator_bits >= minimum,
    }


def crt_reduction_certificate(
    *, max_abs_term: int, fan_in: int, moduli: tuple[int, ...]
) -> dict[str, int | bool]:
    """Certify exact symmetric CRT reconstruction for every reachable sum.

    Componentwise modular addition is associative and commutative regardless of
    range.  It represents the intended signed integer sum injectively only when
    the product of pairwise-coprime moduli is strictly larger than ``2*n*A``.
    """

    minimum_bits = minimum_signed_accumulator_bits(
        max_abs_term=max_abs_term,
        fan_in=fan_in,
    )
    if not moduli:
        raise ValueError("moduli must be non-empty")
    if any(isinstance(m, bool) or not isinstance(m, int) or m <= 1 for m in moduli):
        raise ValueError("every modulus must be an integer greater than one")
    pairwise_coprime = all(
        math.gcd(a, b) == 1 for a, b in itertools.combinations(moduli, 2)
    )
    modulus_product = reduce(mul, moduli, 1)
    sum_abs_bound = max_abs_term * fan_in
    injective = pairwise_coprime and modulus_product > 2 * sum_abs_bound
    return {
        "sum_abs_bound": sum_abs_bound,
        "information_lower_bound_bits": minimum_bits,
        "modulus_product": modulus_product,
        "pairwise_coprime": pairwise_coprime,
        "symmetric_reconstruction_injective": injective,
        "commutative_monoid_componentwise": True,
    }


def float32_ordered_sum(values: np.ndarray, order: tuple[int, ...]) -> np.float32:
    """Sum one vector in an explicit fp32 order with one rounding per add."""

    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 1 or len(order) != array.size or set(order) != set(range(array.size)):
        raise ValueError("order must be a permutation of a one-dimensional vector")
    if not np.all(np.isfinite(array)):
        raise ValueError("values must be finite")
    total = np.float32(0.0)
    for index in order:
        total = np.float32(total + array[index])
    return total


def integer_ordered_sum(values: tuple[int, ...], order: tuple[int, ...]) -> int:
    """Sum one bounded integer vector in an explicit order using exact Python ints."""

    if len(order) != len(values) or set(order) != set(range(len(values))):
        raise ValueError("order must be a permutation of values")
    if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
        raise ValueError("values must be exact non-boolean integers")
    return sum(values[index] for index in order)


def max_plus_ordered_reduce(
    left: np.ndarray, right: np.ndarray, order: tuple[int, ...]
) -> float:
    """Evaluate ``max_i(left_i + right_i)`` in an explicit reduction order."""

    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    if a.ndim != 1 or b.shape != a.shape:
        raise ValueError("left and right must be equal-length vectors")
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise ValueError("max-plus candidates must be finite")
    if len(order) != a.size or set(order) != set(range(a.size)):
        raise ValueError("order must be a permutation")
    value = -math.inf
    for index in order:
        value = max(value, float(a[index] + b[index]))
    return value


def tropical_argmax_ordered_reduce(
    candidates: tuple[float, ...], order: tuple[int, ...]
) -> tuple[float, int]:
    """Reduce to a deterministic argmax under arbitrary visitation order.

    The carrier is the totally ordered pair ``(value, -class_index)``.  Its
    binary operation is lexicographic ``max``, hence associative, commutative,
    and idempotent.  Returning the negated second component recovers the same
    smallest-index tie policy as NumPy/Torch argmax without relying on which
    equal-valued class an atomic reduction happens to visit first.
    """

    if not candidates:
        raise ValueError("candidates must be non-empty")
    if len(order) != len(candidates) or set(order) != set(range(len(candidates))):
        raise ValueError("order must be a permutation")
    if any(not math.isfinite(value) for value in candidates):
        raise ValueError("candidates must be finite")
    winner = (-math.inf, 0)
    for index in order:
        winner = max(winner, (float(candidates[index]), -index))
    return winner[0], -winner[1]


def number_system_disposition(
    *, max_abs_term: int, fan_in: int, crt_moduli: tuple[int, ...]
) -> tuple[dict[str, Any], ...]:
    """Return the algebraic admission table used by the synthesis probe."""

    fixed = fixed_width_reduction_certificate(
        max_abs_term=max_abs_term,
        fan_in=fan_in,
        accumulator_bits=32,
    )
    crt = crt_reduction_certificate(
        max_abs_term=max_abs_term,
        fan_in=fan_in,
        moduli=crt_moduli,
    )
    return (
        {
            "candidate": "bounded_fixed_point_integer",
            "order_invariant": bool(fixed["no_overflow"]),
            "exact_structure": "commutative integer monoid; one deterministic final rounding",
            "minimality": f"{fixed['minimum_signed_bits']} signed bits for the declared bound",
            "pact_disposition": "FIRST_BUILD",
        },
        {
            "candidate": "kulisch_or_binned_superaccumulator",
            "order_invariant": True,
            "exact_structure": "canonical exact fixed-point bins; one final rounding",
            "minimality": "general-float fallback; wider than bounded fixed point",
            "pact_disposition": "FALLBACK_WHEN_ONE_SCALE_CANNOT_CERTIFY",
        },
        {
            "candidate": "posit_without_quire",
            "order_invariant": False,
            "exact_structure": "rounded posit addition is not associative",
            "minimality": "quire restores a Kulisch-like exact accumulator",
            "pact_disposition": "NO_NATIVE_METAL_ANE_ADVANTAGE",
        },
        {
            "candidate": "kahan_neumaier_or_naive_eft",
            "order_invariant": False,
            "exact_structure": "sequential compensation state is order dependent",
            "minimality": "requires canonical expansion merge to become reproducible",
            "pact_disposition": "NOT_AN_L70_PROOF",
        },
        {
            "candidate": "gaussian_integer",
            "order_invariant": bool(fixed["no_overflow"]),
            "exact_structure": "two bounded integer monoids",
            "minimality": "strict overhead for a real scalar reduction",
            "pact_disposition": "ONLY_FOR_SHARED_COMPLEX_ROTATION_GEOMETRY",
        },
        {
            "candidate": "rational_common_denominator",
            "order_invariant": True,
            "exact_structure": (
                "arbitrary-precision integer numerator after denominator normalization; "
                "bounded hardware words still require an overflow certificate"
            ),
            "minimality": "denominator LCM growth usually exceeds one dyadic scale",
            "pact_disposition": "NORMALIZE_TO_FIXED_POINT_WHEN_POSSIBLE",
        },
        {
            "candidate": "crt_residue_number_system",
            "order_invariant": True,
            "exact_structure": "componentwise commutative modular monoids",
            "minimality": (
                f"product={crt['modulus_product']} must exceed "
                f"2B={2 * int(crt['sum_abs_bound'])}"
            ),
            "pact_disposition": (
                "EXACT_CANDIDATE" if crt["symmetric_reconstruction_injective"] else "RANGE_REFUSED"
            ),
        },
        {
            "candidate": "max_plus_tropical",
            "order_invariant": True,
            "exact_structure": "idempotent commutative max reduction",
            "minimality": "exact only for a graph already expressed over max-plus",
            "pact_disposition": "DECISION_HEAD_ONLY_NOT_DROP_IN_CNN_REPLACEMENT",
        },
    )


@dataclass(frozen=True)
class ArgmaxCertificate:
    """Per-sample interval certificate for a final class axis."""

    reference_winner: np.ndarray
    reference_margin: np.ndarray
    robust_margin: np.ndarray
    certified: np.ndarray

    @property
    def certified_count(self) -> int:
        return int(np.count_nonzero(self.certified))

    @property
    def total_count(self) -> int:
        return int(self.certified.size)

    @property
    def certified_fraction(self) -> float:
        return self.certified_count / self.total_count if self.total_count else 0.0


@dataclass(frozen=True)
class OrdinalConcordanceDiagnostic:
    """Separate the decision-relevant top-1 order from full class ordering."""

    reference_winner: np.ndarray
    candidate_winner: np.ndarray
    top1_preserved: np.ndarray
    full_ordinal_concordance: np.ndarray
    pairwise_order_concordance_fraction: np.ndarray


def ordinal_top1_concordance_diagnostic(
    reference_logits: np.ndarray,
    candidate_logits: np.ndarray,
) -> OrdinalConcordanceDiagnostic:
    """Compare top-1 identity with the strictly stronger full ordinal order.

    Ties use the deterministic smallest-class-index policy.  Equality of the
    complete stable descending permutations implies equal top-1, but the
    converse is false: any loser/loser swap preserves the evaluator decision.
    Thus a recos-style full-ranking statistic is useful diagnostic telemetry,
    not a replacement for the minimal winner-vs-rival interval certificate.
    """

    reference = np.asarray(reference_logits, dtype=np.float64)
    candidate = np.asarray(candidate_logits, dtype=np.float64)
    if reference.shape != candidate.shape or reference.ndim < 1:
        raise ValueError("reference and candidate logits must have the same non-scalar shape")
    if reference.shape[-1] < 2:
        raise ValueError("the class axis must contain at least two classes")
    if not np.all(np.isfinite(reference)) or not np.all(np.isfinite(candidate)):
        raise ValueError("logits must be finite")

    reference_order = np.argsort(-reference, axis=-1, kind="stable")
    candidate_order = np.argsort(-candidate, axis=-1, kind="stable")
    reference_winner = reference_order[..., 0]
    candidate_winner = candidate_order[..., 0]
    agreements: list[np.ndarray] = []
    for left, right in itertools.combinations(range(reference.shape[-1]), 2):
        reference_prefers_left = reference[..., left] >= reference[..., right]
        candidate_prefers_left = candidate[..., left] >= candidate[..., right]
        agreements.append(reference_prefers_left == candidate_prefers_left)
    pairwise_fraction = np.mean(np.stack(agreements, axis=-1), axis=-1)
    return OrdinalConcordanceDiagnostic(
        reference_winner=reference_winner,
        candidate_winner=candidate_winner,
        top1_preserved=reference_winner == candidate_winner,
        full_ordinal_concordance=np.all(reference_order == candidate_order, axis=-1),
        pairwise_order_concordance_fraction=pairwise_fraction,
    )


def winner_rival_margin_hinge(
    logits: np.ndarray,
    target_class: np.ndarray,
    *,
    margin: float = 0.0,
) -> np.ndarray:
    """Return ``max(0, margin - (z_y - max_{c!=y} z_c))`` per sample.

    ``margin=0`` is the DERIVED top-1 decision boundary.  It concentrates the
    loss on target-vs-strongest-rival debt, but grants no positive robustness
    radius and does not itself correct class-frequency imbalance.
    """

    values = np.asarray(logits, dtype=np.float64)
    target = np.asarray(target_class)
    if values.ndim < 1 or values.shape[-1] < 2:
        raise ValueError("logits must have a final class axis of size at least two")
    if target.shape != values.shape[:-1]:
        raise ValueError("target_class must match every non-class logit axis")
    if not np.issubdtype(target.dtype, np.integer) or np.issubdtype(target.dtype, np.bool_):
        raise ValueError("target_class must contain integer class indices")
    if np.any(target < 0) or np.any(target >= values.shape[-1]):
        raise ValueError("target_class contains an out-of-range class index")
    if not np.all(np.isfinite(values)) or not math.isfinite(margin) or margin < 0.0:
        raise ValueError("logits and non-negative margin must be finite")
    target_index = np.expand_dims(target.astype(np.intp, copy=False), axis=-1)
    target_value = np.take_along_axis(values, target_index, axis=-1)[..., 0]
    rivals = values.copy()
    np.put_along_axis(rivals, target_index, -np.inf, axis=-1)
    signed_margin = target_value - np.max(rivals, axis=-1)
    return np.maximum(float(margin) - signed_margin, 0.0)


@dataclass(frozen=True)
class SurfaceTensionMetricCertificate:
    """Metric/lower-semicontinuity certificate for multiphase tensions."""

    triangle_violations: tuple[dict[str, float | int], ...]
    metric_closure: np.ndarray
    gamma_limit_metric_admissible: bool
    spatial_orientation_anisotropy_certified: bool = False


def multiphase_surface_tension_metric_certificate(
    sigma: np.ndarray,
    *,
    tolerance: float = 1e-12,
) -> SurfaceTensionMetricCertificate:
    """Check pair-tension triangle inequalities and return metric closure.

    For the sharp multiphase energy

    ``E = 1/2 sum_ij sigma_ij H^(d-1)(boundary_i intersect boundary_j)``,

    lower semicontinuity requires ``sigma_ik <= sigma_ij + sigma_jk``.
    When it fails, an arbitrarily thin intermediate phase can wet an interface;
    the relaxed pair cost is the all-pairs shortest-path (metric) closure.

    A scalar ``sigma_ij`` multiplies Euclidean perimeter and is anisotropic only
    across *class pairs*.  It contains no dependence on interface normal, so it
    does not by itself certify a spatial Wulff/Finsler anisotropy.
    """

    values = np.asarray(sigma, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != values.shape[1] or values.shape[0] < 2:
        raise ValueError("sigma must be a square matrix with at least two phases")
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("tolerance must be finite and non-negative")
    if not np.all(np.isfinite(values)) or not np.allclose(
        values, values.T, rtol=0.0, atol=tolerance
    ):
        raise ValueError("sigma must be finite and symmetric")
    off_diagonal = ~np.eye(values.shape[0], dtype=bool)
    if np.any(values[off_diagonal] <= 0.0):
        raise ValueError("off-diagonal surface tensions must be positive")

    distances = values.copy()
    np.fill_diagonal(distances, 0.0)
    closure = distances.copy()
    for middle in range(values.shape[0]):
        closure = np.minimum(
            closure,
            closure[:, middle, None] + closure[None, middle, :],
        )
    violations: list[dict[str, float | int]] = []
    for left in range(values.shape[0]):
        for right in range(left + 1, values.shape[0]):
            for middle in range(values.shape[0]):
                if middle in (left, right):
                    continue
                direct = float(distances[left, right])
                via = float(distances[left, middle] + distances[middle, right])
                if direct > via + tolerance:
                    violations.append(
                        {
                            "left": left,
                            "middle": middle,
                            "right": right,
                            "direct": direct,
                            "via": via,
                            "excess": direct - via,
                        }
                    )
    return SurfaceTensionMetricCertificate(
        triangle_violations=tuple(violations),
        metric_closure=closure,
        gamma_limit_metric_admissible=not violations,
    )


def certify_argmax_intervals(
    reference_logits: np.ndarray,
    class_abs_error_bound: np.ndarray | float,
) -> ArgmaxCertificate:
    """Certify reference argmax preservation under independent class intervals.

    With winner ``a``, the condition is

    ``z_a - e_a > max_{c != a}(z_c + e_c)``.

    It is sufficient, and tight given only the independent interval bounds.
    For a uniform bound ``e`` it reduces to ``top1 - top2 > 2e``.
    """

    logits = np.asarray(reference_logits)
    if logits.ndim < 1 or logits.shape[-1] < 2:
        raise ValueError("reference_logits must have a final class axis of size at least two")
    if not np.issubdtype(logits.dtype, np.floating):
        logits = logits.astype(np.float64)
    if not np.all(np.isfinite(logits)):
        raise ValueError("reference_logits must be finite")
    error = np.asarray(class_abs_error_bound, dtype=np.float64)
    try:
        error = np.broadcast_to(error, logits.shape)
    except ValueError as exc:
        raise ValueError("class_abs_error_bound must broadcast to logits") from exc
    if not np.all(np.isfinite(error)) or np.any(error < 0.0):
        raise ValueError("error bounds must be finite and non-negative")

    winner = np.argmax(logits, axis=-1)
    winner_index = np.expand_dims(winner, axis=-1)
    winner_logit = np.take_along_axis(logits, winner_index, axis=-1)[..., 0]
    winner_error = np.take_along_axis(error, winner_index, axis=-1)[..., 0]

    competitor_logits = logits.copy()
    np.put_along_axis(competitor_logits, winner_index, -np.inf, axis=-1)
    top_competitor = np.max(competitor_logits, axis=-1)

    competitor_upper = logits + error
    np.put_along_axis(competitor_upper, winner_index, -np.inf, axis=-1)
    worst_upper = np.max(competitor_upper, axis=-1)
    robust_margin = (winner_logit - winner_error) - worst_upper
    return ArgmaxCertificate(
        reference_winner=winner,
        reference_margin=winner_logit - top_competitor,
        robust_margin=robust_margin,
        certified=robust_margin > 0.0,
    )


@dataclass(frozen=True)
class PrecisionOption:
    bits: int
    error_bound: float
    measured_cost: float
    mode: str = "integer"

    def __post_init__(self) -> None:
        if isinstance(self.bits, bool) or not isinstance(self.bits, int) or self.bits < 1:
            raise ValueError("bits must be a positive integer")
        if (
            isinstance(self.error_bound, bool)
            or not isinstance(self.error_bound, (int, float))
            or not math.isfinite(self.error_bound)
            or self.error_bound < 0.0
        ):
            raise ValueError("error_bound must be finite and non-negative")
        if (
            isinstance(self.measured_cost, bool)
            or not isinstance(self.measured_cost, (int, float))
            or not math.isfinite(self.measured_cost)
            or self.measured_cost < 0.0
        ):
            raise ValueError("measured_cost must be finite and non-negative")
        if not isinstance(self.mode, str) or not self.mode:
            raise ValueError("mode must be a non-empty string")


@dataclass(frozen=True)
class PrecisionLayer:
    name: str
    options: tuple[PrecisionOption, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("layer name must be a non-empty string")
        if not self.options:
            raise ValueError("each layer must have at least one precision option")
        signatures = {(option.bits, option.mode) for option in self.options}
        if len(signatures) != len(self.options):
            raise ValueError("precision option signatures must be unique within a layer")


@dataclass(frozen=True)
class PrecisionAllocation:
    choices: tuple[PrecisionOption, ...]
    total_error_bound: float
    total_measured_cost: float
    pareto_state_count: int


@dataclass(frozen=True)
class _ParetoState:
    error: float
    cost: float
    choices: tuple[PrecisionOption, ...]

    @property
    def signature(self) -> tuple[tuple[int, str], ...]:
        return tuple((choice.bits, choice.mode) for choice in self.choices)


def _prune_pareto(states: list[_ParetoState]) -> list[_ParetoState]:
    ordered = sorted(states, key=lambda state: (state.cost, state.error, state.signature))
    result: list[_ParetoState] = []
    best_error = math.inf
    for state in ordered:
        # This is a certificate frontier, so near-equal states must not be
        # merged: a smaller error by one ulp can be the only feasible state at
        # a tight budget.
        if state.error < best_error:
            result.append(state)
            best_error = state.error
    return result


def solve_discrete_precision_waterfill(
    layers: tuple[PrecisionLayer, ...],
    *,
    error_budget: float,
    max_pareto_states: int = 1_000_000,
) -> PrecisionAllocation:
    """Exactly solve the separable discrete bit/cost allocation Pareto frontier."""

    if not layers:
        raise ValueError("layers must be non-empty")
    if not math.isfinite(error_budget) or error_budget < 0.0:
        raise ValueError("error_budget must be finite and non-negative")
    if isinstance(max_pareto_states, bool) or max_pareto_states < 1:
        raise ValueError("max_pareto_states must be positive")

    frontier = [_ParetoState(error=0.0, cost=0.0, choices=())]
    for layer in layers:
        expanded = [
            _ParetoState(
                error=state.error + option.error_bound,
                cost=state.cost + option.measured_cost,
                choices=(*state.choices, option),
            )
            for state in frontier
            for option in layer.options
        ]
        if len(expanded) > max_pareto_states:
            raise RuntimeError("precision Pareto frontier exceeded max_pareto_states")
        frontier = _prune_pareto(expanded)

    feasible = [state for state in frontier if state.error <= error_budget]
    if not feasible:
        raise ValueError("no precision allocation satisfies the error budget")
    selected = min(feasible, key=lambda state: (state.cost, state.error, state.signature))
    return PrecisionAllocation(
        choices=selected.choices,
        total_error_bound=selected.error,
        total_measured_cost=selected.cost,
        pareto_state_count=len(frontier),
    )


def continuous_margin_waterfill(
    *,
    sensitivity_coefficients: np.ndarray,
    cost_per_bit: np.ndarray,
    error_budget: float,
    min_bits: np.ndarray,
    max_bits: np.ndarray,
) -> dict[str, np.ndarray | float]:
    """Solve the continuous KKT relaxation ``sum a_l 2^-b_l <= epsilon``.

    The interior solution is
    ``b_l = log2(lambda ln(2) a_l / c_l)``, clipped to each layer's bit box.
    This is a relaxation; executable integer choices use
    :func:`solve_discrete_precision_waterfill`.
    """

    a = np.asarray(sensitivity_coefficients, dtype=np.float64)
    c = np.asarray(cost_per_bit, dtype=np.float64)
    lo = np.asarray(min_bits, dtype=np.float64)
    hi = np.asarray(max_bits, dtype=np.float64)
    if not (a.shape == c.shape == lo.shape == hi.shape) or a.ndim != 1 or a.size == 0:
        raise ValueError("waterfill arrays must be equal-length non-empty vectors")
    if not all(np.all(np.isfinite(array)) for array in (a, c, lo, hi)):
        raise ValueError("waterfill inputs must be finite")
    if np.any(a <= 0.0) or np.any(c <= 0.0) or np.any(lo < 0.0) or np.any(hi < lo):
        raise ValueError("invalid coefficient, cost, or bit bounds")
    if not math.isfinite(error_budget) or error_budget <= 0.0:
        raise ValueError("error_budget must be finite and positive")

    error_at_lo = float(np.sum(a * np.exp2(-lo)))
    error_at_hi = float(np.sum(a * np.exp2(-hi)))
    if error_budget < error_at_hi:
        raise ValueError("error budget is below the maximum-bit certificate floor")
    if error_budget >= error_at_lo:
        bits = lo.copy()
    else:
        # Work in log2(lambda) and derive a bracket from the caller's actual
        # bit boxes.  A fixed [-128,128] natural-log bracket silently fails on
        # valid wide boxes (for example b_max=1000 and epsilon=2^-500).
        # Never form ``a / c``: finite operands may overflow the quotient and
        # turn an otherwise valid certificate into NaNs.  The equivalent log
        # expression keeps the whole finite binary64 domain representable.
        offset = np.log2(a) + math.log2(math.log(2.0)) - np.log2(c)
        if not np.all(np.isfinite(offset)):
            raise RuntimeError("continuous waterfill produced a non-finite KKT offset")
        log2_lambda_lo = float(np.min(lo - offset) - 1.0)
        log2_lambda_hi = float(np.max(hi - offset) + 1.0)
        bits = hi.copy()
        for _ in range(512):
            log2_lambda = (log2_lambda_lo + log2_lambda_hi) / 2.0
            bits = np.clip(log2_lambda + offset, lo, hi)
            error = float(np.sum(a * np.exp2(-bits)))
            if error > error_budget:
                log2_lambda_lo = log2_lambda
            else:
                log2_lambda_hi = log2_lambda
        # The high bracket is feasible by invariant; use it rather than the
        # final midpoint so the returned certificate never exceeds epsilon.
        bits = np.clip(log2_lambda_hi + offset, lo, hi)
    total_error = float(np.sum(a * np.exp2(-bits)))
    total_cost = float(np.sum(c * bits))
    if (
        not np.all(np.isfinite(bits))
        or not math.isfinite(total_error)
        or not math.isfinite(total_cost)
    ):
        raise RuntimeError("continuous waterfill produced a non-finite certificate")
    if total_error > error_budget and not math.isclose(
        total_error, error_budget, rel_tol=1e-12, abs_tol=0.0
    ):
        raise RuntimeError("continuous waterfill failed its error-budget postcondition")
    return {
        "bits": bits,
        "total_error_bound": total_error,
        "total_cost": total_cost,
    }


@dataclass(frozen=True)
class SupportCost:
    name: str
    dense_flops: float
    requested_active_fraction: float
    closed_active_fraction: float
    global_dependency: bool = False

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("support-cost name must be non-empty")
        if not math.isfinite(self.dense_flops) or self.dense_flops <= 0.0:
            raise ValueError("dense_flops must be finite and positive")
        for name, value in (
            ("requested_active_fraction", self.requested_active_fraction),
            ("closed_active_fraction", self.closed_active_fraction),
        ):
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be a finite fraction")
        if self.closed_active_fraction < self.requested_active_fraction:
            raise ValueError("support closure cannot reduce the requested support")
        if self.global_dependency and self.closed_active_fraction != 1.0:
            raise ValueError("a global dependency requires full closed support")


def support_closure_flop_accounting(stages: tuple[SupportCost, ...]) -> dict[str, float]:
    """Compare naive mask FLOPs with dependency-closed exact FLOPs."""

    if not stages:
        raise ValueError("stages must be non-empty")
    dense = sum(stage.dense_flops for stage in stages)
    naive = sum(stage.dense_flops * stage.requested_active_fraction for stage in stages)
    closed = sum(stage.dense_flops * stage.closed_active_fraction for stage in stages)
    return {
        "dense_flops": dense,
        "naive_mask_flops": naive,
        "dependency_closed_flops": closed,
        "naive_mask_speedup_upper_bound": dense / naive if naive > 0.0 else math.inf,
        "dependency_closed_speedup_upper_bound": dense / closed if closed > 0.0 else math.inf,
        "closure_tax_flops": closed - naive,
    }


def cadence_compute_fraction(
    *,
    disjoint_area_fractions: np.ndarray,
    refresh_cadences: np.ndarray,
    remainder_cadence: int = 1,
) -> float:
    """Return average refreshed area per step for disjoint spatial tiers."""

    if (
        isinstance(remainder_cadence, (bool, np.bool_))
        or not isinstance(remainder_cadence, (int, np.integer))
        or remainder_cadence < 1
    ):
        raise ValueError("cadences must be positive integers")
    areas = np.asarray(disjoint_area_fractions, dtype=np.float64)
    raw_cadences = np.asarray(refresh_cadences)
    if not np.issubdtype(raw_cadences.dtype, np.integer):
        if not np.issubdtype(raw_cadences.dtype, np.floating):
            raise ValueError("cadences must be positive integers")
        if not np.all(np.isfinite(raw_cadences)) or not np.all(
            raw_cadences == np.floor(raw_cadences)
        ):
            raise ValueError("cadences must be positive integers")
    cadences = raw_cadences.astype(np.int64)
    if areas.ndim != 1 or cadences.shape != areas.shape or areas.size == 0:
        raise ValueError("areas and cadences must be equal-length non-empty vectors")
    if not np.all(np.isfinite(areas)) or np.any(areas < 0.0) or float(np.sum(areas)) > 1.0 + 1e-12:
        raise ValueError("areas must be finite, non-negative, and disjoint within unit mass")
    if np.any(cadences < 1):
        raise ValueError("cadences must be positive integers")
    remainder = max(0.0, 1.0 - float(np.sum(areas)))
    return float(np.sum(areas / cadences) + remainder / remainder_cadence)


def covariant_pair_reuse_break_even(
    *, exact_teacher_cost: float, warp_cost: float, refresh_cost: float
) -> float:
    """Return the minimum continuous reuse horizon for compute-once-warp.

    Amortized cost is ``warp_cost + refresh_cost / K``.  Reuse wins only for
    ``K > refresh_cost / (exact_teacher_cost - warp_cost)``.
    """

    values = (exact_teacher_cost, warp_cost, refresh_cost)
    if not all(math.isfinite(value) and value >= 0.0 for value in values):
        raise ValueError("costs must be finite and non-negative")
    if exact_teacher_cost <= warp_cost:
        return math.inf
    return refresh_cost / (exact_teacher_cost - warp_cost)


__all__ = [
    "ArgmaxCertificate",
    "PrecisionAllocation",
    "PrecisionLayer",
    "PrecisionOption",
    "SupportCost",
    "cadence_compute_fraction",
    "certify_argmax_intervals",
    "continuous_margin_waterfill",
    "covariant_pair_reuse_break_even",
    "crt_reduction_certificate",
    "fixed_width_reduction_certificate",
    "float32_ordered_sum",
    "integer_ordered_sum",
    "max_plus_ordered_reduce",
    "minimum_signed_accumulator_bits",
    "minimum_signed_bits_for_abs_bound",
    "number_system_disposition",
    "solve_discrete_precision_waterfill",
    "support_closure_flop_accounting",
    "tropical_argmax_ordered_reduce",
]
