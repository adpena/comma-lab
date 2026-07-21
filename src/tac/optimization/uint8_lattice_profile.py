# SPDX-License-Identifier: MIT
"""Exact-or-bounded profiles of factor-2 uint8 affine preimages.

This module profiles the integer equation ``c dot u = target_integer`` over
``u in [0,255]^n``.  Exhaustion yields an exact intersection cardinality.  A
node/time/plugin budget yields a proved lower bound and a sound finite upper
bound; it is never converted into infeasibility.

Selection is independently governed by a deterministic description-cost model
and a typed pose-feasibility plug-in.  The default selector is a receiver-public
signed-residual code, not a Euclidean or source-frame preference.
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
import time
import zlib
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from math import gcd
from numbers import Integral
from typing import Any, Protocol, runtime_checkable

import brotli
import numpy as np


class LatticeProfileError(ValueError):
    """Fail-closed malformed equation, selector, plug-in, or stream error."""


class ProfileStatus(StrEnum):
    EXACT = "EXACT"
    INFEASIBLE_EXHAUSTIVE = "INFEASIBLE_EXHAUSTIVE"
    BOUNDED_NODE_CAP = "BOUNDED_NODE_CAP"
    BOUNDED_TIME_CAP = "BOUNDED_TIME_CAP"
    PLUGIN_ERROR_UNKNOWN = "PLUGIN_ERROR_UNKNOWN"


@dataclass(frozen=True)
class PoseFilterDecision:
    feasible: bool
    additional_cost_bits: int = 0
    diagnostic: Mapping[str, Any] = field(default_factory=dict)


@runtime_checkable
class PoseFeasibilityPlugin(Protocol):
    """Deterministic candidate filter for a future Seg/Pose intersection."""

    @property
    def identity(self) -> str: ...

    def evaluate(self, candidate: tuple[int, ...]) -> PoseFilterDecision: ...


@dataclass(frozen=True)
class NoOpPosePlugin:
    identity: str = "pose.noop_seg_only.v1"

    def evaluate(self, candidate: tuple[int, ...]) -> PoseFilterDecision:
        del candidate
        return PoseFilterDecision(True, 0, {"mode": "seg_only"})


@runtime_checkable
class DescriptionCostModel(Protocol):
    """Deterministic, receiver-reproducible candidate description cost."""

    @property
    def identity(self) -> str: ...

    def cost_bits(self, candidate: tuple[int, ...]) -> int: ...


def _zigzag(value: int) -> int:
    return 2 * value if value >= 0 else -2 * value - 1


def _uleb128(value: int) -> bytes:
    if value < 0:
        raise LatticeProfileError("ULEB128 value must be nonnegative")
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        out.append(byte | (0x80 if value else 0))
        if not value:
            return bytes(out)


@dataclass(frozen=True)
class SignedResidualCostModel:
    """Fixed predictor plus signed-zigzag ULEB128 residual code.

    The predictor is part of the public identity and never comes from the
    hidden/source camera block.  Stream-level headers are constant across
    candidates and are charged by :func:`candidate_stream_accounting`.
    """

    predictor: int | tuple[int, ...] = 128

    def __post_init__(self) -> None:
        values = (self.predictor,) if isinstance(self.predictor, Integral) else tuple(self.predictor)
        if not values or any(
            isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or not 0 <= int(value) <= 255
            for value in values
        ):
            raise LatticeProfileError("predictor must contain uint8 integer values")
        normalized: int | tuple[int, ...]
        normalized = int(values[0]) if isinstance(self.predictor, Integral) else tuple(int(v) for v in values)
        object.__setattr__(self, "predictor", normalized)

    @property
    def identity(self) -> str:
        values = [self.predictor] if isinstance(self.predictor, int) else list(self.predictor)
        return "signed_residual_zigzag_uleb128.v1:" + ",".join(str(value) for value in values)

    def predictor_for(self, length: int) -> tuple[int, ...]:
        if isinstance(self.predictor, int):
            return (self.predictor,) * length
        if len(self.predictor) != length:
            raise LatticeProfileError("predictor arity does not match candidate")
        return self.predictor

    def encode_candidate(self, candidate: tuple[int, ...]) -> bytes:
        predictor = self.predictor_for(len(candidate))
        return b"".join(_uleb128(_zigzag(value - base)) for value, base in zip(candidate, predictor, strict=True))

    def cost_bits(self, candidate: tuple[int, ...]) -> int:
        return 8 * len(self.encode_candidate(candidate))


@dataclass(frozen=True)
class BlockProfileResult:
    coefficients: tuple[int, ...]
    denominator: int
    target_integer: int
    selector_identity: str
    pose_plugin_identity: str
    status: ProfileStatus
    exhaustive: bool
    exact_cardinality: int | None
    cardinality_lower_bound: int
    cardinality_upper_bound: int
    affine_feasible_seen: int
    pose_rejected_seen: int
    nodes_visited: int
    selected_candidate: tuple[int, ...] | None
    selected_cost_bits: int | None
    selection_globally_exact: bool
    plugin_error: str | None = None

    @property
    def proved_infeasible(self) -> bool:
        return self.exhaustive and self.exact_cardinality == 0


@dataclass(frozen=True)
class SourceWitnessBounds:
    """Vectorized root-state bounds backed by an exact real-source witness."""

    cardinality_lower_bound: np.ndarray
    cardinality_upper_bound: np.ndarray
    witness_verified_blocks: int
    lower_bound_method: str = "MAX_DISJOINT_PAIR_NULL_FIBER_PRODUCT"
    derivation: str = "DERIVED_BOUNDS_FROM_REAL_N600_SOURCE_WITNESS"


def _require_int64_domain(raw: np.ndarray, *, name: str) -> None:
    """Reject non-integral or out-of-domain values before any int64 cast."""

    if raw.dtype.kind not in "iu" or raw.dtype.kind == "b":
        raise LatticeProfileError(f"{name} must contain non-boolean integers")
    if raw.size == 0:
        return
    minimum = int(raw.min())
    maximum = int(raw.max())
    info = np.iinfo(np.int64)
    if minimum < info.min or maximum > info.max:
        raise LatticeProfileError(f"{name} contains values outside the int64 domain")


def _pair_null_fiber_count(
    coefficients: np.ndarray,
    witnesses: np.ndarray,
    first_index: int,
    second_index: int,
) -> np.ndarray:
    """Count the certified in-cube integer null-fiber for one tap pair."""

    first_coefficient = coefficients[..., first_index]
    second_coefficient = coefficients[..., second_index]
    divisor = np.gcd(first_coefficient, second_coefficient)
    first_step = second_coefficient // divisor
    second_step = first_coefficient // divisor
    first_witness = witnesses[..., first_index]
    second_witness = witnesses[..., second_index]

    # ceil(a / b) == -floor(-a / b) for positive b.  Every intermediate
    # remains inside int64 because witnesses are uint8-domain and steps are
    # positive int64-domain values.
    lower = np.maximum(
        -np.floor_divide(first_witness, first_step),
        -np.floor_divide(-(second_witness - 255), second_step),
    )
    upper = np.minimum(
        np.floor_divide(255 - first_witness, first_step),
        np.floor_divide(second_witness, second_step),
    )
    return np.maximum(0, upper - lower + 1).astype(np.int64, copy=False)


def vectorized_source_witness_bounds(
    coefficients: np.ndarray,
    target_integers: np.ndarray,
    source_witnesses: np.ndarray,
) -> SourceWitnessBounds:
    """Verify exact witnesses and derive sound root bounds without Python DFS.

    Inputs broadcast over every dimension except the final tap dimension.  A
    valid source witness seeds certified pairwise integer null-fibers.  Products
    are taken only across disjoint tap pairs, so the resulting candidates
    combine independently and injectively.  The upper bound applies root range
    pruning and whole-equation gcd pruning, but is deliberately not labeled an
    exact cardinality.
    """

    raw_coefficients = np.asarray(coefficients)
    raw_targets = np.asarray(target_integers)
    raw_witnesses = np.asarray(source_witnesses)
    for name, raw in (
        ("coefficients", raw_coefficients),
        ("target_integers", raw_targets),
        ("source_witnesses", raw_witnesses),
    ):
        _require_int64_domain(raw, name=name)
    if raw_coefficients.ndim < 1 or not 1 <= raw_coefficients.shape[-1] <= 4:
        raise LatticeProfileError("coefficients require a final tap dimension of one to four")
    if raw_witnesses.ndim < 1 or raw_witnesses.shape[-1] != raw_coefficients.shape[-1]:
        raise LatticeProfileError("source witnesses and coefficients disagree on tap arity")
    arity = raw_coefficients.shape[-1]
    coefficient_limit = np.iinfo(np.int64).max // (255 * arity)
    if np.any(raw_coefficients <= 0):
        raise LatticeProfileError("coefficients must be positive")
    if np.any(raw_coefficients > coefficient_limit):
        raise LatticeProfileError("coefficients exceed the safe int64 accumulation limit")
    if np.any(raw_witnesses < 0) or np.any(raw_witnesses > 255):
        raise LatticeProfileError("source witnesses must stay inside uint8")
    try:
        coefficient_view, witness_view = np.broadcast_arrays(
            raw_coefficients,
            raw_witnesses,
        )
        target_view = np.broadcast_to(raw_targets, coefficient_view.shape[:-1])
    except ValueError as exc:
        raise LatticeProfileError("source-witness bound geometry is not broadcast-compatible") from exc
    coefficient64 = coefficient_view.astype(np.int64, copy=False)
    witness64 = witness_view.astype(np.int64, copy=False)
    target64 = target_view.astype(np.int64, copy=False)
    if np.any(coefficient64 <= 0):
        raise LatticeProfileError("coefficients must be positive")
    if np.any(coefficient64 > coefficient_limit):
        raise LatticeProfileError("coefficients exceed the safe int64 accumulation limit")
    if np.any(witness64 < 0) or np.any(witness64 > 255):
        raise LatticeProfileError("source witnesses must stay inside uint8")
    coefficient_sum = np.sum(coefficient64, axis=-1, dtype=np.int64)
    if np.any(coefficient_sum > np.iinfo(np.int64).max // 255):
        raise LatticeProfileError("source-witness bound accumulation exceeds int64")
    realized = np.sum(coefficient64 * witness64, axis=-1, dtype=np.int64)
    if not np.array_equal(realized, target64):
        mismatches = int(np.count_nonzero(realized != target64))
        raise LatticeProfileError(f"real source witness violates exact integer equation for {mismatches} blocks")

    equation_gcd = np.gcd.reduce(coefficient64, axis=-1)
    in_range = (target64 >= 0) & (target64 <= 255 * coefficient_sum)
    gcd_valid = np.remainder(target64, equation_gcd) == 0
    first = coefficient64[..., 0]
    tail_sum = coefficient_sum - first
    low = np.maximum(0, -np.floor_divide(-(target64 - 255 * tail_sum), first))
    high = np.minimum(255, np.floor_divide(target64, first))
    root_values = np.maximum(0, high - low + 1).astype(np.int64, copy=False)
    upper = root_values * np.int64(256 ** (coefficient64.shape[-1] - 1))
    upper = np.where(in_range & gcd_valid, upper, 0).astype(np.int64, copy=False)
    if arity == 1:
        lower = np.ones(target64.shape, dtype=np.int64)
    elif arity == 2:
        lower = _pair_null_fiber_count(coefficient64, witness64, 0, 1)
    elif arity == 3:
        lower = np.maximum.reduce(
            [
                _pair_null_fiber_count(coefficient64, witness64, 0, 1),
                _pair_null_fiber_count(coefficient64, witness64, 0, 2),
                _pair_null_fiber_count(coefficient64, witness64, 1, 2),
            ]
        )
    else:
        lower = np.maximum.reduce(
            [
                _pair_null_fiber_count(coefficient64, witness64, 0, 1)
                * _pair_null_fiber_count(coefficient64, witness64, 2, 3),
                _pair_null_fiber_count(coefficient64, witness64, 0, 2)
                * _pair_null_fiber_count(coefficient64, witness64, 1, 3),
                _pair_null_fiber_count(coefficient64, witness64, 0, 3)
                * _pair_null_fiber_count(coefficient64, witness64, 1, 2),
            ]
        )
    lower = np.asarray(lower, dtype=np.int64)
    if np.any(lower < 1):
        raise LatticeProfileError("pair null-fiber bound lost the verified source witness")
    if np.any(upper < lower):
        raise LatticeProfileError("root range/gcd bound excluded the verified source witness")
    return SourceWitnessBounds(
        cardinality_lower_bound=lower,
        cardinality_upper_bound=upper,
        witness_verified_blocks=int(target64.size),
    )


@dataclass(frozen=True)
class _SearchState:
    index: int
    residual: int
    prefix: tuple[int, ...]


def _integer(value: object, *, name: str, minimum: int | None = None) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise LatticeProfileError(f"{name} must be an integer")
    result = int(value)
    if minimum is not None and result < minimum:
        raise LatticeProfileError(f"{name} must be at least {minimum}")
    return result


def _identity(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 1024:
        raise LatticeProfileError(f"{name} must be a nonempty bounded string")
    return value


def _suffix_geometry(coefficients: tuple[int, ...]) -> tuple[list[int], list[int]]:
    suffix_sum = [0] * (len(coefficients) + 1)
    suffix_gcd = [0] * (len(coefficients) + 1)
    for index in range(len(coefficients) - 1, -1, -1):
        suffix_sum[index] = suffix_sum[index + 1] + coefficients[index]
        suffix_gcd[index] = gcd(coefficients[index], suffix_gcd[index + 1])
    return suffix_sum, suffix_gcd


def _state_upper_bound(
    state: _SearchState,
    coefficients: tuple[int, ...],
    suffix_sum: Sequence[int],
    suffix_gcd: Sequence[int],
) -> int:
    index = state.index
    residual = state.residual
    remaining = len(coefficients) - index
    if remaining == 0:
        return int(residual == 0)
    if residual < 0 or residual > 255 * suffix_sum[index]:
        return 0
    if residual % suffix_gcd[index]:
        return 0
    if remaining == 1:
        coefficient = coefficients[index]
        return int(residual % coefficient == 0 and 0 <= residual // coefficient <= 255)
    coefficient = coefficients[index]
    tail_max = 255 * suffix_sum[index + 1]
    lo = max(0, -((-(residual - tail_max)) // coefficient))
    hi = min(255, residual // coefficient)
    if lo > hi:
        return 0
    # Sound and finite.  Tail equation/gcd pruning can only reduce this number.
    return (hi - lo + 1) * (256 ** (remaining - 1))


def _validated_pose_decision(plugin: PoseFeasibilityPlugin, candidate: tuple[int, ...]) -> PoseFilterDecision:
    decision = plugin.evaluate(candidate)
    if not isinstance(decision, PoseFilterDecision) or type(decision.feasible) is not bool:
        raise LatticeProfileError("pose plug-in returned a malformed decision")
    cost = _integer(
        decision.additional_cost_bits,
        name="pose additional_cost_bits",
        minimum=0,
    )
    try:
        json.dumps(dict(decision.diagnostic), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise LatticeProfileError("pose plug-in diagnostic is not finite JSON") from exc
    return PoseFilterDecision(decision.feasible, cost, dict(decision.diagnostic))


def _normalized_seed_candidate(
    seed_candidate: Sequence[int] | None,
    *,
    coefficients: tuple[int, ...],
    target_integer: int,
) -> tuple[int, ...] | None:
    """Validate an optional external uint8 witness against the exact equation."""

    if seed_candidate is None:
        return None
    try:
        raw_seed = tuple(seed_candidate)
    except TypeError as exc:
        raise LatticeProfileError("seed_candidate must be a uint8 integer sequence") from exc
    if len(raw_seed) != len(coefficients):
        raise LatticeProfileError("seed_candidate arity does not match coefficients")
    seed = tuple(_integer(value, name=f"seed_candidate[{index}]", minimum=0) for index, value in enumerate(raw_seed))
    if any(value > 255 for value in seed):
        raise LatticeProfileError("seed_candidate values must be uint8 integers")
    realized = sum(coefficient * value for coefficient, value in zip(coefficients, seed, strict=True))
    if realized != target_integer:
        raise LatticeProfileError("seed_candidate violates the exact integer equation")
    return seed


def _candidate_total_cost(
    selector: DescriptionCostModel,
    candidate: tuple[int, ...],
    decision: PoseFilterDecision,
) -> int:
    raw_cost = selector.cost_bits(candidate)
    cost = _integer(raw_cost, name="description cost bits", minimum=0)
    return cost + decision.additional_cost_bits


def _state_contains_candidate(
    state: _SearchState,
    coefficients: tuple[int, ...],
    candidate: tuple[int, ...],
) -> bool:
    """Return whether one unresolved DFS subtree still contains ``candidate``."""

    index = state.index
    if state.prefix != candidate[:index]:
        return False
    residual = sum(
        coefficient * value
        for coefficient, value in zip(
            coefficients[index:],
            candidate[index:],
            strict=True,
        )
    )
    return state.residual == residual


def profile_integer_block(
    coefficients: Sequence[int],
    denominator: int,
    target_integer: int,
    *,
    cost_model: DescriptionCostModel | None = None,
    pose_plugin: PoseFeasibilityPlugin | None = None,
    seed_candidate: Sequence[int] | None = None,
    max_nodes: int = 4096,
    time_limit_seconds: float | None = None,
) -> BlockProfileResult:
    """Enumerate/profile one factor-2 channel block.

    ``denominator`` is included in custody/cache identity even though the exact
    equation uses the numerator ``target_integer`` directly.
    """

    raw_coefficients = tuple(coefficients)
    if not raw_coefficients or len(raw_coefficients) > 4:
        raise LatticeProfileError("factor-2 profile requires one to four coefficients")
    coeff = tuple(
        _integer(value, name=f"coefficients[{index}]", minimum=1) for index, value in enumerate(raw_coefficients)
    )
    denominator = _integer(denominator, name="denominator", minimum=1)
    target_integer = _integer(target_integer, name="target_integer")
    max_nodes = _integer(max_nodes, name="max_nodes", minimum=1)
    if time_limit_seconds is not None:
        if (
            isinstance(time_limit_seconds, bool)
            or not isinstance(time_limit_seconds, (int, float))
            or not math.isfinite(float(time_limit_seconds))
            or float(time_limit_seconds) <= 0.0
        ):
            raise LatticeProfileError("time_limit_seconds must be finite and positive")
        deadline = time.monotonic() + float(time_limit_seconds)
    else:
        deadline = None
    selector = SignedResidualCostModel() if cost_model is None else cost_model
    plugin = NoOpPosePlugin() if pose_plugin is None else pose_plugin
    selector_identity = _identity(getattr(selector, "identity", None), name="selector identity")
    plugin_identity = _identity(getattr(plugin, "identity", None), name="pose plug-in identity")
    if not callable(getattr(selector, "cost_bits", None)) or not callable(getattr(plugin, "evaluate", None)):
        raise LatticeProfileError("selector/pose plug-in does not satisfy its protocol")

    seed = _normalized_seed_candidate(
        seed_candidate,
        coefficients=coeff,
        target_integer=target_integer,
    )

    # Preserve coefficient order: the candidate tuple maps directly to camera taps.
    suffix_sum, suffix_gcd = _suffix_geometry(coeff)
    stack = [_SearchState(0, target_integer, ())]
    nodes = 0
    lower = 0
    affine_seen = 0
    pose_rejected = 0
    selected: tuple[int, ...] | None = None
    selected_cost: int | None = None
    budget_status: ProfileStatus | None = None
    plugin_error: str | None = None

    # A seed is an externally supplied exact affine witness.  Resolve it once
    # through the same pose and description-cost surfaces as a DFS leaf.  Its
    # DFS leaf remains in the traversal for node/exhaustion semantics, but is
    # skipped when reached so neither counts nor plug-in work are duplicated.
    if seed is not None:
        affine_seen = 1
        try:
            seed_decision = _validated_pose_decision(plugin, seed)
        except Exception as exc:  # plug-in error is UNKNOWN, never infeasible
            root_upper = _state_upper_bound(stack[0], coeff, suffix_sum, suffix_gcd)
            return BlockProfileResult(
                coefficients=coeff,
                denominator=denominator,
                target_integer=target_integer,
                selector_identity=selector_identity,
                pose_plugin_identity=plugin_identity,
                status=ProfileStatus.PLUGIN_ERROR_UNKNOWN,
                exhaustive=False,
                exact_cardinality=None,
                cardinality_lower_bound=0,
                cardinality_upper_bound=root_upper,
                affine_feasible_seen=affine_seen,
                pose_rejected_seen=0,
                nodes_visited=0,
                selected_candidate=None,
                selected_cost_bits=None,
                selection_globally_exact=False,
                plugin_error=f"{type(exc).__name__}: {exc}",
            )
        if seed_decision.feasible:
            lower = 1
            selected = seed
            selected_cost = _candidate_total_cost(selector, seed, seed_decision)
        else:
            pose_rejected = 1

    while stack:
        if nodes >= max_nodes:
            budget_status = ProfileStatus.BOUNDED_NODE_CAP
            break
        if deadline is not None and time.monotonic() >= deadline:
            budget_status = ProfileStatus.BOUNDED_TIME_CAP
            break
        state = stack.pop()
        nodes += 1
        if _state_upper_bound(state, coeff, suffix_sum, suffix_gcd) == 0:
            continue
        if state.index == len(coeff):
            candidate = state.prefix
            if seed is not None and candidate == seed:
                continue
            affine_seen += 1
            try:
                decision = _validated_pose_decision(plugin, candidate)
            except Exception as exc:  # plug-in error is UNKNOWN, never infeasible
                budget_status = ProfileStatus.PLUGIN_ERROR_UNKNOWN
                plugin_error = f"{type(exc).__name__}: {exc}"
                # This exact affine candidate remains unresolved by the plugin.
                stack.append(state)
                break
            if not decision.feasible:
                pose_rejected += 1
                continue
            lower += 1
            total_cost = _candidate_total_cost(selector, candidate, decision)
            key = (total_cost, candidate)
            if selected is None or key < (int(selected_cost), selected):
                selected = candidate
                selected_cost = total_cost
            continue
        coefficient = coeff[state.index]
        tail_max = 255 * suffix_sum[state.index + 1]
        lo = max(0, -((-(state.residual - tail_max)) // coefficient))
        hi = min(255, state.residual // coefficient)
        # Reverse push makes ascending uint8 values the stable traversal order.
        for value in range(hi, lo - 1, -1):
            child = _SearchState(
                state.index + 1,
                state.residual - coefficient * value,
                (*state.prefix, value),
            )
            if _state_upper_bound(child, coeff, suffix_sum, suffix_gcd):
                stack.append(child)

    if budget_status is None:
        exact = lower
        status = ProfileStatus.EXACT if exact else ProfileStatus.INFEASIBLE_EXHAUSTIVE
        return BlockProfileResult(
            coefficients=coeff,
            denominator=denominator,
            target_integer=target_integer,
            selector_identity=selector_identity,
            pose_plugin_identity=plugin_identity,
            status=status,
            exhaustive=True,
            exact_cardinality=exact,
            cardinality_lower_bound=exact,
            cardinality_upper_bound=exact,
            affine_feasible_seen=affine_seen,
            pose_rejected_seen=pose_rejected,
            nodes_visited=nodes,
            selected_candidate=selected,
            selected_cost_bits=selected_cost,
            selection_globally_exact=True,
        )
    unresolved = sum(_state_upper_bound(state, coeff, suffix_sum, suffix_gcd) for state in stack)
    if seed is not None and any(_state_contains_candidate(state, coeff, seed) for state in stack):
        # The externally resolved seed is still represented once in the
        # unresolved affine frontier.  Remove it whether pose accepted or
        # rejected; otherwise both lower and upper cardinalities double-count
        # that known leaf.
        unresolved -= 1
        if unresolved < 0:
            raise LatticeProfileError("seed subtraction underflowed unresolved cardinality")
    upper = lower + unresolved
    if upper < lower:  # pragma: no cover - Python integers do not overflow
        raise LatticeProfileError("cardinality upper bound underflow")
    return BlockProfileResult(
        coefficients=coeff,
        denominator=denominator,
        target_integer=target_integer,
        selector_identity=selector_identity,
        pose_plugin_identity=plugin_identity,
        status=budget_status,
        exhaustive=False,
        exact_cardinality=None,
        cardinality_lower_bound=lower,
        cardinality_upper_bound=upper,
        affine_feasible_seen=affine_seen,
        pose_rejected_seen=pose_rejected,
        nodes_visited=nodes,
        selected_candidate=selected,
        selected_cost_bits=selected_cost,
        selection_globally_exact=False,
        plugin_error=plugin_error,
    )


def profile_cache_key(
    *,
    coefficients: Sequence[int],
    denominator: int,
    target_integer: int,
    selector_identity: str,
    pose_plugin_identity: str,
    seed_candidate: Sequence[int] | None = None,
) -> str:
    """Complete reuse key; target-only reuse is impossible by construction."""

    normalized_coefficients = tuple(_integer(value, name="cache coefficient", minimum=1) for value in coefficients)
    normalized_target = _integer(target_integer, name="cache target_integer")
    seed = _normalized_seed_candidate(
        seed_candidate,
        coefficients=normalized_coefficients,
        target_integer=normalized_target,
    )
    payload = {
        "coefficients": list(normalized_coefficients),
        "denominator": _integer(denominator, name="cache denominator", minimum=1),
        "target_integer": normalized_target,
        "selector_identity": _identity(selector_identity, name="selector identity"),
        "pose_plugin_identity": _identity(pose_plugin_identity, name="pose plug-in identity"),
        "seed_candidate": None if seed is None else list(seed),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


_STREAM_MAGIC = b"U8RDS1"


def encode_candidate_stream(
    candidates: Iterable[tuple[int, ...] | None],
    *,
    cost_model: SignedResidualCostModel | None = None,
) -> bytes:
    """Encode selected blocks with explicit count, arity, and absent markers."""

    model = SignedResidualCostModel() if cost_model is None else cost_model
    rows = tuple(candidates)
    out = bytearray(_STREAM_MAGIC)
    out.extend(struct.pack("<I", len(rows)))
    for candidate in rows:
        if candidate is None:
            out.append(0)
            continue
        if not isinstance(candidate, tuple) or not 1 <= len(candidate) <= 4:
            raise LatticeProfileError("candidate stream supports uint8 arity one to four")
        normalized = tuple(_integer(value, name="candidate stream value", minimum=0) for value in candidate)
        if any(value > 255 for value in normalized):
            raise LatticeProfileError("candidate stream values must be uint8 integers")
        out.append(len(candidate))
        payload = model.encode_candidate(normalized)
        out.extend(_uleb128(len(payload)))
        out.extend(payload)
    return bytes(out)


def _decode_uleb128(
    payload: bytes,
    offset: int,
    *,
    limit: int,
    field: str,
    max_bytes: int,
) -> tuple[int, int]:
    """Decode one canonical bounded ULEB128 integer within ``[offset, limit)``."""

    start = offset
    value = 0
    shift = 0
    for _ in range(max_bytes):
        if offset >= limit:
            raise LatticeProfileError(f"{field} ULEB128 is unterminated")
        byte = payload[offset]
        offset += 1
        value |= (byte & 0x7F) << shift
        if byte & 0x80 == 0:
            encoded = payload[start:offset]
            if encoded != _uleb128(value):
                raise LatticeProfileError(f"{field} ULEB128 is noncanonical")
            return value, offset
        shift += 7
    raise LatticeProfileError(f"{field} ULEB128 exceeds the bounded encoding")


def _unzigzag(value: int) -> int:
    return value // 2 if value % 2 == 0 else -(value // 2) - 1


def decode_candidate_stream(
    payload: bytes | bytearray | memoryview,
    *,
    cost_model: SignedResidualCostModel | None = None,
) -> tuple[tuple[int, ...] | None, ...]:
    """Strictly invert :func:`encode_candidate_stream`.

    The decoder refuses malformed magic/count/arity fields, unterminated or
    noncanonical ULEB128 values, row-payload under/over-consumption, uint8
    reconstruction overflow, and all trailing bytes.
    """

    if not isinstance(payload, (bytes, bytearray, memoryview)):
        raise LatticeProfileError("candidate stream must be bytes-like")
    raw = bytes(payload)
    header_bytes = len(_STREAM_MAGIC) + struct.calcsize("<I")
    if len(raw) < header_bytes:
        raise LatticeProfileError("candidate stream header is truncated")
    if raw[: len(_STREAM_MAGIC)] != _STREAM_MAGIC:
        raise LatticeProfileError("candidate stream header mismatch")
    (row_count,) = struct.unpack_from("<I", raw, len(_STREAM_MAGIC))
    offset = header_bytes
    # Every row consumes at least its arity/absence byte.  This also prevents a
    # hostile count field from driving an unbounded parse loop.
    if row_count > len(raw) - offset:
        raise LatticeProfileError("candidate stream row count exceeds available bytes")
    model = SignedResidualCostModel() if cost_model is None else cost_model
    if not isinstance(model, SignedResidualCostModel):
        raise LatticeProfileError("candidate stream decoder requires SignedResidualCostModel")

    rows: list[tuple[int, ...] | None] = []
    for row_index in range(row_count):
        if offset >= len(raw):
            raise LatticeProfileError("candidate stream row count is truncated")
        arity = raw[offset]
        offset += 1
        if arity == 0:
            rows.append(None)
            continue
        if not 1 <= arity <= 4:
            raise LatticeProfileError(f"candidate stream row {row_index} has invalid arity")
        payload_bytes, offset = _decode_uleb128(
            raw,
            offset,
            limit=len(raw),
            field=f"candidate stream row {row_index} payload length",
            max_bytes=5,
        )
        # One canonical residual consumes one or two bytes in the uint8/public
        # predictor domain.  Reject impossible lengths before slicing.
        if not arity <= payload_bytes <= 2 * arity:
            raise LatticeProfileError(f"candidate stream row {row_index} payload length is impossible")
        row_end = offset + payload_bytes
        if row_end > len(raw):
            raise LatticeProfileError(f"candidate stream row {row_index} payload is truncated")
        predictor = model.predictor_for(arity)
        values: list[int] = []
        for value_index, base in enumerate(predictor):
            encoded_residual, offset = _decode_uleb128(
                raw,
                offset,
                limit=row_end,
                field=f"candidate stream row {row_index} residual {value_index}",
                max_bytes=2,
            )
            value = base + _unzigzag(encoded_residual)
            if not 0 <= value <= 255:
                raise LatticeProfileError(f"candidate stream row {row_index} reconstructs outside uint8")
            values.append(value)
        if offset != row_end:
            raise LatticeProfileError(f"candidate stream row {row_index} has trailing payload bytes")
        rows.append(tuple(values))
    if offset != len(raw):
        raise LatticeProfileError("candidate stream has trailing bytes")
    return tuple(rows)


def _order0_iid_plugin_estimate(payload: bytes) -> dict[str, Any]:
    """Return the ideal empirical order-0 IID code length.

    This estimate deliberately gives the empirical PMF and its model to the
    coder for free.  It is not a lower bound on arbitrary lossless coding:
    context, run-length, grammar, or higher-order models can encode the same
    byte stream in fewer bits.
    """

    if not payload:
        raise LatticeProfileError("entropy accounting requires a nonempty stream")
    counts = np.bincount(np.frombuffer(payload, dtype=np.uint8), minlength=256)
    nonzero = counts[counts > 0].astype(np.float64)
    probabilities = nonzero / len(payload)
    entropy = float(-np.sum(probabilities * np.log2(probabilities)))
    return {
        "label": "ORDER0_IID_PLUGIN_IDEAL_LENGTH_ESTIMATE_NOT_UNIVERSAL_LOWER_BOUND",
        "assumptions": (
            "empirical order-0 IID PMF and model are free; no header or "
            "termination charge; not a bound for context or grammar coders"
        ),
        "bits_per_byte_symbol": entropy,
        "rounded_up_bytes": math.ceil(entropy * len(payload) / 8.0),
    }


def candidate_stream_accounting(payload: bytes) -> dict[str, Any]:
    """Return actual raw/zlib/Brotli bytes, including stream termination."""

    raw = bytes(payload)
    if not raw.startswith(_STREAM_MAGIC):
        raise LatticeProfileError("candidate stream header mismatch")
    zlib_bytes = zlib.compress(raw, level=9)
    brotli_bytes = brotli.compress(raw, quality=11)

    def row(label: str, value: bytes) -> dict[str, Any]:
        return {
            "label": label,
            "bytes": len(value),
            "sha256": hashlib.sha256(value).hexdigest(),
        }

    return {
        "raw": row("MEASURED_ACTUAL_RAW_STREAM_BYTES", raw),
        "zlib_level9": row("MEASURED_ACTUAL_ZLIB_LEVEL9_BYTES", zlib_bytes),
        "brotli_quality11": row("MEASURED_ACTUAL_BROTLI_QUALITY11_BYTES", brotli_bytes),
        "headers_and_termination_included": True,
        "order0_entropy": _order0_iid_plugin_estimate(raw),
    }


@dataclass
class _CompactStats:
    bin_width: float = 0.25
    bins: list[int] = field(default_factory=lambda: [0] * 129)
    count: int = 0
    zero_count: int = 0
    total: float = 0.0
    minimum: float | None = None
    maximum: float | None = None

    def add_count(self, cardinality: int) -> None:
        cardinality = _integer(cardinality, name="cardinality", minimum=0)
        self.count += 1
        if cardinality == 0:
            self.zero_count += 1
            return
        value = math.log2(cardinality)
        self.total += value
        self.minimum = value if self.minimum is None else min(self.minimum, value)
        self.maximum = value if self.maximum is None else max(self.maximum, value)
        index = min(len(self.bins) - 1, int(value / self.bin_width))
        self.bins[index] += 1

    def add_counts_array(self, cardinalities: np.ndarray) -> None:
        raw = np.asarray(cardinalities)
        if raw.dtype.kind not in "iu" or raw.dtype.kind == "b":
            raise LatticeProfileError("cardinality batch must contain non-boolean integers")
        values = raw.astype(np.int64, copy=False).reshape(-1)
        if np.any(values < 0):
            raise LatticeProfileError("cardinality batch must be nonnegative")
        self.count += int(values.size)
        self.zero_count += int(np.count_nonzero(values == 0))
        nonzero = values[values > 0]
        if nonzero.size == 0:
            return
        logs = np.log2(nonzero.astype(np.float64))
        self.total += float(logs.sum(dtype=np.float64))
        batch_min = float(logs.min())
        batch_max = float(logs.max())
        self.minimum = batch_min if self.minimum is None else min(self.minimum, batch_min)
        self.maximum = batch_max if self.maximum is None else max(self.maximum, batch_max)
        indices = np.minimum(len(self.bins) - 1, (logs / self.bin_width).astype(np.int64))
        counts = np.bincount(indices, minlength=len(self.bins))
        self.bins = [left + int(right) for left, right in zip(self.bins, counts, strict=True)]

    def _quantile_interval(self, quantile: float) -> list[float] | None:
        nonzero = self.count - self.zero_count
        if nonzero == 0:
            return None
        rank = max(0, math.ceil(quantile * nonzero) - 1)
        running = 0
        for index, count in enumerate(self.bins):
            running += count
            if running > rank:
                lower = index * self.bin_width
                upper = lower + self.bin_width
                return [lower, upper]
        raise LatticeProfileError("histogram quantile accounting drift")

    def summary(self) -> dict[str, Any]:
        nonzero = self.count - self.zero_count
        return {
            "observations": self.count,
            "zero_count": self.zero_count,
            "min_log2": self.minimum,
            "mean_log2_nonzero": None if nonzero == 0 else self.total / nonzero,
            "median_log2_histogram_interval": self._quantile_interval(0.5),
            "p90_log2_histogram_interval": self._quantile_interval(0.9),
            "p99_log2_histogram_interval": self._quantile_interval(0.99),
            "max_log2": self.maximum,
            "histogram_bin_width": self.bin_width,
            "histogram_counts": list(self.bins),
        }

    def state(self) -> dict[str, Any]:
        return self.__dict__.copy()

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> _CompactStats:
        result = cls()
        result.bin_width = float(state["bin_width"])
        result.bins = [int(value) for value in state["bins"]]
        result.count = int(state["count"])
        result.zero_count = int(state["zero_count"])
        result.total = float(state["total"])
        result.minimum = None if state["minimum"] is None else float(state["minimum"])
        result.maximum = None if state["maximum"] is None else float(state["maximum"])
        if result.bin_width != 0.25 or len(result.bins) != 129:
            raise LatticeProfileError("compact histogram state geometry drift")
        return result


@dataclass
class _AggregateBucket:
    scorer_pixels: int = 0
    channel_blocks: int = 0
    exact_blocks: int = 0
    bounded_blocks: int = 0
    lower: _CompactStats = field(default_factory=_CompactStats)
    upper: _CompactStats = field(default_factory=_CompactStats)

    def add_pixel(self, results: Sequence[BlockProfileResult]) -> None:
        if not results:
            raise LatticeProfileError("pixel aggregation requires channel blocks")
        self.scorer_pixels += 1
        self.channel_blocks += len(results)
        for result in results:
            if result.exhaustive:
                self.exact_blocks += 1
            else:
                self.bounded_blocks += 1
            self.lower.add_count(result.cardinality_lower_bound)
            self.upper.add_count(result.cardinality_upper_bound)

    def add_bounds_batch(
        self,
        lower: np.ndarray,
        upper: np.ndarray,
        *,
        scorer_pixels: int,
    ) -> None:
        lower_array = np.asarray(lower)
        upper_array = np.asarray(upper)
        pixels = _integer(scorer_pixels, name="scorer_pixels", minimum=0)
        if lower_array.shape != upper_array.shape or lower_array.size < pixels:
            raise LatticeProfileError("bound batch geometry is malformed")
        if np.any(upper_array < lower_array):
            raise LatticeProfileError("bound batch upper cardinality is below lower")
        self.scorer_pixels += pixels
        self.channel_blocks += int(lower_array.size)
        # Source-witness root bounds are not exhaustive certificates, even when
        # lower and upper happen to coincide for a particular block.
        self.bounded_blocks += int(lower_array.size)
        self.lower.add_counts_array(lower_array)
        self.upper.add_counts_array(upper_array)

    def summary(self) -> dict[str, Any]:
        denominator = max(1, self.channel_blocks)
        return {
            "scorer_pixels": self.scorer_pixels,
            "rgb_channel_blocks": self.channel_blocks,
            "exact_blocks": self.exact_blocks,
            "bounded_blocks": self.bounded_blocks,
            "exact_count_fraction": self.exact_blocks / denominator,
            "bounded_fraction": self.bounded_blocks / denominator,
            "zero_lower_bound_anomalies": self.lower.zero_count,
            "lower_log2_count": self.lower.summary(),
            "upper_log2_count": self.upper.summary(),
        }

    def state(self) -> dict[str, Any]:
        return {
            "scorer_pixels": self.scorer_pixels,
            "channel_blocks": self.channel_blocks,
            "exact_blocks": self.exact_blocks,
            "bounded_blocks": self.bounded_blocks,
            "lower": self.lower.state(),
            "upper": self.upper.state(),
        }

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> _AggregateBucket:
        return cls(
            scorer_pixels=int(state["scorer_pixels"]),
            channel_blocks=int(state["channel_blocks"]),
            exact_blocks=int(state["exact_blocks"]),
            bounded_blocks=int(state["bounded_blocks"]),
            lower=_CompactStats.from_state(state["lower"]),
            upper=_CompactStats.from_state(state["upper"]),
        )


class StreamingProfileAggregator:
    """Compact global/per-class/named-stratum accumulation."""

    def __init__(self, *, n_classes: int, named_strata: Iterable[str] = ()):
        self.n_classes = _integer(n_classes, name="n_classes", minimum=1)
        names = tuple(sorted(set(named_strata)))
        if any(not isinstance(name, str) or not name for name in names):
            raise LatticeProfileError("named strata must be nonempty strings")
        self.global_bucket = _AggregateBucket()
        self.per_class = {index: _AggregateBucket() for index in range(self.n_classes)}
        self.strata = {name: _AggregateBucket() for name in names}

    def add_pixel(
        self,
        *,
        target_class: int,
        channel_results: Sequence[BlockProfileResult],
        strata: Iterable[str] = (),
    ) -> None:
        class_index = _integer(target_class, name="target_class", minimum=0)
        if class_index >= self.n_classes:
            raise LatticeProfileError("target_class is outside configured classes")
        results = tuple(channel_results)
        self.global_bucket.add_pixel(results)
        self.per_class[class_index].add_pixel(results)
        for name in sorted(set(strata)):
            if name in self.strata:
                self.strata[name].add_pixel(results)

    def add_bounds_batch(
        self,
        *,
        target_classes: np.ndarray,
        lower_bounds: np.ndarray,
        upper_bounds: np.ndarray,
        strata: Mapping[str, np.ndarray] | None = None,
    ) -> None:
        classes = np.asarray(target_classes)
        lower = np.asarray(lower_bounds)
        upper = np.asarray(upper_bounds)
        if classes.dtype.kind not in "iu" or classes.dtype.kind == "b":
            raise LatticeProfileError("target class batch must contain integers")
        if lower.shape != upper.shape or lower.shape[:-1] != classes.shape or lower.ndim < 2:
            raise LatticeProfileError("vectorized aggregate geometry is malformed")
        if np.any(classes < 0) or np.any(classes >= self.n_classes):
            raise LatticeProfileError("target class batch is outside configured classes")
        self.global_bucket.add_bounds_batch(lower, upper, scorer_pixels=int(classes.size))
        for class_index, bucket in self.per_class.items():
            mask = classes == class_index
            bucket.add_bounds_batch(
                lower[mask],
                upper[mask],
                scorer_pixels=int(np.count_nonzero(mask)),
            )
        for name, mask_value in ({} if strata is None else strata).items():
            if name not in self.strata:
                continue
            mask = np.asarray(mask_value)
            if mask.dtype.kind != "b" or mask.shape != classes.shape:
                raise LatticeProfileError(f"stratum {name!r} mask geometry is malformed")
            self.strata[name].add_bounds_batch(
                lower[mask],
                upper[mask],
                scorer_pixels=int(np.count_nonzero(mask)),
            )

    def merge(self, other: StreamingProfileAggregator) -> None:
        if self.n_classes != other.n_classes or set(self.strata) != set(other.strata):
            raise LatticeProfileError("aggregate merge geometry mismatch")

        def merge_bucket(left: _AggregateBucket, right: _AggregateBucket) -> None:
            left.scorer_pixels += right.scorer_pixels
            left.channel_blocks += right.channel_blocks
            left.exact_blocks += right.exact_blocks
            left.bounded_blocks += right.bounded_blocks
            for left_stats, right_stats in ((left.lower, right.lower), (left.upper, right.upper)):
                left_stats.count += right_stats.count
                left_stats.zero_count += right_stats.zero_count
                left_stats.total += right_stats.total
                left_stats.minimum = (
                    right_stats.minimum
                    if left_stats.minimum is None
                    else left_stats.minimum
                    if right_stats.minimum is None
                    else min(left_stats.minimum, right_stats.minimum)
                )
                left_stats.maximum = (
                    right_stats.maximum
                    if left_stats.maximum is None
                    else left_stats.maximum
                    if right_stats.maximum is None
                    else max(left_stats.maximum, right_stats.maximum)
                )
                left_stats.bins = [a + b for a, b in zip(left_stats.bins, right_stats.bins, strict=True)]

        merge_bucket(self.global_bucket, other.global_bucket)
        for index in self.per_class:
            merge_bucket(self.per_class[index], other.per_class[index])
        for name in self.strata:
            merge_bucket(self.strata[name], other.strata[name])

    def summary(self) -> dict[str, Any]:
        return {
            "global": self.global_bucket.summary(),
            "per_class": {str(index): bucket.summary() for index, bucket in self.per_class.items()},
            "named_strata": {name: bucket.summary() for name, bucket in self.strata.items()},
        }

    def state(self) -> dict[str, Any]:
        return {
            "n_classes": self.n_classes,
            "global": self.global_bucket.state(),
            "per_class": {str(index): bucket.state() for index, bucket in self.per_class.items()},
            "strata": {name: bucket.state() for name, bucket in self.strata.items()},
        }

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> StreamingProfileAggregator:
        n_classes = int(state["n_classes"])
        result = cls(n_classes=n_classes, named_strata=state["strata"].keys())
        result.global_bucket = _AggregateBucket.from_state(state["global"])
        result.per_class = {
            index: _AggregateBucket.from_state(state["per_class"][str(index)]) for index in range(n_classes)
        }
        result.strata = {name: _AggregateBucket.from_state(bucket) for name, bucket in state["strata"].items()}
        return result


def build_rd_row(
    *,
    selected_block_count: int,
    total_block_count: int,
    stream_accounting: Mapping[str, Any],
    axis: str,
    cache_scope: str,
    receiver_scope: str,
    mismatch_count: int | None = None,
    scorer_pixel_count: int | None = None,
    rate_scope_frames: Sequence[int] | None = None,
    scorer_scope_frames: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Build an honest Seg-side RD row; never substitutes a margin proxy."""

    selected = _integer(selected_block_count, name="selected_block_count", minimum=0)
    total = _integer(total_block_count, name="total_block_count", minimum=0)
    if selected > total:
        raise LatticeProfileError("selected block count exceeds total")
    row: dict[str, Any] = {
        "exactness_insistence_rule": (
            "candidate blocks satisfy the exact integer resize equation; bounded/unknown "
            "blocks are omitted rather than relabeled feasible"
        ),
        "selected_block_count": selected,
        "total_rgb_channel_blocks": total,
        "stream_bytes": dict(stream_accounting),
        "cache_scope": _identity(cache_scope, name="cache_scope"),
        "receiver_scope": _identity(receiver_scope, name="receiver_scope"),
        "axis": _identity(axis, name="axis"),
        "rate_scope_frames": None if rate_scope_frames is None else list(rate_scope_frames),
        "scorer_scope_frames": None if scorer_scope_frames is None else list(scorer_scope_frames),
    }
    if mismatch_count is None or scorer_pixel_count is None:
        row.update(
            {
                "scorer_custody": "NO_VERDICT_SCORER_CUSTODY",
                "frozen_segnet_mismatch_count": None,
                "d_seg": None,
            }
        )
        return row
    mismatches = _integer(mismatch_count, name="mismatch_count", minimum=0)
    pixels = _integer(scorer_pixel_count, name="scorer_pixel_count", minimum=1)
    if rate_scope_frames is None or scorer_scope_frames is None:
        raise LatticeProfileError("scored RD rows require explicit rate and scorer scopes")
    rate_scope = tuple(_integer(value, name="rate scope frame", minimum=0) for value in rate_scope_frames)
    scorer_scope = tuple(_integer(value, name="scorer scope frame", minimum=0) for value in scorer_scope_frames)
    if rate_scope != scorer_scope:
        raise LatticeProfileError("scored RD row refuses mixed scorer/rate scope")
    if mismatches > pixels:
        raise LatticeProfileError("mismatch count exceeds scorer pixels")
    row.update(
        {
            "scorer_custody": "FROZEN_SEGNET_ARGMAX_MEASURED",
            "frozen_segnet_mismatch_count": mismatches,
            "frozen_segnet_scorer_pixels": pixels,
            "d_seg": mismatches / pixels,
        }
    )
    return row


def noncorner_positive_control() -> dict[str, Any]:
    """Named fixture where corner-only and budget-as-infeasible are false."""

    coefficients = (2, 3)
    target = 5
    witness = (1, 1)
    corners = ((0, 0), (0, 255), (255, 0), (255, 255))
    return {
        "name": "noncorner_feasible_false_certificate_control",
        "coefficients": coefficients,
        "target_integer": target,
        "witness": witness,
        "witness_satisfies": sum(coefficient * value for coefficient, value in zip(coefficients, witness, strict=True))
        == target,
        "any_corner_satisfies": any(
            sum(coefficient * value for coefficient, value in zip(coefficients, corner, strict=True)) == target
            for corner in corners
        ),
    }


__all__ = [
    "BlockProfileResult",
    "DescriptionCostModel",
    "LatticeProfileError",
    "NoOpPosePlugin",
    "PoseFeasibilityPlugin",
    "PoseFilterDecision",
    "ProfileStatus",
    "SignedResidualCostModel",
    "SourceWitnessBounds",
    "StreamingProfileAggregator",
    "build_rd_row",
    "candidate_stream_accounting",
    "decode_candidate_stream",
    "encode_candidate_stream",
    "noncorner_positive_control",
    "profile_cache_key",
    "profile_integer_block",
    "vectorized_source_witness_bounds",
]
