# SPDX-License-Identifier: MIT
"""Planning-only dynamic sparse gate primitive for compiler-hint generation.

This borrows the useful MUDD-style idea, not the full transformer architecture:
use the current state to generate sparse mixture coefficients over candidate
sources or operations, then hand the selected operations to the deterministic
compiler path under explicit false authority.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from tac.optimization.byte_shaving_campaign import FALSE_AUTHORITY
from tac.optimization.inverse_steganalysis_operation_set_compiler import (
    OPERATION_SET_COMPILER_HINT_SCHEMA,
)
from tac.optimization.materializer_feedback import (
    DEFAULT_LOCAL_MATERIALIZER_AXIS,
    materializer_observation_feedback_rows,
)
from tac.optimization.proxy_candidate_contract import (
    ordered_unique,
    require_no_truthy_authority_fields,
)
from tac.score_composition import CANONICAL_RATE_DENOM_BYTES, CANONICAL_RATE_MULTIPLIER

DYNAMIC_SPARSE_SKIP_MIXTURE_SCHEMA = "dynamic_sparse_skip_mixture_oracle.v1"
DYNAMIC_SPARSE_GATE_OPERATION_SELECTION_SCHEMA = "dynamic_sparse_gate_operation_selection.v1"
DYNAMIC_SPARSE_CHANNEL_GATE_OPERATION_SELECTION_SCHEMA = (
    "dynamic_sparse_channel_gate_operation_selection.v1"
)
DYNAMIC_SPARSE_OBSERVATION_FEEDBACK_SELECTION_SCHEMA = (
    "dynamic_sparse_gate_observation_feedback_selection.v1"
)
DEFAULT_RMS_EPSILON = 1.0e-6
DEFAULT_RATE_SCORE_PER_BYTE = CANONICAL_RATE_MULTIPLIER / float(CANONICAL_RATE_DENOM_BYTES)
DEFAULT_OBSERVATION_CHANNEL_IDS = (
    "rate_saving",
    "receiver_proof",
    "runtime_efficiency",
)


class DynamicSparseGateOracleError(ValueError):
    """Raised when dynamic-gate inputs would produce ambiguous planner signal."""


def _array(value: Any, *, label: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    if not np.all(np.isfinite(arr)):
        raise DynamicSparseGateOracleError(f"{label} must be finite")
    return arr


def _as_float(value: Any, *, label: str, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise DynamicSparseGateOracleError(f"{label} must be numeric") from exc
    if not math.isfinite(parsed):
        raise DynamicSparseGateOracleError(f"{label} must be finite")
    return parsed


def _as_nonnegative_int(value: Any, *, label: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise DynamicSparseGateOracleError(f"{label} must be an integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise DynamicSparseGateOracleError(f"{label} must be an integer") from exc
    if parsed < 0:
        raise DynamicSparseGateOracleError(f"{label} must be nonnegative")
    return parsed


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray, str)):
        return [str(item) for item in value if str(item)]
    return [str(value)] if str(value) else []


def _ordered_sources(
    sources: Mapping[str, Any] | Sequence[Any],
    *,
    source_order: Sequence[str] | None,
) -> tuple[list[str], list[np.ndarray]]:
    if isinstance(sources, Mapping):
        order = [str(item) for item in (source_order or list(sources.keys()))]
        missing = [item for item in order if item not in sources]
        if missing:
            raise DynamicSparseGateOracleError(f"source_order has missing sources: {missing!r}")
        return order, [_array(sources[item], label=f"sources[{item!r}]") for item in order]
    order = [str(item) for item in (source_order or [f"source_{idx:04d}" for idx, _ in enumerate(sources)])]
    values = [_array(item, label=f"sources[{idx}]") for idx, item in enumerate(sources)]
    if len(order) != len(values):
        raise DynamicSparseGateOracleError("source_order length must match sources")
    return order, values


def rms_norm(x: Any, *, epsilon: float = DEFAULT_RMS_EPSILON) -> np.ndarray:
    """Return RMS-normalized ``x`` along the last dimension."""

    arr = _array(x, label="x")
    if arr.ndim < 1:
        raise DynamicSparseGateOracleError("x must have at least one dimension")
    eps = _as_float(epsilon, label="epsilon", default=DEFAULT_RMS_EPSILON)
    if eps <= 0.0:
        raise DynamicSparseGateOracleError("epsilon must be positive")
    denom = np.sqrt(np.mean(arr * arr, axis=-1, keepdims=True) + eps, dtype=np.float32)
    return arr / denom


def gelu(x: Any) -> np.ndarray:
    """GELU activation in NumPy, stable enough for parity tests and planning."""

    arr = _array(x, label="x")
    return np.float32(0.5) * arr * (
        np.float32(1.0)
        + np.tanh(np.float32(math.sqrt(2.0 / math.pi)) * (arr + np.float32(0.044715) * arr**3))
    )


def dynamic_sparse_skip_mixture(
    base: Any,
    sources: Mapping[str, Any] | Sequence[Any],
    *,
    w1: Any,
    w2: Any,
    b1: Any | None = None,
    b2: Any | None = None,
    source_order: Sequence[str] | None = None,
    epsilon: float = DEFAULT_RMS_EPSILON,
    residual_scale: float = 1.0,
    coefficient_mode: str = "linear",
) -> dict[str, Any]:
    """Apply a MUDD-style dynamic sparse mixture with zero-init no-op support.

    ``linear`` mode treats the second projection as signed residual mixture
    coefficients, so ``w2=0`` and ``b2=0`` produce an exact no-op. ``softmax_delta``
    subtracts a uniform prior from the softmax coefficients, giving the same
    zero-init no-op property while keeping bounded relative weights.
    """

    base_arr = _array(base, label="base")
    if base_arr.ndim < 1:
        raise DynamicSparseGateOracleError("base must have at least one dimension")
    names, source_arrays = _ordered_sources(sources, source_order=source_order)
    if not source_arrays:
        raise DynamicSparseGateOracleError("at least one source is required")
    for index, source in enumerate(source_arrays):
        if source.shape != base_arr.shape:
            raise DynamicSparseGateOracleError(
                f"source {names[index]!r} shape {source.shape} does not match base {base_arr.shape}"
            )

    w1_arr = _array(w1, label="w1")
    w2_arr = _array(w2, label="w2")
    width = int(base_arr.shape[-1])
    if w1_arr.ndim != 2 or w1_arr.shape[0] != width:
        raise DynamicSparseGateOracleError("w1 must have shape (width, hidden)")
    hidden_width = int(w1_arr.shape[1])
    if w2_arr.shape != (hidden_width, len(source_arrays)):
        raise DynamicSparseGateOracleError("w2 must have shape (hidden, source_count)")
    b1_arr = np.zeros((hidden_width,), dtype=np.float32) if b1 is None else _array(b1, label="b1")
    b2_arr = np.zeros((len(source_arrays),), dtype=np.float32) if b2 is None else _array(b2, label="b2")
    if b1_arr.shape != (hidden_width,):
        raise DynamicSparseGateOracleError("b1 must have shape (hidden,)")
    if b2_arr.shape != (len(source_arrays),):
        raise DynamicSparseGateOracleError("b2 must have shape (source_count,)")

    hidden = gelu(rms_norm(base_arr, epsilon=epsilon) @ w1_arr + b1_arr)
    logits = hidden @ w2_arr + b2_arr
    if coefficient_mode == "linear":
        coefficients = logits
    elif coefficient_mode == "softmax_delta":
        shifted = logits - np.max(logits, axis=-1, keepdims=True)
        exp = np.exp(shifted)
        probabilities = exp / np.sum(exp, axis=-1, keepdims=True)
        coefficients = probabilities - np.float32(1.0 / len(source_arrays))
    else:
        raise DynamicSparseGateOracleError(f"unsupported coefficient_mode: {coefficient_mode!r}")

    delta = np.zeros_like(base_arr)
    for index, source in enumerate(source_arrays):
        delta += coefficients[..., index, None] * source
    scale = _as_float(residual_scale, label="residual_scale", default=1.0)
    mixed = base_arr + np.float32(scale) * delta
    return {
        "schema": DYNAMIC_SPARSE_SKIP_MIXTURE_SCHEMA,
        "source_order": names,
        "coefficient_mode": coefficient_mode,
        "rms_epsilon": float(epsilon),
        "residual_scale": float(scale),
        "coefficients": coefficients,
        "delta": delta,
        "mixed": mixed,
        "zero_init_noop_proven": bool(np.allclose(delta, 0.0, atol=0.0, rtol=0.0)),
        "allowed_use": "candidate_generation_and_operation_ordering_only",
        "forbidden_use": "score_claim_or_promotion_or_rank_kill_authority",
        **FALSE_AUTHORITY,
    }


def _gate_scores(coefficients: Any) -> np.ndarray:
    coeff = _array(coefficients, label="coefficients")
    if coeff.ndim < 1:
        raise DynamicSparseGateOracleError("coefficients must have at least one dimension")
    if coeff.shape[-1] < 1:
        raise DynamicSparseGateOracleError("coefficients must include at least one source")
    if coeff.ndim == 1:
        return np.abs(coeff)
    axes = tuple(range(coeff.ndim - 1))
    return np.mean(np.abs(coeff), axis=axes)


def _channel_gate_scores(coefficients: Any) -> np.ndarray:
    coeff = _array(coefficients, label="coefficients")
    if coeff.ndim < 2:
        raise DynamicSparseGateOracleError(
            "channel coefficients must have at least channel and source dimensions"
        )
    if coeff.shape[-2] < 1 or coeff.shape[-1] < 1:
        raise DynamicSparseGateOracleError(
            "channel coefficients must include at least one channel and source"
        )
    if coeff.ndim == 2:
        return np.abs(coeff)
    axes = tuple(range(coeff.ndim - 2))
    return np.mean(np.abs(coeff), axis=axes)


def _operation_candidate(row: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    try:
        require_no_truthy_authority_fields(row, context=label)
    except ValueError as exc:
        raise DynamicSparseGateOracleError(str(exc)) from exc
    operation = dict(row)
    for key in FALSE_AUTHORITY:
        operation[key] = False
    return operation


def _candidate_source_id(row: Mapping[str, Any]) -> str:
    return str(
        row.get("dynamic_gate_source_id")
        or row.get("dynamic_sparse_gate_source_id")
        or row.get("source_id")
        or row.get("source")
        or ""
    )


def _candidate_channel_id(row: Mapping[str, Any]) -> str:
    return str(
        row.get("dynamic_gate_channel_id")
        or row.get("dynamic_sparse_gate_channel_id")
        or row.get("channel_id")
        or row.get("channel")
        or ""
    )


def _optional_finite_float(value: Any, *, label: str) -> float | None:
    if value in (None, ""):
        return None
    parsed = _as_float(value, label=label, default=0.0)
    return parsed


def _optional_int(value: Any) -> int | None:
    if value in (None, "") or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _first_nonempty_text(*values: Any) -> str:
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def _source_id_from_observation(row: Mapping[str, Any], *, index: int) -> str:
    for key in (
        "source_unit_ids",
        "backlog_keys",
        "work_ids",
        "source_selection_ids",
        "candidate_ids",
    ):
        values = _string_list(row.get(key))
        if values:
            return values[0]
    return _first_nonempty_text(
        row.get("source_unit_id"),
        row.get("backlog_key"),
        row.get("work_id"),
        row.get("candidate_id"),
        row.get("target_kind"),
        f"observation_{index:04d}",
    )


def _observation_saved_bytes(row: Mapping[str, Any]) -> int:
    for key in ("saved_bytes", "archive_delta_bytes", "realized_saved_bytes"):
        parsed = _optional_int(row.get(key))
        if parsed is not None:
            return parsed
    return 0


def _observation_rate_score(
    row: Mapping[str, Any],
    *,
    rate_score_per_byte: float,
) -> float:
    observed = _optional_finite_float(row.get("observed_rate_gain"), label="observed_rate_gain")
    if observed is None:
        observed = _optional_finite_float(row.get("observed_score_gain"), label="observed_score_gain")
    if observed is not None and observed > 0.0:
        return observed
    saved_bytes = _observation_saved_bytes(row)
    if row.get("rate_positive") is True and saved_bytes > 0:
        return float(saved_bytes) * rate_score_per_byte
    if row.get("savings_realized") is True and saved_bytes > 0:
        return float(saved_bytes) * rate_score_per_byte
    return 0.0


def _observation_channel_scores(
    row: Mapping[str, Any],
    *,
    channel_ids: Sequence[str],
    rate_score_per_byte: float,
) -> dict[str, float]:
    rate_score = _observation_rate_score(row, rate_score_per_byte=rate_score_per_byte)
    elapsed = _optional_finite_float(row.get("elapsed_seconds"), label="elapsed_seconds")
    receiver_ok = row.get("receiver_contract_satisfied") is True or row.get("inflate_parity_satisfied") is True
    queue_health_ok = row.get("queue_observation_health") is not False
    score_by_channel: dict[str, float] = {}
    for channel_id in channel_ids:
        if channel_id == "rate_saving":
            score = rate_score
        elif channel_id == "receiver_proof":
            score = rate_score if receiver_ok else 0.0
        elif channel_id == "runtime_efficiency":
            score = rate_score / max(float(elapsed or 1.0), 1.0) if rate_score > 0.0 else 0.0
        elif channel_id == "queue_health":
            score = rate_score if queue_health_ok else 0.0
        else:
            score = _optional_finite_float(row.get(channel_id), label=channel_id) or 0.0
        if not queue_health_ok and channel_id != "queue_health":
            score = 0.0
        score_by_channel[channel_id] = max(0.0, float(score))
    return score_by_channel


def _best_observation_channel(score_by_channel: Mapping[str, float]) -> tuple[str, float]:
    ranked = sorted(
        ((float(score), str(channel_id)) for channel_id, score in score_by_channel.items()),
        key=lambda item: (-item[0], item[1]),
    )
    return (ranked[0][1], ranked[0][0]) if ranked else ("", 0.0)


def _observation_candidate(
    row: Mapping[str, Any],
    *,
    index: int,
    source_id: str,
    channel_id: str,
    channel_scores: Mapping[str, float],
) -> dict[str, Any] | None:
    target_kind = _first_nonempty_text(row.get("target_kind"))
    if not target_kind:
        return None
    saved_bytes = max(0, _observation_saved_bytes(row))
    observation_id = _first_nonempty_text(row.get("observation_id"), f"observation_{index:04d}")
    candidate_id = _first_nonempty_text(row.get("candidate_id"), source_id)
    params: dict[str, Any] = {}
    if isinstance(row.get("params"), Mapping):
        params.update(dict(row["params"]))
    params["dynamic_sparse_observation_feedback"] = {
        "schema": DYNAMIC_SPARSE_OBSERVATION_FEEDBACK_SELECTION_SCHEMA,
        "observation_id": observation_id,
        "candidate_id": candidate_id,
        "source_id": source_id,
        "selected_channel_id": channel_id,
        "channel_scores": {str(key): float(value) for key, value in channel_scores.items()},
        "source_observation_schema": _first_nonempty_text(row.get("schema")),
        "observation_kind": _first_nonempty_text(row.get("observation_kind")),
        "axis": _first_nonempty_text(row.get("axis")),
        "queue_id": _first_nonempty_text(row.get("queue_id")),
        "experiment_id": _first_nonempty_text(row.get("experiment_id")),
        "step_id": _first_nonempty_text(row.get("step_id")),
        "source_path": _first_nonempty_text(row.get("source_path")),
        "saved_bytes": saved_bytes,
        "observed_rate_gain": float(channel_scores.get("rate_saving", 0.0)),
        **FALSE_AUTHORITY,
    }
    materializer = _first_nonempty_text(row.get("materializer"), row.get("materializer_id"))
    if materializer:
        params["materializer_id"] = materializer
    receiver_contract_kind = _first_nonempty_text(row.get("receiver_contract_kind"))
    candidate = {
        "unit_id": source_id,
        "operation_id": f"dynamic_sparse_observation_{source_id}",
        "target_kind": target_kind,
        "candidate_id": candidate_id,
        "candidate_saved_bytes": saved_bytes,
        "params": params,
        "dynamic_gate_channel_id": channel_id,
        "dynamic_gate_source_id": source_id,
        "blockers": ordered_unique(
            [
                *_string_list(row.get("readiness_blockers")),
                *_string_list(row.get("dispatch_blockers")),
                "dynamic_sparse_observation_feedback_candidate_generation_only",
            ]
        ),
        **FALSE_AUTHORITY,
    }
    if materializer:
        candidate["materializer"] = materializer
    if receiver_contract_kind:
        candidate["receiver_contract_kind"] = receiver_contract_kind
    return candidate


def _channel_candidate_grid(
    operation_candidates: Sequence[Mapping[str, Any]],
    *,
    channel_ids: Sequence[str],
    source_ids: Sequence[str],
) -> dict[tuple[int, int], Mapping[str, Any]]:
    explicit: dict[tuple[int, int], Mapping[str, Any]] = {}
    has_explicit_ids = any(
        _candidate_channel_id(row) or _candidate_source_id(row)
        for row in operation_candidates
    )
    if has_explicit_ids:
        channel_index = {str(value): index for index, value in enumerate(channel_ids)}
        source_index = {str(value): index for index, value in enumerate(source_ids)}
        for candidate_index, row in enumerate(operation_candidates):
            channel_id = _candidate_channel_id(row)
            source_id = _candidate_source_id(row)
            if channel_id not in channel_index or source_id not in source_index:
                raise DynamicSparseGateOracleError(
                    "operation_candidates with explicit channel/source ids must "
                    f"match channel_ids/source_ids: index {candidate_index}"
                )
            key = (channel_index[channel_id], source_index[source_id])
            if key in explicit:
                raise DynamicSparseGateOracleError(
                    f"duplicate operation candidate for channel/source pair {key}"
                )
            explicit[key] = row
        return explicit

    expected = len(channel_ids) * len(source_ids)
    if len(operation_candidates) != expected:
        raise DynamicSparseGateOracleError(
            "operation_candidates without explicit channel/source ids must have "
            "len(channel_ids) * len(source_ids) entries"
        )
    for flat_index, row in enumerate(operation_candidates):
        channel_index = flat_index // len(source_ids)
        source_index = flat_index % len(source_ids)
        explicit[(channel_index, source_index)] = row
    return explicit


def operation_set_compiler_hint_from_gate_scores(
    operation_candidates: Sequence[Mapping[str, Any]],
    coefficients: Any,
    *,
    operation_set_id: str,
    source_ids: Sequence[str] = (),
    max_operations: int | None = None,
    min_abs_gate: float = 0.0,
    lane_id: str | None = None,
    candidate_id: str | None = None,
    coefficient_mode: str = "linear",
) -> dict[str, Any]:
    """Convert dynamic-gate coefficients into an inverse-action compiler hint."""

    if not operation_set_id:
        raise DynamicSparseGateOracleError("operation_set_id is required")
    scores = _gate_scores(coefficients)
    if len(operation_candidates) != int(scores.shape[0]):
        raise DynamicSparseGateOracleError("operation_candidates length must match coefficient source count")
    limit = _as_nonnegative_int(max_operations, label="max_operations")
    threshold = _as_float(min_abs_gate, label="min_abs_gate", default=0.0)
    if threshold < 0.0:
        raise DynamicSparseGateOracleError("min_abs_gate must be nonnegative")
    source_labels = [str(item) for item in source_ids]
    if source_labels and len(source_labels) != len(operation_candidates):
        raise DynamicSparseGateOracleError("source_ids length must match operation_candidates")
    if not source_labels:
        source_labels = [f"source_{idx:04d}" for idx in range(len(operation_candidates))]

    ranked: list[tuple[float, int, dict[str, Any]]] = []
    for index, raw in enumerate(operation_candidates):
        candidate = _operation_candidate(raw, label=f"operation_candidates[{index}]")
        score = float(scores[index])
        if score < threshold:
            continue
        saved = _as_float(candidate.get("candidate_saved_bytes"), label="candidate_saved_bytes", default=0.0)
        ranked.append((score, index, {**candidate, "_saved_sort": saved}))
    ranked.sort(key=lambda item: (-item[0], -float(item[2].get("_saved_sort") or 0.0), item[1]))
    if limit is not None:
        ranked = ranked[:limit]
    if not ranked:
        raise DynamicSparseGateOracleError("no operation candidate survived gate selection")

    selected: list[dict[str, Any]] = []
    for rank, (score, index, candidate) in enumerate(ranked):
        candidate.pop("_saved_sort", None)
        params = dict(candidate.get("params")) if isinstance(candidate.get("params"), Mapping) else {}
        params["dynamic_sparse_gate"] = {
            "schema": DYNAMIC_SPARSE_GATE_OPERATION_SELECTION_SCHEMA,
            "source_id": source_labels[index],
            "source_index": index,
            "abs_mean_coefficient": score,
            "coefficient_mode": coefficient_mode,
            **FALSE_AUTHORITY,
        }
        selected.append(
            {
                **candidate,
                "operation_rank": rank,
                "dynamic_gate_source_id": source_labels[index],
                "dynamic_gate_abs_mean_coefficient": score,
                "params": params,
                "blockers": ordered_unique(
                    [
                        *_string_list(candidate.get("blockers")),
                        "dynamic_sparse_gate_candidate_generation_only",
                        "requires_materializer_contexts",
                        "requires_runtime_consumption_proof_before_exact_eval",
                    ]
                ),
                **FALSE_AUTHORITY,
            }
        )

    return {
        "schema": OPERATION_SET_COMPILER_HINT_SCHEMA,
        "source_schema": DYNAMIC_SPARSE_GATE_OPERATION_SELECTION_SCHEMA,
        "operation_set_id": operation_set_id,
        "lane_id": lane_id,
        "candidate_id": candidate_id,
        "selection_source": "dynamic_sparse_skip_mixture_gate_scores",
        "source_ids": source_labels,
        "selected_operations": selected,
        "allowed_use": "inverse_action_compiler_planning_handoff_only",
        "forbidden_use": "score_claim_or_promotion_or_rank_kill_authority",
        **FALSE_AUTHORITY,
    }


def operation_set_compiler_hint_from_channel_gate_scores(
    operation_candidates: Sequence[Mapping[str, Any]],
    coefficients: Any,
    *,
    operation_set_id: str,
    source_ids: Sequence[str],
    channel_ids: Sequence[str],
    max_operations: int | None = None,
    min_abs_gate: float = 0.0,
    lane_id: str | None = None,
    candidate_id: str | None = None,
    coefficient_mode: str = "linear",
    shared_projection_id: str | None = None,
    topology_id: str | None = None,
) -> dict[str, Any]:
    """Convert channel/source gate coefficients into a compiler hint.

    This keeps MUDD-style V/residual/VE-gate channels separate instead of
    averaging them into one source score, preserving the useful sparse topology
    signal for downstream materializer selection.
    """

    if not operation_set_id:
        raise DynamicSparseGateOracleError("operation_set_id is required")
    channel_labels = [str(item) for item in channel_ids]
    source_labels = [str(item) for item in source_ids]
    if not channel_labels:
        raise DynamicSparseGateOracleError("channel_ids must include at least one channel")
    if not source_labels:
        raise DynamicSparseGateOracleError("source_ids must include at least one source")
    if len(set(channel_labels)) != len(channel_labels):
        raise DynamicSparseGateOracleError("channel_ids must be unique")
    if len(set(source_labels)) != len(source_labels):
        raise DynamicSparseGateOracleError("source_ids must be unique")

    scores = _channel_gate_scores(coefficients)
    if scores.shape != (len(channel_labels), len(source_labels)):
        raise DynamicSparseGateOracleError(
            "coefficients channel/source dimensions must match channel_ids/source_ids"
        )
    candidate_grid = _channel_candidate_grid(
        operation_candidates,
        channel_ids=channel_labels,
        source_ids=source_labels,
    )
    limit = _as_nonnegative_int(max_operations, label="max_operations")
    threshold = _as_float(min_abs_gate, label="min_abs_gate", default=0.0)
    if threshold < 0.0:
        raise DynamicSparseGateOracleError("min_abs_gate must be nonnegative")

    ranked: list[tuple[float, float, int, int, dict[str, Any]]] = []
    for (channel_index, source_index), raw in candidate_grid.items():
        candidate = _operation_candidate(
            raw,
            label=f"operation_candidates[{channel_labels[channel_index]},{source_labels[source_index]}]",
        )
        score = float(scores[channel_index, source_index])
        if score < threshold:
            continue
        saved = _as_float(candidate.get("candidate_saved_bytes"), label="candidate_saved_bytes", default=0.0)
        ranked.append((score, saved, channel_index, source_index, candidate))
    ranked.sort(key=lambda item: (-item[0], -item[1], item[2], item[3]))
    if limit is not None:
        ranked = ranked[:limit]
    if not ranked:
        raise DynamicSparseGateOracleError("no channel operation candidate survived gate selection")

    selected: list[dict[str, Any]] = []
    for rank, (score, _saved, channel_index, source_index, candidate) in enumerate(ranked):
        channel_id = channel_labels[channel_index]
        source_id = source_labels[source_index]
        params = dict(candidate.get("params")) if isinstance(candidate.get("params"), Mapping) else {}
        params["dynamic_sparse_channel_gate"] = {
            "schema": DYNAMIC_SPARSE_CHANNEL_GATE_OPERATION_SELECTION_SCHEMA,
            "source_id": source_id,
            "source_index": source_index,
            "channel_id": channel_id,
            "channel_index": channel_index,
            "abs_mean_coefficient": score,
            "coefficient_mode": coefficient_mode,
            "shared_projection_id": shared_projection_id,
            "topology_id": topology_id,
            **FALSE_AUTHORITY,
        }
        selected.append(
            {
                **candidate,
                "operation_rank": rank,
                "dynamic_gate_source_id": source_id,
                "dynamic_gate_source_index": source_index,
                "dynamic_gate_channel_id": channel_id,
                "dynamic_gate_channel_index": channel_index,
                "dynamic_gate_abs_mean_coefficient": score,
                "params": params,
                "blockers": ordered_unique(
                    [
                        *_string_list(candidate.get("blockers")),
                        "dynamic_sparse_channel_gate_candidate_generation_only",
                        "requires_materializer_contexts",
                        "requires_runtime_consumption_proof_before_exact_eval",
                    ]
                ),
                **FALSE_AUTHORITY,
            }
        )

    return {
        "schema": OPERATION_SET_COMPILER_HINT_SCHEMA,
        "source_schema": DYNAMIC_SPARSE_CHANNEL_GATE_OPERATION_SELECTION_SCHEMA,
        "operation_set_id": operation_set_id,
        "lane_id": lane_id,
        "candidate_id": candidate_id,
        "selection_source": "dynamic_sparse_channel_gate_scores",
        "source_ids": source_labels,
        "channel_ids": channel_labels,
        "shared_projection_id": shared_projection_id,
        "topology_id": topology_id,
        "selected_operations": selected,
        "allowed_use": "inverse_action_compiler_planning_handoff_only",
        "forbidden_use": "score_claim_or_promotion_or_rank_kill_authority",
        **FALSE_AUTHORITY,
    }


def operation_set_compiler_hint_from_observation_feedback(
    observations: Sequence[Mapping[str, Any]],
    *,
    operation_set_id: str,
    channel_ids: Sequence[str] = DEFAULT_OBSERVATION_CHANNEL_IDS,
    max_operations: int | None = None,
    min_abs_gate: float = 0.0,
    lane_id: str | None = None,
    candidate_id: str | None = None,
    coefficient_mode: str = "observation_feedback",
    shared_projection_id: str | None = None,
    topology_id: str | None = None,
    rate_score_per_byte: float = DEFAULT_RATE_SCORE_PER_BYTE,
) -> dict[str, Any]:
    """Build a channel-gate compiler hint from real queue/materializer observations."""

    if not observations:
        raise DynamicSparseGateOracleError("observations must include at least one row")
    channels = [str(item) for item in channel_ids if str(item)]
    if not channels:
        raise DynamicSparseGateOracleError("channel_ids must include at least one channel")
    rate_per_byte = _as_float(
        rate_score_per_byte,
        label="rate_score_per_byte",
        default=DEFAULT_RATE_SCORE_PER_BYTE,
    )
    if rate_per_byte <= 0.0:
        raise DynamicSparseGateOracleError("rate_score_per_byte must be positive")

    source_ids: list[str] = []
    score_rows: list[dict[str, float]] = []
    candidates: list[dict[str, Any]] = []
    seen_sources: set[str] = set()
    for index, raw_observation in enumerate(observations):
        if not isinstance(raw_observation, Mapping):
            raise DynamicSparseGateOracleError(f"observations[{index}] must be an object")
        try:
            require_no_truthy_authority_fields(raw_observation, context=f"observations[{index}]")
        except ValueError as exc:
            raise DynamicSparseGateOracleError(str(exc)) from exc
        source_id = _source_id_from_observation(raw_observation, index=index)
        if source_id in seen_sources:
            source_id = f"{source_id}_{index:04d}"
        seen_sources.add(source_id)
        channel_scores = _observation_channel_scores(
            raw_observation,
            channel_ids=channels,
            rate_score_per_byte=rate_per_byte,
        )
        best_channel, best_score = _best_observation_channel(channel_scores)
        if best_score <= 0.0:
            continue
        candidate = _observation_candidate(
            raw_observation,
            index=index,
            source_id=source_id,
            channel_id=best_channel,
            channel_scores=channel_scores,
        )
        if candidate is None:
            continue
        source_ids.append(source_id)
        score_rows.append(channel_scores)
        candidates.append(candidate)

    if not candidates:
        raise DynamicSparseGateOracleError("no observation feedback row produced a selectable operation candidate")

    coefficients = np.zeros((len(channels), len(source_ids)), dtype=np.float32)
    for source_index, channel_scores in enumerate(score_rows):
        for channel_index, channel_id in enumerate(channels):
            coefficients[channel_index, source_index] = np.float32(channel_scores.get(channel_id, 0.0))
    hint = operation_set_compiler_hint_from_channel_gate_scores(
        candidates,
        coefficients,
        operation_set_id=operation_set_id,
        source_ids=source_ids,
        channel_ids=channels,
        max_operations=max_operations,
        min_abs_gate=min_abs_gate,
        lane_id=lane_id,
        candidate_id=candidate_id,
        coefficient_mode=coefficient_mode,
        shared_projection_id=shared_projection_id,
        topology_id=topology_id,
    )
    hint["selection_source"] = "dynamic_sparse_observation_feedback"
    hint["observation_feedback"] = {
        "schema": DYNAMIC_SPARSE_OBSERVATION_FEEDBACK_SELECTION_SCHEMA,
        "observation_count": len(observations),
        "selectable_observation_count": len(candidates),
        "rate_score_per_byte": rate_per_byte,
        "channel_ids": channels,
        "source_ids": source_ids,
        "allowed_use": "queue_materializer_feedback_candidate_generation_only",
        "forbidden_use": "score_claim_or_promotion_or_rank_kill_authority",
        **FALSE_AUTHORITY,
    }
    return hint


def operation_set_compiler_hint_from_materializer_feedback(
    materializer_feedback: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    *,
    operation_set_id: str,
    source_path: str | None = None,
    default_axis: str = DEFAULT_LOCAL_MATERIALIZER_AXIS,
    channel_ids: Sequence[str] = DEFAULT_OBSERVATION_CHANNEL_IDS,
    max_operations: int | None = None,
    min_abs_gate: float = 0.0,
    lane_id: str | None = None,
    candidate_id: str | None = None,
    coefficient_mode: str = "materializer_feedback",
    shared_projection_id: str | None = None,
    topology_id: str | None = None,
    rate_score_per_byte: float = DEFAULT_RATE_SCORE_PER_BYTE,
) -> dict[str, Any]:
    """Build a compiler hint directly from materializer manifests or sweep output."""

    rows = materializer_observation_feedback_rows(
        materializer_feedback,
        source_path=source_path,
        default_axis=default_axis,
        rate_score_per_byte=rate_score_per_byte,
    )
    if not rows:
        raise DynamicSparseGateOracleError(
            "materializer feedback produced no canonical observation rows"
        )
    hint = operation_set_compiler_hint_from_observation_feedback(
        rows,
        operation_set_id=operation_set_id,
        channel_ids=channel_ids,
        max_operations=max_operations,
        min_abs_gate=min_abs_gate,
        lane_id=lane_id,
        candidate_id=candidate_id,
        coefficient_mode=coefficient_mode,
        shared_projection_id=shared_projection_id,
        topology_id=topology_id,
        rate_score_per_byte=rate_score_per_byte,
    )
    hint["selection_source"] = "dynamic_sparse_materializer_feedback"
    hint["materializer_feedback"] = {
        "schema": "dynamic_sparse_materializer_feedback_selection.v1",
        "source_path": source_path,
        "artifact_count": 1 if isinstance(materializer_feedback, Mapping) else len(materializer_feedback),
        "normalized_observation_count": len(rows),
        "default_axis": default_axis,
        "allowed_use": "queue_materializer_feedback_candidate_generation_only",
        "forbidden_use": "score_claim_or_promotion_or_rank_kill_authority",
        **FALSE_AUTHORITY,
    }
    return hint


__all__ = [
    "DYNAMIC_SPARSE_CHANNEL_GATE_OPERATION_SELECTION_SCHEMA",
    "DYNAMIC_SPARSE_GATE_OPERATION_SELECTION_SCHEMA",
    "DYNAMIC_SPARSE_OBSERVATION_FEEDBACK_SELECTION_SCHEMA",
    "DYNAMIC_SPARSE_SKIP_MIXTURE_SCHEMA",
    "DynamicSparseGateOracleError",
    "dynamic_sparse_skip_mixture",
    "gelu",
    "operation_set_compiler_hint_from_channel_gate_scores",
    "operation_set_compiler_hint_from_gate_scores",
    "operation_set_compiler_hint_from_materializer_feedback",
    "operation_set_compiler_hint_from_observation_feedback",
    "rms_norm",
]
