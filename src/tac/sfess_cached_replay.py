# SPDX-License-Identifier: MIT
"""Dependency-light clean-room SFESS replay over a SHA-pinned objective table.

This module is deliberately scorer-free.  It can only read a previously measured
exact-enumeration JSONL and exposes search-time values through a hard counted oracle.
The estimator never receives the table and therefore cannot enumerate unseen states.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

PINNED_UGC64_SHA256 = "249c19af0b8c117412de491e944bcacb6194c870c9d9ec57d5c93b5e55f1a979"
_TABLE_SCHEMA = "sfess_cached_objective_table_v1"
_SNAPSHOT_SCHEMA = "sfess_fixed_k_search_snapshot_v2"
_VALID_QUERY_PURPOSES = frozenset(
    {"initial", "sfess_sample", "strict_exact_gate", "budget_padding", "degenerate_initial"}
)


class SFESSError(ValueError):
    """Fail-closed cached-replay contract violation."""


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _strict_mask(mask: Sequence[int] | np.ndarray, n_bits: int) -> np.ndarray:
    array = np.asarray(mask)
    if array.shape != (n_bits,):
        raise SFESSError(f"mask must have shape ({n_bits},), got {array.shape}")
    if not np.all((array == 0) | (array == 1)):
        raise SFESSError("mask must be binary")
    return array.astype(np.uint8, copy=True)


def little_endian_mask_index(mask: Sequence[int] | np.ndarray) -> int:
    """Return ``sum(mask[i] * 2**i)``; bit zero is the first JSON mask entry."""

    array = np.asarray(mask)
    if array.ndim != 1 or not np.all((array == 0) | (array == 1)):
        raise SFESSError("mask must be a one-dimensional binary vector")
    return sum(int(bit) << index for index, bit in enumerate(array.tolist()))


def _mask_bytes(mask: np.ndarray) -> bytes:
    return np.asarray([mask.size], dtype="<u4").tobytes() + mask.astype(
        np.uint8, copy=False
    ).tobytes(order="C")


def _state_sha(mask: np.ndarray, value: float) -> str:
    return cached_state_sha256(mask, value)


def cached_state_sha256(mask: Sequence[int] | np.ndarray, value: float) -> str:
    """Fingerprint one binary state with little-endian float64 value custody."""

    array = np.asarray(mask)
    if array.ndim != 1 or array.size == 0 or not np.all((array == 0) | (array == 1)):
        raise SFESSError("state fingerprint mask must be a non-empty binary vector")
    numeric_value = float(value)
    if not math.isfinite(numeric_value):
        raise SFESSError("state fingerprint value must be finite")
    checked = array.astype(np.uint8, copy=True)
    return _sha256(_mask_bytes(checked) + np.asarray([numeric_value], dtype="<f8").tobytes())


@dataclass(frozen=True)
class CachedObjectiveTable:
    """Validated cached objective with fingerprints over order and exact state values."""

    source_sha256: str
    n_bits: int
    state_count: int
    order_sha256: str
    objective_sha256: str
    _values: tuple[float, ...]
    _state_sha256s: tuple[str, ...]

    def value(self, mask: Sequence[int] | np.ndarray) -> float:
        checked = _strict_mask(mask, self.n_bits)
        return self._values[little_endian_mask_index(checked)]

    def state_sha256(self, mask: Sequence[int] | np.ndarray) -> str:
        checked = _strict_mask(mask, self.n_bits)
        return self._state_sha256s[little_endian_mask_index(checked)]


def load_cached_objective_jsonl(
    path: str | Path,
    expected_sha256: str = PINNED_UGC64_SHA256,
    n_bits: int = 6,
) -> CachedObjectiveTable:
    """Load the exact 64-state receipt, rejecting any custody or ordering drift."""

    if n_bits <= 0 or n_bits > 20:
        raise SFESSError("n_bits must be in [1, 20]")
    source = Path(path)
    raw = source.read_bytes()
    actual_sha256 = _sha256(raw)
    if actual_sha256 != expected_sha256:
        raise SFESSError(
            f"cached objective SHA mismatch: expected {expected_sha256}, got {actual_sha256}"
        )
    lines = raw.splitlines()
    expected_states = 1 << n_bits
    if len(lines) != expected_states:
        raise SFESSError(f"cached objective requires {expected_states} rows, got {len(lines)}")

    values: list[float] = []
    state_shas: list[str] = []
    order_payload = bytearray(_TABLE_SCHEMA.encode("ascii"))
    seen: set[int] = set()
    required = {
        "candidate_mask",
        "candidate_value",
        "estimator",
        "function_evals_after",
        "proposal_index",
    }
    for row_index, encoded in enumerate(lines):
        try:
            row = json.loads(encoded)
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise SFESSError(f"row {row_index} is not valid JSON") from error
        if not isinstance(row, dict) or not required.issubset(row):
            missing = sorted(required.difference(row if isinstance(row, dict) else ()))
            raise SFESSError(f"row {row_index} missing required fields: {missing}")
        if row["estimator"] != "exact_enumeration":
            raise SFESSError(f"row {row_index} is not exact_enumeration")
        if type(row["proposal_index"]) is not int or row["proposal_index"] != row_index:
            raise SFESSError(f"row {row_index} proposal_index/order mismatch")
        if type(row["function_evals_after"]) is not int or row["function_evals_after"] != row_index + 1:
            raise SFESSError(f"row {row_index} function_evals_after/order mismatch")
        mask = _strict_mask(row["candidate_mask"], n_bits)
        state_index = little_endian_mask_index(mask)
        if state_index != row_index:
            raise SFESSError(f"row {row_index} violates little-endian mask order")
        if state_index in seen:
            raise SFESSError(f"duplicate state {state_index}")
        seen.add(state_index)
        if isinstance(row["candidate_value"], bool):
            raise SFESSError(f"row {row_index} candidate_value is not numeric")
        try:
            value = float(row["candidate_value"])
        except (TypeError, ValueError) as error:
            raise SFESSError(f"row {row_index} candidate_value is not numeric") from error
        if not math.isfinite(value):
            raise SFESSError(f"row {row_index} candidate_value is nonfinite")
        values.append(value)
        state_sha = _state_sha(mask, value)
        state_shas.append(state_sha)
        order_payload.extend(np.asarray([state_index], dtype="<u8").tobytes())

    if seen != set(range(expected_states)):
        raise SFESSError("cached objective does not contain every state exactly once")
    objective_payload = bytearray(_TABLE_SCHEMA.encode("ascii"))
    objective_payload.extend(_sha256(bytes(order_payload)).encode("ascii"))
    for state_sha in state_shas:
        objective_payload.extend(state_sha.encode("ascii"))
    return CachedObjectiveTable(
        source_sha256=actual_sha256,
        n_bits=n_bits,
        state_count=expected_states,
        order_sha256=_sha256(bytes(order_payload)),
        objective_sha256=_sha256(bytes(objective_payload)),
        _values=tuple(values),
        _state_sha256s=tuple(state_shas),
    )


@dataclass(frozen=True)
class QueryRecord:
    call_index: int
    mask: tuple[int, ...]
    state_sha256: str
    value: float
    purpose: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "call_index": self.call_index,
            "mask": list(self.mask),
            "state_sha256": self.state_sha256,
            "value": self.value,
            "purpose": self.purpose,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any], n_bits: int) -> QueryRecord:
        required = {"call_index", "mask", "state_sha256", "value", "purpose"}
        if not required.issubset(payload):
            raise SFESSError("prior query record is missing required fields")
        if type(payload["call_index"]) is not int or payload["call_index"] < 1:
            raise SFESSError("prior query call_index must be an integer >= 1")
        if isinstance(payload["value"], bool):
            raise SFESSError("prior query value must be numeric")
        if not isinstance(payload["state_sha256"], str) or len(payload["state_sha256"]) != 64:
            raise SFESSError("prior query state_sha256 must be a SHA-256 string")
        if not isinstance(payload["purpose"], str) or not payload["purpose"]:
            raise SFESSError("prior query purpose must be a non-empty string")
        if payload["purpose"] not in _VALID_QUERY_PURPOSES:
            raise SFESSError(f"prior query purpose {payload['purpose']!r} is not registered")
        try:
            mask = _strict_mask(payload["mask"], n_bits)
            value = float(payload["value"])
        except (TypeError, ValueError, OverflowError) as error:
            raise SFESSError("prior query record has invalid field types") from error
        if not math.isfinite(value):
            raise SFESSError("prior query value is nonfinite")
        return cls(
            call_index=payload["call_index"],
            mask=tuple(int(x) for x in mask),
            state_sha256=payload["state_sha256"],
            value=value,
            purpose=payload["purpose"],
        )


class CountedCachedOracle:
    """Only search-time path to cached values; every authorized lookup consumes budget."""

    def __init__(
        self,
        table: CachedObjectiveTable,
        budget: int,
        authorize_lookup: Callable[[np.ndarray], bool],
        prior_records: Iterable[QueryRecord | dict[str, Any]] = (),
    ) -> None:
        if budget <= 0:
            raise SFESSError("oracle budget must be positive")
        if not callable(authorize_lookup):
            raise SFESSError("authorize_lookup must be callable")
        self.table = table
        self.budget = int(budget)
        self._authorize_lookup = authorize_lookup
        restored: list[QueryRecord] = []
        for raw_record in prior_records:
            record = (
                raw_record
                if isinstance(raw_record, QueryRecord)
                else QueryRecord.from_dict(raw_record, table.n_bits)
            )
            if record.call_index != len(restored) + 1:
                raise SFESSError("prior query records are not contiguous")
            mask = np.asarray(record.mask, dtype=np.uint8)
            if table.state_sha256(mask) != record.state_sha256:
                raise SFESSError("prior query state fingerprint mismatch")
            if table.value(mask) != record.value:
                raise SFESSError("prior query value mismatch")
            restored.append(record)
        if len(restored) > self.budget:
            raise SFESSError("prior records exceed oracle budget")
        self._records = restored

    @property
    def calls(self) -> int:
        return len(self._records)

    @property
    def records(self) -> tuple[QueryRecord, ...]:
        return tuple(self._records)

    def __call__(self, mask: Sequence[int] | np.ndarray, *, purpose: str) -> float:
        checked = _strict_mask(mask, self.table.n_bits)
        if not isinstance(purpose, str) or not purpose:
            raise SFESSError("cached objective query purpose must be a non-empty string")
        if self.calls >= self.budget:
            raise SFESSError("cached objective budget exhausted")
        if not bool(self._authorize_lookup(checked.copy())):
            raise SFESSError("cached objective lookup was not authorized")
        value = self.table.value(checked)
        self._records.append(
            QueryRecord(
                call_index=self.calls + 1,
                mask=tuple(int(x) for x in checked),
                state_sha256=self.table.state_sha256(checked),
                value=value,
                purpose=purpose,
            )
        )
        return value

    def restore_records(self, prior_records: Iterable[QueryRecord | dict[str, Any]]) -> None:
        """Restore a validated trace without replacing this oracle or its authorizer closure."""

        if self.calls != 0:
            raise SFESSError("oracle records can only be restored into a fresh oracle")
        validated = CountedCachedOracle(
            self.table,
            self.budget,
            self._authorize_lookup,
            prior_records=prior_records,
        )
        self._records = list(validated.records)


def poisson_binomial_pmf_dft(probabilities: Sequence[float] | np.ndarray) -> np.ndarray:
    """DFT-exact Poisson-binomial PMF.

    Derivation: Fernández and Williams (2010), *Closed-Form Expression for the
    Poisson-Binomial Probability Density Function*, DOI:10.1109/TAES.2010.5461658.
    """

    p = np.asarray(probabilities, dtype=np.float64)
    if p.ndim != 1 or not np.all(np.isfinite(p)):
        raise SFESSError("probabilities must be a finite vector")
    if p.size == 0:
        return np.ones(1, dtype=np.float64)
    if np.any((p < 0.0) | (p > 1.0)):
        raise SFESSError("probabilities must be in [0, 1]")
    count = p.size + 1
    angles = 2.0j * np.pi * np.arange(count, dtype=np.float64) / count
    roots = np.exp(angles)
    evaluations = np.prod((1.0 - p[:, None]) + p[:, None] * roots[None, :], axis=0)
    pmf = np.fft.fft(evaluations).real / count
    tolerance = 128.0 * np.finfo(np.float64).eps * max(1, p.size)
    if np.any(pmf < -tolerance):
        raise SFESSError("DFT Poisson-binomial PMF has material negative mass")
    pmf = np.maximum(pmf, 0.0)
    total = float(pmf.sum())
    if not math.isfinite(total) or total <= 0.0:
        raise SFESSError("DFT Poisson-binomial PMF has invalid total mass")
    return pmf / total


def conditional_inclusion_probabilities(
    logits: Sequence[float] | np.ndarray, k: int
) -> np.ndarray:
    p = _logistic_probabilities(logits)
    if not 0 <= k <= p.size:
        raise SFESSError("k must be in [0, n_bits]")
    denominator = poisson_binomial_pmf_dft(p)[k]
    if denominator <= np.finfo(np.float64).tiny:
        raise SFESSError("conditioning event has zero numerical mass")
    inclusion = np.empty(p.size, dtype=np.float64)
    for index in range(p.size):
        if k == 0:
            inclusion[index] = 0.0
        else:
            reduced = poisson_binomial_pmf_dft(np.delete(p, index))
            inclusion[index] = p[index] * reduced[k - 1] / denominator
    if not np.all(np.isfinite(inclusion)) or not np.isclose(inclusion.sum(), k, atol=1e-9):
        raise SFESSError("conditional inclusion probabilities failed normalization")
    return inclusion


def _logistic_probabilities(logits: Sequence[float] | np.ndarray) -> np.ndarray:
    theta = np.asarray(logits, dtype=np.float64)
    if theta.ndim != 1 or theta.size == 0 or not np.all(np.isfinite(theta)):
        raise SFESSError("logits must be a finite non-empty vector")
    p = np.empty_like(theta)
    positive = theta >= 0.0
    p[positive] = 1.0 / (1.0 + np.exp(-theta[positive]))
    exp_theta = np.exp(theta[~positive])
    p[~positive] = exp_theta / (1.0 + exp_theta)
    return p


def sample_conditional_bernoulli_k_subset(
    logits: Sequence[float] | np.ndarray,
    k: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw an exact conditional-Bernoulli k-subset without objective enumeration.

    The suffix recursion samples each conditional bit from Poisson-binomial masses.
    It is the sampling companion to Wijk, Vinuesa, and Azizpour (2024), *Revisiting
    Score Function Estimators for k-Subset Sampling*, arXiv:2407.16058.
    """

    p = _logistic_probabilities(logits)
    n_bits = p.size
    if not 0 <= k <= n_bits:
        raise SFESSError("k must be in [0, n_bits]")
    suffix = np.zeros((n_bits + 1, n_bits + 1), dtype=np.float64)
    suffix[n_bits, 0] = 1.0
    for index in range(n_bits - 1, -1, -1):
        suffix[index, : n_bits - index + 1] += (1.0 - p[index]) * suffix[
            index + 1, : n_bits - index + 1
        ]
        suffix[index, 1 : n_bits - index + 1] += p[index] * suffix[
            index + 1, : n_bits - index
        ]
    if suffix[0, k] <= np.finfo(np.float64).tiny:
        raise SFESSError("conditioning event has zero numerical mass")
    mask = np.zeros(n_bits, dtype=np.uint8)
    remaining = k
    for index in range(n_bits):
        if remaining == 0:
            break
        available = n_bits - index
        if remaining == available:
            mask[index:] = 1
            break
        mass_zero = (1.0 - p[index]) * suffix[index + 1, remaining]
        mass_one = p[index] * suffix[index + 1, remaining - 1]
        probability_one = mass_one / (mass_zero + mass_one)
        if rng.random() < probability_one:
            mask[index] = 1
            remaining -= 1
    if int(mask.sum()) != k:
        raise SFESSError("conditional sampler violated exact cardinality")
    return mask


def exact_k_subset_logit_score(
    mask: Sequence[int] | np.ndarray,
    logits: Sequence[float] | np.ndarray,
    k: int,
) -> np.ndarray:
    """Exact ``d log P(X | |X|=k) / d logits = X - E[X | |X|=k]``.

    Source: Wijk, Vinuesa, and Azizpour (2024), *Revisiting Score Function
    Estimators for k-Subset Sampling*, arXiv:2407.16058.
    """

    theta = np.asarray(logits, dtype=np.float64)
    checked = _strict_mask(mask, theta.size)
    if int(checked.sum()) != k:
        raise SFESSError("score mask does not have cardinality k")
    return checked.astype(np.float64) - conditional_inclusion_probabilities(theta, k)


@dataclass(frozen=True)
class SFESSGradientSample:
    gradient: np.ndarray
    masks: tuple[tuple[int, ...], ...]
    values: tuple[float, ...]
    scores: tuple[np.ndarray, ...]


def sfess_leave_one_out_gradient(
    objective: Callable[[np.ndarray], float],
    logits: Sequence[float] | np.ndarray,
    k: int,
    samples: int,
    rng: np.random.Generator,
) -> SFESSGradientSample:
    """Unbiased M-sample SFESS gradient with a leave-one-out control variate.

    Each baseline excludes its own sample and is therefore independent of that
    sample's zero-mean score.  The construction follows Wijk, Vinuesa, and
    Azizpour (2024), arXiv:2407.16058, without importing their software.
    """

    if samples < 2:
        raise SFESSError("leave-one-out SFESS requires at least two samples")
    theta = np.asarray(logits, dtype=np.float64)
    masks: list[np.ndarray] = []
    values: list[float] = []
    scores: list[np.ndarray] = []
    for _ in range(samples):
        mask = sample_conditional_bernoulli_k_subset(theta, k, rng)
        value = float(objective(mask.copy()))
        if not math.isfinite(value):
            raise SFESSError("objective returned a nonfinite value")
        masks.append(mask)
        values.append(value)
        scores.append(exact_k_subset_logit_score(mask, theta, k))
    value_array = np.asarray(values, dtype=np.float64)
    score_array = np.stack(scores)
    baselines = (value_array.sum() - value_array) / (samples - 1)
    gradient = np.mean((value_array - baselines)[:, None] * score_array, axis=0)
    return SFESSGradientSample(
        gradient=gradient,
        masks=tuple(tuple(int(x) for x in mask) for mask in masks),
        values=tuple(values),
        scores=tuple(score.copy() for score in scores),
    )


@dataclass(frozen=True)
class SFESSSearchResult:
    current_mask: tuple[int, ...]
    best_mask: tuple[int, ...]
    current_value: float
    best_value: float
    calls: int
    accepted: int
    padding: int
    query_records: tuple[QueryRecord, ...]
    complete: bool


class SFESSFixedKSearch:
    """Deterministic fixed-k one-swap search with strict exact-value retention."""

    def __init__(
        self,
        oracle: CountedCachedOracle,
        n_bits: int,
        k: int,
        samples_per_gradient: int,
        seed: int,
        comparison_noise_floor_s: float,
    ) -> None:
        if oracle.table.n_bits != n_bits:
            raise SFESSError("oracle/table n_bits mismatch")
        if not 0 < k < n_bits:
            raise SFESSError("swap search requires 0 < k < n_bits")
        if samples_per_gradient < 2:
            raise SFESSError("samples_per_gradient must be at least two")
        if isinstance(comparison_noise_floor_s, bool):
            raise SFESSError("comparison_noise_floor_s must be finite and nonnegative")
        try:
            checked_noise_floor = float(comparison_noise_floor_s)
        except (TypeError, ValueError) as error:
            raise SFESSError("comparison_noise_floor_s must be finite and nonnegative") from error
        if not math.isfinite(checked_noise_floor) or checked_noise_floor < 0.0:
            raise SFESSError("comparison_noise_floor_s must be finite and nonnegative")
        self.oracle = oracle
        self.n_bits = n_bits
        self.k = k
        self.samples_per_gradient = samples_per_gradient
        self.seed = int(seed)
        self.comparison_noise_floor_s = checked_noise_floor
        self.rng = np.random.default_rng(seed)
        # The pre-registered cached replay uses one constant uniform fixed-k
        # policy, p_i=k/n.  It is a distributional anchor, not a learning-rate
        # control surface; the current SFESS gradient proposes the swap directly.
        probability = k / n_bits
        self.logits = np.full(n_bits, math.log(probability / (1.0 - probability)))
        self.current_mask = np.zeros(n_bits, dtype=np.uint8)
        self.current_mask[:k] = 1
        self.current_value: float | None = None
        self.best_mask = self.current_mask.copy()
        self.best_value: float | None = None
        self.accepted = 0
        self.padding = 0

    def _snapshot_payload(self, eval_budget: int) -> dict[str, Any]:
        if self.current_value is None or self.best_value is None:
            raise SFESSError("cannot snapshot an uninitialized search")
        return {
            "schema": _SNAPSHOT_SCHEMA,
            "source_sha256": self.oracle.table.source_sha256,
            "objective_sha256": self.oracle.table.objective_sha256,
            "order_sha256": self.oracle.table.order_sha256,
            "n_bits": self.n_bits,
            "k": self.k,
            "samples_per_gradient": self.samples_per_gradient,
            "seed": self.seed,
            "comparison_noise_floor_s": self.comparison_noise_floor_s,
            "eval_budget": eval_budget,
            "logits": self.logits.tolist(),
            "current_mask": self.current_mask.tolist(),
            "current_value": self.current_value,
            "best_mask": self.best_mask.tolist(),
            "best_value": self.best_value,
            "accepted": self.accepted,
            "padding": self.padding,
            "rng_state": self.rng.bit_generator.state,
            "query_records": [record.to_dict() for record in self.oracle.records],
        }

    def _write_snapshot(self, path: Path, eval_budget: int) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(
            self._snapshot_payload(eval_budget), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_name, path)
        finally:
            if os.path.exists(temporary_name):
                os.unlink(temporary_name)

    def _restore_state_from_trace(self, records: tuple[QueryRecord, ...]) -> None:
        """Replay the deterministic state machine; never trust snapshot counters or RNG."""

        if not records:
            raise SFESSError("SFESS snapshot trace must contain the counted initial lookup")
        for index, record in enumerate(records, start=1):
            if record.call_index != index:
                raise SFESSError("SFESS snapshot trace call indices are not contiguous")
            if sum(record.mask) != self.k:
                raise SFESSError("SFESS snapshot trace violates fixed cardinality")
            mask = np.asarray(record.mask, dtype=np.uint8)
            if self.oracle.table.state_sha256(mask) != record.state_sha256:
                raise SFESSError("SFESS snapshot trace state fingerprint mismatch")
            if self.oracle.table.value(mask) != record.value:
                raise SFESSError("SFESS snapshot trace value mismatch")

        first = records[0]
        expected_initial = tuple(int(x) for x in self.current_mask)
        if first.purpose != "initial" or first.mask != expected_initial:
            raise SFESSError("SFESS snapshot trace initial lookup mismatch")
        self.current_value = first.value
        self.best_value = first.value
        self.best_mask = self.current_mask.copy()

        cursor = 1
        while cursor < len(records):
            next_record = records[cursor]
            if next_record.purpose == "budget_padding":
                remaining_budget = self.oracle.budget - cursor
                if remaining_budget >= self.samples_per_gradient + 1:
                    raise SFESSError("SFESS snapshot entered padding before the residual budget")
                for padding_record in records[cursor:]:
                    if padding_record.purpose != "budget_padding":
                        raise SFESSError("SFESS snapshot trace leaves padding phase")
                    if padding_record.mask != tuple(int(x) for x in self.current_mask):
                        raise SFESSError("SFESS snapshot padding mask is not the incumbent")
                    if padding_record.value != self.current_value:
                        raise SFESSError("SFESS snapshot padding value is not the incumbent")
                    self.padding += 1
                cursor = len(records)
                continue

            step_end = cursor + self.samples_per_gradient + 1
            if step_end > len(records):
                raise SFESSError("SFESS snapshot ends inside an estimator/gate step")
            sample_records = records[cursor : cursor + self.samples_per_gradient]
            gate_record = records[cursor + self.samples_per_gradient]
            if any(record.purpose != "sfess_sample" for record in sample_records):
                raise SFESSError("SFESS snapshot estimator sample schedule mismatch")
            if gate_record.purpose != "strict_exact_gate":
                raise SFESSError("SFESS snapshot strict-gate schedule mismatch")

            sample_index = 0

            def replay_objective(
                mask: np.ndarray, records_for_step: tuple[QueryRecord, ...] = sample_records
            ) -> float:
                nonlocal sample_index
                record = records_for_step[sample_index]
                generated = tuple(int(x) for x in mask)
                if generated != record.mask:
                    raise SFESSError("SFESS snapshot sample mask/RNG trace mismatch")
                sample_index += 1
                return record.value

            sample = sfess_leave_one_out_gradient(
                replay_objective,
                self.logits,
                self.k,
                self.samples_per_gradient,
                self.rng,
            )
            if sample_index != self.samples_per_gradient:
                raise SFESSError("SFESS snapshot sample trace was not fully replayed")
            selected = np.flatnonzero(self.current_mask == 1)
            unselected = np.flatnonzero(self.current_mask == 0)
            remove = int(selected[np.argmax(sample.gradient[selected])])
            add = int(unselected[np.argmin(sample.gradient[unselected])])
            proposal = self.current_mask.copy()
            proposal[remove] = 0
            proposal[add] = 1
            if gate_record.mask != tuple(int(x) for x in proposal):
                raise SFESSError("SFESS snapshot strict-gate proposal mismatch")
            if gate_record.value < self.current_value - self.comparison_noise_floor_s:
                self.current_mask = proposal
                self.current_value = gate_record.value
                self.best_mask = proposal.copy()
                self.best_value = gate_record.value
                self.accepted += 1
            cursor = step_end

    def run(
        self,
        eval_budget: int,
        snapshot_path: str | Path,
        *,
        stop_after_calls: int | None = None,
    ) -> SFESSSearchResult:
        if eval_budget != self.oracle.budget:
            raise SFESSError("eval_budget must equal the hard oracle budget")
        snapshot = Path(snapshot_path)
        if self.current_value is None:
            self.current_value = self.oracle(self.current_mask, purpose="initial")
            self.best_value = self.current_value
            self._write_snapshot(snapshot, eval_budget)
        while self.oracle.calls < eval_budget:
            if stop_after_calls is not None and self.oracle.calls >= stop_after_calls:
                break
            remaining = eval_budget - self.oracle.calls
            required = self.samples_per_gradient + 1
            if remaining < required:
                while self.oracle.calls < eval_budget:
                    self.oracle(self.current_mask, purpose="budget_padding")
                    self.padding += 1
                    if stop_after_calls is not None and self.oracle.calls >= stop_after_calls:
                        break
                self._write_snapshot(snapshot, eval_budget)
                continue

            sample = sfess_leave_one_out_gradient(
                lambda mask: self.oracle(mask, purpose="sfess_sample"),
                self.logits,
                self.k,
                self.samples_per_gradient,
                self.rng,
            )
            selected = np.flatnonzero(self.current_mask == 1)
            unselected = np.flatnonzero(self.current_mask == 0)
            remove = int(selected[np.argmax(sample.gradient[selected])])
            add = int(unselected[np.argmin(sample.gradient[unselected])])
            proposal = self.current_mask.copy()
            proposal[remove] = 0
            proposal[add] = 1
            proposal_value = self.oracle(proposal, purpose="strict_exact_gate")
            if proposal_value < self.current_value - self.comparison_noise_floor_s:
                self.current_mask = proposal
                self.current_value = proposal_value
                self.accepted += 1
                if self.best_value is None or proposal_value < self.best_value:
                    self.best_mask = proposal.copy()
                    self.best_value = proposal_value
            self._write_snapshot(snapshot, eval_budget)

        if self.current_value is None or self.best_value is None:
            raise SFESSError("search did not initialize")
        return SFESSSearchResult(
            current_mask=tuple(int(x) for x in self.current_mask),
            best_mask=tuple(int(x) for x in self.best_mask),
            current_value=self.current_value,
            best_value=self.best_value,
            calls=self.oracle.calls,
            accepted=self.accepted,
            padding=self.padding,
            query_records=self.oracle.records,
            complete=self.oracle.calls == eval_budget,
        )

    @classmethod
    def resume_from(
        cls,
        snapshot_path: str | Path,
        oracle: CountedCachedOracle,
        *,
        expected_k: int,
        expected_samples_per_gradient: int,
        expected_seed: int,
        expected_comparison_noise_floor_s: float,
    ) -> SFESSFixedKSearch:
        try:
            payload = json.loads(Path(snapshot_path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise SFESSError("invalid SFESS snapshot") from error
        if payload.get("schema") != _SNAPSHOT_SCHEMA:
            raise SFESSError("SFESS snapshot schema mismatch")
        required = {
            "source_sha256",
            "objective_sha256",
            "order_sha256",
            "n_bits",
            "k",
            "samples_per_gradient",
            "seed",
            "comparison_noise_floor_s",
            "eval_budget",
            "logits",
            "current_mask",
            "current_value",
            "best_mask",
            "best_value",
            "accepted",
            "padding",
            "rng_state",
            "query_records",
        }
        if not required.issubset(payload):
            raise SFESSError("SFESS snapshot is missing required fields")
        table = oracle.table
        for key, expected in (
            ("source_sha256", table.source_sha256),
            ("objective_sha256", table.objective_sha256),
            ("order_sha256", table.order_sha256),
        ):
            if payload.get(key) != expected:
                raise SFESSError(f"SFESS snapshot {key} mismatch")
        if not isinstance(payload["query_records"], list):
            raise SFESSError("SFESS snapshot query_records must be a list")
        records = tuple(
            QueryRecord.from_dict(row, table.n_bits)
            for row in payload["query_records"]
            if isinstance(row, dict)
        )
        if len(records) != len(payload["query_records"]):
            raise SFESSError("SFESS snapshot query record is not an object")
        if oracle.calls != 0 and oracle.records != records:
            raise SFESSError("resume oracle prior_records do not match snapshot")
        integer_fields = ("n_bits", "k", "samples_per_gradient", "seed", "eval_budget")
        for key in integer_fields:
            if type(payload[key]) is not int:
                raise SFESSError(f"SFESS snapshot {key} must be an integer")
        if payload["eval_budget"] != oracle.budget:
            raise SFESSError("SFESS snapshot budget mismatch")
        for key, expected in (
            ("k", expected_k),
            ("samples_per_gradient", expected_samples_per_gradient),
            ("seed", expected_seed),
        ):
            if type(expected) is not int:
                raise SFESSError(f"expected SFESS {key} must be an integer")
            if payload[key] != expected:
                raise SFESSError(f"SFESS snapshot {key} mismatch")
        if isinstance(expected_comparison_noise_floor_s, bool):
            raise SFESSError("expected comparison noise floor must be finite and nonnegative")
        try:
            expected_floor = float(expected_comparison_noise_floor_s)
            declared_floor = float(payload["comparison_noise_floor_s"])
        except (TypeError, ValueError) as error:
            raise SFESSError("SFESS snapshot comparison noise floor is invalid") from error
        if (
            not math.isfinite(expected_floor)
            or expected_floor < 0.0
            or isinstance(payload["comparison_noise_floor_s"], bool)
            or not math.isfinite(declared_floor)
            or declared_floor < 0.0
        ):
            raise SFESSError("SFESS snapshot comparison noise floor is invalid")
        if declared_floor != expected_floor:
            raise SFESSError("SFESS snapshot comparison noise floor mismatch")
        search = cls(
            oracle,
            payload["n_bits"],
            payload["k"],
            payload["samples_per_gradient"],
            payload["seed"],
            expected_floor,
        )
        declared_logits = np.asarray(payload["logits"], dtype=np.float64)
        if declared_logits.shape != (search.n_bits,) or not np.all(np.isfinite(declared_logits)):
            raise SFESSError("SFESS snapshot logits invalid")
        if not np.array_equal(declared_logits, search.logits):
            raise SFESSError("SFESS snapshot changed the registered constant logits")
        declared_current_mask = _strict_mask(payload["current_mask"], search.n_bits)
        declared_best_mask = _strict_mask(payload["best_mask"], search.n_bits)
        if int(declared_current_mask.sum()) != search.k or int(declared_best_mask.sum()) != search.k:
            raise SFESSError("SFESS snapshot violates fixed cardinality")
        if isinstance(payload["current_value"], bool) or isinstance(payload["best_value"], bool):
            raise SFESSError("SFESS snapshot values must be numeric")
        declared_current_value = float(payload["current_value"])
        declared_best_value = float(payload["best_value"])
        if not math.isfinite(declared_current_value) or not math.isfinite(declared_best_value):
            raise SFESSError("SFESS snapshot contains nonfinite values")
        if type(payload["accepted"]) is not int or type(payload["padding"]) is not int:
            raise SFESSError("SFESS snapshot accepted/padding counters must be integers")
        if payload["accepted"] < 0 or payload["padding"] < 0:
            raise SFESSError("SFESS snapshot accepted/padding counters must be nonnegative")

        search._restore_state_from_trace(records)
        if not np.array_equal(declared_current_mask, search.current_mask):
            raise SFESSError("SFESS snapshot current mask disagrees with replayed trace")
        if not np.array_equal(declared_best_mask, search.best_mask):
            raise SFESSError("SFESS snapshot best mask disagrees with replayed trace")
        if declared_current_value != search.current_value:
            raise SFESSError("SFESS snapshot current value disagrees with replayed trace")
        if declared_best_value != search.best_value:
            raise SFESSError("SFESS snapshot best value disagrees with replayed trace")
        if payload["accepted"] != search.accepted or payload["padding"] != search.padding:
            raise SFESSError("SFESS snapshot counters disagree with replayed trace")
        try:
            declared_rng_state = json.dumps(payload["rng_state"], sort_keys=True, separators=(",", ":"))
            replayed_rng_state = json.dumps(
                search.rng.bit_generator.state, sort_keys=True, separators=(",", ":")
            )
        except (TypeError, ValueError) as error:
            raise SFESSError("SFESS snapshot RNG state invalid") from error
        if declared_rng_state != replayed_rng_state:
            raise SFESSError("SFESS snapshot RNG state disagrees with replayed trace")
        if oracle.calls == 0:
            oracle.restore_records(records)
        elif oracle.records != records:
            raise SFESSError("resume oracle prior_records do not match snapshot")
        return search
