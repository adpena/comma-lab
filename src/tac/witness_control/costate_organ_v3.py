# SPDX-License-Identifier: MIT
"""Deterministic rank-sharpening primitives for exact-anchor costate ORGAN v3.

This module is advisory and contains no trainer, process, provider, scorer, or
frontier mutation surface.  Predictions have zero learned parameters.  The
only write surface is an explicitly requested, fcntl-locked append to the
realized-DeltaS corpus.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

SCHEMA = "costate_organ_rank_sharpen.v3"
CORPUS_SCHEMA = "costate_realized_delta_backtest_row.v1"
R1B7_SCHEMA = "r1b7_uint8_survival_carrier_measurement.v1"
POOL_EQUATION_ID = "witness_measured_reverse_waterfill_v1"
EMA_EQUATION_ID = "ema_decay_run_geometry_v1"
COMPOSITION_EQUATION_ID = "costate_v3_rank_sharpen_composition_v1"
R1B7_RECEIPT_SHA256 = "61f3d03930ac765b3ad5a287cbff29a3073c800eb5a5f2b98b8a701bc086d03c"
EMA_DECAY_MEASURED = 0.997
REFERENCE_PAIR_COUNT = 600
DEFAULT_CORPUS_PATH = Path(".omx/research/costate_realized_delta_backtest_corpus.jsonl")
REPO_ROOT = Path(__file__).resolve().parents[3]

R1B7_STAGE_ORDER = (
    "killed_at_uint8",
    "killed_at_resize_dilution",
    "killed_at_stem",
    "killed_at_head_same_rival",
    "killed_at_head_wrong_rival",
    "survived_but_collateral",
    "survived_clean",
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite(name: str, value: Any) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _nonnegative(name: str, value: Any) -> float:
    number = _finite(name, value)
    if number < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return number


def _jsonable(value: Any) -> None:
    try:
        encoded = json.dumps(value, allow_nan=False, sort_keys=True)
    except (TypeError, ValueError) as exc:
        raise ValueError("value must be finite JSON") from exc
    if not encoded:
        raise ValueError("value must be JSON encodable")


@dataclass(frozen=True)
class StageSurvivalDistribution:
    """Finite-sample r1b7 survival probabilities.

    The posterior mean uses the pre-registered Jeffreys ``Beta(1/2, 1/2)``
    smoothing rule.  It keeps observed zero-death stages below one without
    inventing an unmeasured failure count.
    """

    total: int
    through_uint8: float
    through_resize: float
    through_stem: float
    through_head: float
    clean: float
    collateral: float
    histogram: Mapping[str, int]
    smoothing: str = "Jeffreys Beta(1/2,1/2) posterior mean"

    def probability(self, route_stage: str) -> float:
        values = {
            "scorer_space": 1.0,
            "temporal_stop": 1.0,
            "uint8": self.through_uint8,
            "resize": self.through_resize,
            "stem": self.through_stem,
            "head": self.through_head,
            "clean": self.clean,
        }
        try:
            return values[str(route_stage)]
        except KeyError as exc:
            raise ValueError(f"unknown realization route stage: {route_stage!r}") from exc


def stage_survival_distribution(histogram: Mapping[str, Any]) -> StageSurvivalDistribution:
    if set(histogram) != set(R1B7_STAGE_ORDER):
        raise ValueError("r1b7 stage histogram keys are not canonical")
    counts: dict[str, int] = {}
    for key in R1B7_STAGE_ORDER:
        value = histogram[key]
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError("r1b7 stage counts must be nonnegative integers")
        counts[key] = value
    total = sum(counts.values())
    if total <= 0:
        raise ValueError("r1b7 stage histogram must be nonempty")

    def posterior(survivors: int) -> float:
        return (survivors + 0.5) / (total + 1.0)

    through_uint8_count = total - counts["killed_at_uint8"]
    through_resize_count = through_uint8_count - counts["killed_at_resize_dilution"]
    through_stem_count = through_resize_count - counts["killed_at_stem"]
    through_head_count = counts["survived_but_collateral"] + counts["survived_clean"]
    return StageSurvivalDistribution(
        total=total,
        through_uint8=posterior(through_uint8_count),
        through_resize=posterior(through_resize_count),
        through_stem=posterior(through_stem_count),
        through_head=posterior(through_head_count),
        clean=posterior(counts["survived_clean"]),
        collateral=posterior(counts["survived_but_collateral"]),
        histogram=counts,
    )


def load_r1b7_survival(path: str | Path) -> StageSurvivalDistribution:
    receipt_path = Path(path)
    if sha256_file(receipt_path) != R1B7_RECEIPT_SHA256:
        raise ValueError("r1b7 receipt SHA-256 drifted")
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("schema") != R1B7_SCHEMA:
        raise ValueError("r1b7 receipt schema drifted")
    autopsy = receipt.get("stage_autopsy", {})
    distribution = stage_survival_distribution(autopsy.get("histogram", {}))
    if distribution.total != autopsy.get("histogram_total"):
        raise ValueError("r1b7 histogram total drifted")
    return distribution


def graded_realizability(
    *,
    distribution: StageSurvivalDistribution,
    route_stage: str,
    strength: float = 1.0,
    formulation_valid: bool = True,
    apparatus_valid: bool = True,
) -> dict[str, Any]:
    """Return a route-specific survival probability without target leakage."""

    scale = _nonnegative("strength", strength)
    if scale > 1.0:
        raise ValueError("strength must be <= 1")
    base = distribution.probability(route_stage)
    if not apparatus_valid:
        value, status = 0.0, "apparatus_invalid"
    elif not formulation_valid:
        value, status = 0.0, "formulation_scoped_negative"
    else:
        value, status = base * scale, "r1b7_stage_probability"
    return {
        "value": value,
        "base_probability": base,
        "strength": scale,
        "route_stage": route_stage,
        "status": status,
        "receipt_sha256": R1B7_RECEIPT_SHA256,
        "histogram_total": distribution.total,
        "learned_parameters": 0,
    }


def row_route_stage(row: Mapping[str, Any]) -> str:
    context = row.get("factor_context", {})
    if isinstance(context, Mapping) and context.get("realization_stage"):
        return str(context["realization_stage"])
    if row.get("corpus") == "#205_asof_trajectory":
        return "temporal_stop"
    return "clean"


def row_strength(row: Mapping[str, Any], *, design_realizability: float) -> float:
    if row.get("corpus") == "#205_asof_trajectory":
        return 1.0
    factors = row.get("factors", {})
    if not isinstance(factors, Mapping):
        raise ValueError("row factors must be a mapping")
    old = _nonnegative("realizability", factors.get("realizability", 0.0))
    if design_realizability <= 0.0:
        raise ValueError("design_realizability must be positive")
    return min(old / design_realizability, 1.0)


def sharpen_realizability_row(
    row: Mapping[str, Any],
    *,
    distribution: StageSurvivalDistribution,
    design_realizability: float,
) -> dict[str, Any]:
    context = row.get("factor_context", {})
    formulation_valid = not (isinstance(context, Mapping) and context.get("formulation_valid") is False)
    result = graded_realizability(
        distribution=distribution,
        route_stage=row_route_stage(row),
        strength=row_strength(row, design_realizability=design_realizability),
        formulation_valid=formulation_valid,
        apparatus_valid=bool(row.get("apparatus_valid", False)),
    )
    factors = row.get("factors", {})
    result["lambda"] = (
        _nonnegative("exact_gap", factors.get("exact_gap", 0.0))
        * _nonnegative("visibility", factors.get("visibility", 0.0))
        * result["value"]
        * _nonnegative("byte_price", factors.get("byte_price", 0.0))
    )
    return result


def opportunity_claims(row: Mapping[str, Any], raw_lambda: float) -> dict[str, float]:
    """Map the typed C2 intervention vocabulary onto exclusive opportunity pools."""

    raw = _nonnegative("raw_lambda", raw_lambda)
    row_id = str(row.get("id", ""))
    if row.get("corpus") == "#205_asof_trajectory":
        return {f"temporal_stop:{row_id}": raw}
    vehicle = str(row.get("vehicle", "unknown"))
    variant = str(row.get("variant", "unknown"))
    prefix = f"c2:{vehicle}"
    if variant == "oneside_lane":
        return {f"{prefix}:road_lane_boundary": raw}
    if variant in {"oneside_movable", "movable_meancolor"}:
        return {f"{prefix}:movable_boundary": raw}
    if variant == "oneside_shallow":
        # v2's exact-gap custody is 0.001098 for Road-Lane and 0.004937 for
        # the movable/far/near/edge group.  Preserve that measured split.
        lane_gap, movable_gap = 0.001098, 0.004937
        total = lane_gap + movable_gap
        return {
            f"{prefix}:road_lane_boundary": raw * lane_gap / total,
            f"{prefix}:movable_boundary": raw * movable_gap / total,
        }
    return {f"{prefix}:unmapped:{variant}": raw}


def pool_kkt_marginals(rows: Sequence[Mapping[str, Any]], raw_values: Sequence[float]) -> list[dict[str, Any]]:
    """Compute KKT marginals against current candidate-set pool ceilings.

    Pool ceilings are the largest typed claim offered for that pool in this
    candidate set.  This represents the current allocation surface: alternative
    interventions compete for one debt pool instead of summing independent
    full-debt claims.
    """

    if len(rows) != len(raw_values):
        raise ValueError("rows and raw_values length mismatch")
    claims = [opportunity_claims(row, raw) for row, raw in zip(rows, raw_values, strict=True)]
    ceilings: dict[str, float] = defaultdict(float)
    for claim in claims:
        for pool, value in claim.items():
            ceilings[pool] = max(ceilings[pool], value)
    remaining = dict(ceilings)
    order = sorted(
        range(len(rows)),
        key=lambda index: (-_nonnegative("raw_lambda", raw_values[index]), str(rows[index].get("id", ""))),
    )
    out: list[dict[str, Any] | None] = [None] * len(rows)
    for index in order:
        consumed: dict[str, float] = {}
        for pool, claim in claims[index].items():
            value = min(claim, remaining[pool])
            consumed[pool] = value
            remaining[pool] -= value
        out[index] = {
            "value": sum(consumed.values()),
            "raw_lambda": float(raw_values[index]),
            "claims": claims[index],
            "consumed": consumed,
            "pool_remaining_after": {pool: remaining[pool] for pool in consumed},
            "pool_ceilings": {pool: ceilings[pool] for pool in consumed},
            "equation_id": POOL_EQUATION_ID,
            "same_pool_addition_forbidden": True,
        }
    return [dict(value) for value in out if value is not None]


def ema_response(*, decay: float, updates: int) -> float:
    value = _finite("decay", decay)
    if not 0.0 < value < 1.0:
        raise ValueError("decay must be in (0,1)")
    if not isinstance(updates, int) or isinstance(updates, bool) or updates <= 0:
        raise ValueError("updates must be a positive integer")
    return 1.0 - value**updates


def denoise_realized_target(row: Mapping[str, Any]) -> dict[str, Any]:
    """EMA de-lag and apparatus precision weighting on the realized side only."""

    observed = _finite("realized_benefit_s", row.get("realized_benefit_s"))
    if not bool(row.get("apparatus_valid", False)):
        return {
            "value": observed,
            "weight": 0.0,
            "status": "apparatus_invalid_zero_weight",
            "equation_id": EMA_EQUATION_ID,
        }
    if row.get("corpus") == "#205_asof_trajectory":
        epochs = row.get("source_epochs")
        if not isinstance(epochs, Sequence) or isinstance(epochs, (str, bytes)) or len(epochs) != 2:
            return {
                "value": observed,
                "weight": 0.0,
                "status": "ema_geometry_uncustodied_zero_weight",
                "equation_id": EMA_EQUATION_ID,
            }
        updates_float = _finite("EMA update horizon", epochs[1]) - _finite("EMA start", epochs[0])
        updates = int(updates_float)
        if updates_float != updates or updates <= 0:
            raise ValueError("EMA source epoch horizon must be a positive integer")
        response = ema_response(decay=EMA_DECAY_MEASURED, updates=updates)
        return {
            "value": observed / response,
            "weight": response * response,
            "status": "constant_decay_inverse_response",
            "observed": observed,
            "decay": EMA_DECAY_MEASURED,
            "updates": updates,
            "response": response,
            "noise_amplification": 1.0 / (response * response),
            "equation_id": EMA_EQUATION_ID,
        }
    scope = str(row.get("source_scope", ""))
    sample_count = REFERENCE_PAIR_COUNT
    if "120 frames of n600" in scope or "stride-5" in scope:
        sample_count = 120
    weight = sample_count / REFERENCE_PAIR_COUNT
    return {
        "value": observed,
        "weight": weight,
        "status": "inverse_variance_pair_fraction",
        "sample_count": sample_count,
        "reference_pair_count": REFERENCE_PAIR_COUNT,
        "equation_id": EMA_EQUATION_ID,
    }


def average_ranks(values: Sequence[Any]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(order):
        end = cursor + 1
        while end < len(order) and values[order[end]] == values[order[cursor]]:
            end += 1
        rank = (cursor + 1 + end) / 2.0
        for index in order[cursor:end]:
            ranks[index] = rank
        cursor = end
    return ranks


def weighted_pearson(a: Sequence[float], b: Sequence[float], weights: Sequence[float]) -> float:
    if not (len(a) == len(b) == len(weights)) or len(a) < 2:
        return float("nan")
    w = [_nonnegative("weight", value) for value in weights]
    total = sum(w)
    if total <= 0.0:
        return float("nan")
    am = sum(weight * value for weight, value in zip(w, a, strict=True)) / total
    bm = sum(weight * value for weight, value in zip(w, b, strict=True)) / total
    covariance = sum(weight * (left - am) * (right - bm) for weight, left, right in zip(w, a, b, strict=True))
    avar = sum(weight * (value - am) ** 2 for weight, value in zip(w, a, strict=True))
    bvar = sum(weight * (value - bm) ** 2 for weight, value in zip(w, b, strict=True))
    denominator = math.sqrt(avar * bvar)
    return 0.0 if denominator == 0.0 else covariance / denominator


def spearman_from_keys(
    prediction_keys: Sequence[Any], targets: Sequence[float], weights: Sequence[float] | None = None
) -> float:
    target_ranks = average_ranks(targets)
    prediction_ranks = average_ranks(prediction_keys)
    active_weights = [1.0] * len(targets) if weights is None else list(weights)
    return weighted_pearson(prediction_ranks, target_ranks, active_weights)


def top_k_precision(prediction_keys: Sequence[Any], targets: Sequence[float], *, k: int = 8) -> float:
    if len(prediction_keys) != len(targets) or not 0 < k <= len(targets):
        raise ValueError("invalid top-k metric inputs")
    order = sorted(range(len(targets)), key=lambda index: prediction_keys[index], reverse=True)
    return sum(targets[index] > 0.0 for index in order[:k]) / k


def ndcg_at_k(
    prediction_keys: Sequence[Any],
    targets: Sequence[float],
    *,
    weights: Sequence[float] | None = None,
    k: int = 8,
) -> float:
    if len(prediction_keys) != len(targets) or not 0 < k <= len(targets):
        raise ValueError("invalid NDCG inputs")
    active_weights = [1.0] * len(targets) if weights is None else list(weights)
    gains = [
        max(_finite("target", target), 0.0) * _nonnegative("weight", weight)
        for target, weight in zip(targets, active_weights, strict=True)
    ]

    def dcg(order: Sequence[int]) -> float:
        return sum(gains[index] / math.log2(rank + 2.0) for rank, index in enumerate(order[:k]))

    predicted = sorted(range(len(targets)), key=lambda index: prediction_keys[index], reverse=True)
    ideal = sorted(range(len(targets)), key=lambda index: gains[index], reverse=True)
    denominator = dcg(ideal)
    return 0.0 if denominator == 0.0 else dcg(predicted) / denominator


def tie_pairs(keys: Sequence[Any]) -> int:
    counts: dict[Any, int] = defaultdict(int)
    for key in keys:
        counts[key] += 1
    return sum(count * (count - 1) // 2 for count in counts.values() if count > 1)


def tie_summary(keys: Sequence[Any]) -> dict[str, int]:
    counts: dict[Any, int] = defaultdict(int)
    for key in keys:
        counts[key] += 1
    tied = [count for count in counts.values() if count > 1]
    return {
        "tie_pairs": sum(count * (count - 1) // 2 for count in tied),
        "tied_rows": sum(tied),
        "tie_groups": len(tied),
        "largest_tie_group": max(tied, default=1),
    }


def rank_metrics(
    prediction_keys: Sequence[Any],
    targets: Sequence[float],
    *,
    weights: Sequence[float] | None = None,
    ids: Sequence[str] | None = None,
    k: int = 8,
) -> dict[str, Any]:
    active_weights = [1.0] * len(targets) if weights is None else list(weights)
    order = sorted(range(len(targets)), key=lambda index: prediction_keys[index], reverse=True)
    return {
        "spearman": spearman_from_keys(prediction_keys, targets),
        "weighted_spearman": spearman_from_keys(prediction_keys, targets, active_weights),
        "top8_precision": top_k_precision(prediction_keys, targets, k=k),
        "decision_ndcg_at_8": ndcg_at_k(prediction_keys, targets, weights=active_weights, k=k),
        **tie_summary(prediction_keys),
        "top8_ids": [ids[index] for index in order[:k]] if ids is not None else None,
    }


MetricFunction = Callable[[Sequence[Any], Sequence[float], Sequence[float]], float]


def paired_bootstrap_delta(
    *,
    before_keys: Sequence[Any],
    after_keys: Sequence[Any],
    before_targets: Sequence[float],
    after_targets: Sequence[float],
    before_weights: Sequence[float],
    after_weights: Sequence[float],
    strata: Sequence[str],
    metric: MetricFunction,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    """Deterministic paired, stratum-preserving bootstrap confidence interval."""

    length = len(before_targets)
    if not (
        len(before_keys)
        == len(after_keys)
        == len(after_targets)
        == len(before_weights)
        == len(after_weights)
        == len(strata)
        == length
    ):
        raise ValueError("bootstrap input length mismatch")
    if not isinstance(replicates, int) or replicates < 100:
        raise ValueError("bootstrap replicates must be an integer >= 100")
    groups: dict[str, list[int]] = defaultdict(list)
    for index, stratum in enumerate(strata):
        groups[str(stratum)].append(index)
    rng = random.Random(seed)
    deltas: list[float] = []
    for _ in range(replicates):
        sample: list[int] = []
        for name in sorted(groups):
            indices = groups[name]
            sample.extend(rng.choice(indices) for _ in indices)
        sampled_before_targets = [before_targets[index] for index in sample]
        sampled_after_targets = [after_targets[index] for index in sample]
        sampled_before_weights = [before_weights[index] for index in sample]
        sampled_after_weights = [after_weights[index] for index in sample]
        before = metric(
            [before_keys[index] for index in sample],
            sampled_before_targets,
            sampled_before_weights,
        )
        after = metric(
            [after_keys[index] for index in sample],
            sampled_after_targets,
            sampled_after_weights,
        )
        if math.isfinite(before) and math.isfinite(after):
            deltas.append(after - before)
    if not deltas:
        raise ValueError("bootstrap produced no finite deltas")
    deltas.sort()

    def quantile(probability: float) -> float:
        position = probability * (len(deltas) - 1)
        lower = math.floor(position)
        upper = math.ceil(position)
        fraction = position - lower
        return deltas[lower] * (1.0 - fraction) + deltas[upper] * fraction

    return {
        "delta": metric(after_keys, after_targets, after_weights) - metric(before_keys, before_targets, before_weights),
        "ci95": [quantile(0.025), quantile(0.975)],
        "replicates_requested": replicates,
        "replicates_finite": len(deltas),
        "seed": seed,
        "resampling": "paired_stratum_preserving",
    }


@dataclass(frozen=True)
class RealizedDeltaRow:
    id: str
    factors: Mapping[str, float]
    factor_context: Mapping[str, Any]
    realized_benefit_s: float
    apparatus_valid: bool
    corpus: str
    byte_delta: int
    source_receipt: str
    source_receipt_sha256: str
    producer: str

    def validated(self) -> RealizedDeltaRow:
        if not self.id.strip() or not self.corpus.strip() or not self.producer.strip():
            raise ValueError("id, corpus, and producer must be nonempty")
        source_ref = Path(self.source_receipt)
        if not self.source_receipt.strip() or source_ref.is_absolute() or ".." in source_ref.parts:
            raise ValueError("source_receipt must be a durable repo-relative path")
        if len(self.source_receipt_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in self.source_receipt_sha256
        ):
            raise ValueError("source_receipt_sha256 must be lowercase SHA-256")
        required = {"exact_gap", "visibility", "realizability", "byte_price"}
        if not required <= set(self.factors):
            raise ValueError("factors lack the canonical v2 product fields")
        for key, value in self.factors.items():
            _finite(f"factor {key}", value)
        _finite("realized_benefit_s", self.realized_benefit_s)
        if not isinstance(self.apparatus_valid, bool):
            raise ValueError("apparatus_valid must be bool")
        if not isinstance(self.byte_delta, int) or isinstance(self.byte_delta, bool):
            raise ValueError("byte_delta must be int")
        _jsonable(self.factor_context)
        return self

    def to_record(self) -> dict[str, Any]:
        self.validated()
        return {
            "schema": CORPUS_SCHEMA,
            **asdict(self),
            "law_refs": [COMPOSITION_EQUATION_ID, POOL_EQUATION_ID, EMA_EQUATION_ID],
            "score_claim": False,
            "promotion_eligible": False,
            "actuation": "NONE",
        }


def append_realized_delta_row(
    row: RealizedDeltaRow,
    path: str | Path = DEFAULT_CORPUS_PATH,
    *,
    source_root: str | Path = REPO_ROOT,
) -> dict[str, Any]:
    """Append one typed row under an exclusive lock with conflict-safe dedup."""

    record = row.to_record()
    source_root_path = Path(source_root).resolve()
    source_path = (source_root_path / row.source_receipt).resolve()
    try:
        source_path.relative_to(source_root_path)
    except ValueError as exc:
        raise ValueError("source receipt escapes source_root") from exc
    if not source_path.is_file():
        raise ValueError(f"source receipt is absent: {row.source_receipt}")
    if sha256_file(source_path) != row.source_receipt_sha256:
        raise ValueError(f"source receipt SHA-256 drifted: {row.source_receipt}")
    payload = json.dumps(record, allow_nan=False, sort_keys=True, separators=(",", ":"))
    record_sha = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    corpus_path = Path(path)
    corpus_path.parent.mkdir(parents=True, exist_ok=True)
    with corpus_path.open("a+", encoding="utf-8") as handle:
        try:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        except ImportError:  # pragma: no cover - non-POSIX fallback
            fcntl = None  # type: ignore[assignment]
        try:
            handle.seek(0)
            for line in handle:
                try:
                    existing = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError("canonical costate corpus contains malformed JSONL") from exc
                if existing.get("id") != row.id:
                    continue
                existing_payload = json.dumps(existing, allow_nan=False, sort_keys=True, separators=(",", ":"))
                existing_sha = hashlib.sha256(existing_payload.encode("utf-8")).hexdigest()
                if existing_sha == record_sha:
                    return {"status": "EXACT_DUPLICATE_NOOP", "id": row.id, "row_sha256": record_sha}
                raise ValueError(f"conflicting append-only costate row id: {row.id}")
            handle.seek(0, os.SEEK_END)
            handle.write(payload + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        finally:
            if fcntl is not None:  # type: ignore[possibly-undefined]
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    return {"status": "APPENDED", "id": row.id, "row_sha256": record_sha}


def load_realized_delta_corpus(path: str | Path = DEFAULT_CORPUS_PATH) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    corpus_path = Path(path)
    if not corpus_path.exists():
        return rows
    for line in corpus_path.read_text().splitlines():
        row = json.loads(line)
        if row.get("schema") != CORPUS_SCHEMA:
            raise ValueError("canonical costate corpus schema drifted")
        identifier = str(row.get("id", ""))
        if not identifier or identifier in seen:
            raise ValueError("canonical costate corpus contains duplicate/empty id")
        required = {
            "factors",
            "factor_context",
            "realized_benefit_s",
            "apparatus_valid",
            "corpus",
            "byte_delta",
            "source_receipt",
            "source_receipt_sha256",
            "producer",
        }
        if not required <= set(row):
            raise ValueError("canonical costate corpus row is incomplete")
        RealizedDeltaRow(
            id=identifier,
            factors=row["factors"],
            factor_context=row["factor_context"],
            realized_benefit_s=row["realized_benefit_s"],
            apparatus_valid=row["apparatus_valid"],
            corpus=row["corpus"],
            byte_delta=row["byte_delta"],
            source_receipt=row["source_receipt"],
            source_receipt_sha256=row["source_receipt_sha256"],
            producer=row["producer"],
        ).validated()
        seen.add(identifier)
        rows.append(row)
    return rows


def row_from_backtest(row: Mapping[str, Any], *, source_receipt: str, source_receipt_sha256: str) -> RealizedDeltaRow:
    context = dict(row.get("factor_context", {}))
    return RealizedDeltaRow(
        id=str(row["id"]),
        factors={key: float(value) for key, value in dict(row["factors"]).items()},
        factor_context=context,
        realized_benefit_s=float(row["realized_benefit_s"]),
        apparatus_valid=bool(row["apparatus_valid"]),
        corpus=str(row["corpus"]),
        byte_delta=int(context.get("charged_bytes", 0)),
        source_receipt=source_receipt,
        source_receipt_sha256=source_receipt_sha256,
        producer="tools/costate_organ_v3_backtest.py:seed_v2_snapshot",
    )


def emit_m1_byte_close_row(
    receipt: Mapping[str, Any],
    *,
    source_receipt: str,
    source_receipt_sha256: str,
    corpus_path: str | Path = DEFAULT_CORPUS_PATH,
) -> dict[str, Any]:
    """Append an M1 row only when its receipt carries fully realized custody."""

    candidate = receipt.get("costate_realized_delta")
    if not isinstance(candidate, Mapping):
        return {
            "status": "NOT_EMITTED_REALIZED_ROW_ABSENT",
            "blocker": "M1 receipt has no costate_realized_delta block",
        }
    required = {
        "id",
        "factors",
        "factor_context",
        "realized_benefit_s",
        "apparatus_valid",
        "corpus",
        "byte_delta",
    }
    if not required <= set(candidate):
        raise ValueError("M1 costate_realized_delta block is incomplete")
    if not isinstance(candidate["factors"], Mapping) or not isinstance(candidate["factor_context"], Mapping):
        raise ValueError("M1 costate factors and context must be mappings")
    if not isinstance(candidate["apparatus_valid"], bool):
        raise ValueError("M1 apparatus_valid must be bool")
    if not isinstance(candidate["byte_delta"], int) or isinstance(candidate["byte_delta"], bool):
        raise ValueError("M1 byte_delta must be int")
    if candidate["byte_delta"] == 0:
        raise ValueError("M1 proof row must be the first nonzero-byte corpus row")
    row = RealizedDeltaRow(
        id=str(candidate["id"]),
        factors=dict(candidate["factors"]),
        factor_context=dict(candidate["factor_context"]),
        realized_benefit_s=candidate["realized_benefit_s"],
        apparatus_valid=candidate["apparatus_valid"],
        corpus=str(candidate["corpus"]),
        byte_delta=candidate["byte_delta"],
        source_receipt=source_receipt,
        source_receipt_sha256=source_receipt_sha256,
        producer="tools/produce_m1_band_manifest.py",
    )
    return append_realized_delta_row(row, corpus_path)


__all__ = [
    "COMPOSITION_EQUATION_ID",
    "CORPUS_SCHEMA",
    "DEFAULT_CORPUS_PATH",
    "EMA_DECAY_MEASURED",
    "EMA_EQUATION_ID",
    "POOL_EQUATION_ID",
    "R1B7_RECEIPT_SHA256",
    "RealizedDeltaRow",
    "StageSurvivalDistribution",
    "append_realized_delta_row",
    "denoise_realized_target",
    "emit_m1_byte_close_row",
    "graded_realizability",
    "load_r1b7_survival",
    "load_realized_delta_corpus",
    "ndcg_at_k",
    "opportunity_claims",
    "paired_bootstrap_delta",
    "pool_kkt_marginals",
    "rank_metrics",
    "row_from_backtest",
    "sha256_file",
    "sharpen_realizability_row",
    "spearman_from_keys",
    "stage_survival_distribution",
    "top_k_precision",
]
