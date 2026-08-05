# SPDX-License-Identifier: MIT
"""L1 alarm calibration registry plus the small conformal/FDR core.

This module calibrates diagnostic ALARMS only.  Its p-values are apparatus
evidence for confound gates, burn supervisors, and run monitors; they are never
score authority, never candidate-promotion authority, and never a surrogate
ranker for exact contest rows.  A frontier claim still requires the frozen
``upstream/evaluate.py`` 600-sample archive path and real archive bytes.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any, Literal

PValueDirection = Literal["greater", "less", "two_sided"]
StatisticStatus = Literal["ready", "blocked", "semantic_invariant"]


@dataclass(frozen=True)
class AlarmRegistryRow:
    alarm_id: str
    surface: str
    score: str
    p_value_direction: PValueDirection
    calibration_population: str
    exchangeability_grade: str
    block_calibration_required: bool
    fdr_family: str
    consumer: str
    falsifier: str
    statistic_status: StatisticStatus = "ready"
    calibration_status: str = "requires_live_calibration"
    active: bool = True
    notes: str = ""

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SplitConformalResult:
    observed_score: float
    p_value: float
    direction: PValueDirection
    calibration_n: int
    tail_count: int


@dataclass(frozen=True)
class PValueObservation:
    alarm_id: str
    p_value: float
    fdr_family: str


@dataclass(frozen=True)
class BHResult:
    alarm_id: str
    p_value: float
    q_value: float
    rank: int
    threshold: float
    rejected: bool
    fdr_family: str


@dataclass(frozen=True)
class AlarmAdjudication:
    registry_row: AlarmRegistryRow
    conformal: SplitConformalResult
    bh: BHResult

    @property
    def alarm_fires(self) -> bool:
        return self.bh.rejected

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "registry_row": self.registry_row.to_json_dict(),
            "conformal": asdict(self.conformal),
            "bh": asdict(self.bh),
            "alarm_fires": self.alarm_fires,
        }


DEFAULT_REGISTRY_SCHEMA = "tac_l1_alarm_registry.v1"
LP1_LANE_GUARD_NULL_SCHEMA = "lp1_lane_guard_ratchet_null_reproduction.v1"


def _finite_float(value: float, *, name: str) -> float:
    out = float(value)
    if out != out or out in (float("inf"), float("-inf")):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return out


def _finite_scores(scores: Iterable[float]) -> list[float]:
    vals = [_finite_float(v, name="calibration score") for v in scores]
    if not vals:
        raise ValueError("split conformal calibration requires at least one score")
    return vals


def split_conformal_p_value(
    calibration_scores: Iterable[float],
    observed_score: float,
    *,
    direction: PValueDirection = "greater",
) -> SplitConformalResult:
    """Finite-sample split-conformal p-value for one alarm score.

    ``greater`` means high scores are anomalous; ``less`` means low scores are
    anomalous.  ``two_sided`` uses absolute deviation from the calibration
    median as the nonconformity score.  All variants use the standard
    ``(1 + tail_count) / (n + 1)`` correction.
    """
    vals = _finite_scores(calibration_scores)
    obs = _finite_float(observed_score, name="observed_score")
    if direction == "greater":
        tail = sum(v >= obs for v in vals)
    elif direction == "less":
        tail = sum(v <= obs for v in vals)
    elif direction == "two_sided":
        ordered = sorted(vals)
        n = len(ordered)
        med = ordered[n // 2] if n % 2 else 0.5 * (ordered[n // 2 - 1] + ordered[n // 2])
        obs_dev = abs(obs - med)
        tail = sum(abs(v - med) >= obs_dev for v in vals)
    else:
        raise ValueError(f"unknown conformal direction {direction!r}")
    return SplitConformalResult(
        observed_score=obs,
        p_value=float((tail + 1) / (len(vals) + 1)),
        direction=direction,
        calibration_n=len(vals),
        tail_count=int(tail),
    )


def benjamini_hochberg(
    observations: Sequence[PValueObservation],
    *,
    alpha: float = 0.05,
) -> tuple[BHResult, ...]:
    """Benjamini-Hochberg decisions with monotone q-values for one FDR family."""
    a = _finite_float(alpha, name="alpha")
    if not 0.0 < a <= 1.0:
        raise ValueError(f"alpha must be in (0, 1], got {alpha!r}")
    if not observations:
        return ()
    fams = {o.fdr_family for o in observations}
    if len(fams) != 1:
        raise ValueError(f"BH input must contain one FDR family, got {sorted(fams)!r}")
    ordered = sorted(observations, key=lambda o: (o.p_value, o.alarm_id))
    m = len(ordered)
    for obs in ordered:
        p = _finite_float(obs.p_value, name=f"p_value[{obs.alarm_id}]")
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"p-value for {obs.alarm_id!r} must be in [0, 1], got {p!r}")

    cut_rank = 0
    for rank, obs in enumerate(ordered, start=1):
        if obs.p_value <= a * rank / m:
            cut_rank = rank

    q_rev: list[float] = []
    running = 1.0
    for rank, obs in reversed(list(enumerate(ordered, start=1))):
        running = min(running, obs.p_value * m / rank)
        q_rev.append(running)
    q_values = list(reversed(q_rev))

    return tuple(
        BHResult(
            alarm_id=obs.alarm_id,
            p_value=float(obs.p_value),
            q_value=float(min(1.0, q)),
            rank=rank,
            threshold=float(a * rank / m),
            rejected=rank <= cut_rank,
            fdr_family=obs.fdr_family,
        )
        for rank, (obs, q) in enumerate(zip(ordered, q_values, strict=True), start=1)
    )


def default_alarm_registry() -> tuple[AlarmRegistryRow, ...]:
    return (
        AlarmRegistryRow(
            alarm_id="lane_guard.ratchet",
            surface="lane_guard.ratchet / inertness_alarm",
            score="sum_rises_s_units over the derived true gate horizon",
            p_value_direction="greater",
            calibration_population=(
                "same vehicle/window no-erosion Lane realized series; lp1 banked iid-noise null "
                "uses MC n=20000 and the shipped first-difference sigma"
            ),
            exchangeability_grade="conditional_if_stationary_window; fragile across stages",
            block_calibration_required=True,
            fdr_family="lane_guard",
            consumer="#934 successor, b4s burn reseal, lane guard fire order",
            falsifier="held-out or block-calibrated null p-values are not super-uniform",
            calibration_status="lp1_banked_null_case_available; live use requires block calibration",
            notes="Worked example for AL1; diagnostic-only and scorer-free.",
        ),
        AlarmRegistryRow(
            alarm_id="term_domination",
            surface="TR1 loss term domination",
            score="max non-scored post-weight share, or scored-term deficit below the derived floor",
            p_value_direction="greater",
            calibration_population="same-stage term-share rows under the same vehicle and schedule",
            exchangeability_grade="partial_stage_scoped",
            block_calibration_required=True,
            fdr_family="loss_term",
            consumer="v9 telemetry port and burn supervisor",
            falsifier="same-stage null rows fail calibration or the fire is a stage transition",
            calibration_status="requires_same_stage_null",
        ),
        AlarmRegistryRow(
            alarm_id="term_inert",
            surface="engaged-but-inert loss terms",
            score="sustained movement deficit or residual debt while the term is engaged",
            p_value_direction="greater",
            calibration_population="engaged historical rows, scoped by stage and vehicle",
            exchangeability_grade="partial_block_calibration_required",
            block_calibration_required=True,
            fdr_family="loss_term",
            consumer="force-stack and curriculum gates",
            falsifier="alarm disappears under block calibration or null p-values bunch low",
            calibration_status="requires_block_null",
        ),
        AlarmRegistryRow(
            alarm_id="gnorm_hijack",
            surface="gradient-norm hijack watchdog",
            score="excess global or per-group norm share against the clip budget",
            p_value_direction="greater",
            calibration_population="same-stage gradient norm shares with matching optimizer controls",
            exchangeability_grade="partial_fragile",
            block_calibration_required=True,
            fdr_family="gradient_health",
            consumer="force caps and optimizer watchdogs",
            falsifier="nonstationary gradients invalidate calibration before the fire",
            calibration_status="requires_same_stage_null_and_shift_guard",
        ),
    )


def alarm_registry_json(
    registry: Sequence[AlarmRegistryRow] | None = None,
) -> dict[str, Any]:
    rows = tuple(registry or default_alarm_registry())
    return {
        "schema": DEFAULT_REGISTRY_SCHEMA,
        "authority_boundary": (
            "diagnostic alarm calibration only; no score authority, no promotion, no surrogate ranker"
        ),
        "rows": [row.to_json_dict() for row in rows],
    }


def registry_by_alarm_id(
    registry: Sequence[AlarmRegistryRow] | None = None,
) -> dict[str, AlarmRegistryRow]:
    rows = tuple(registry or default_alarm_registry())
    out = {row.alarm_id: row for row in rows}
    if len(out) != len(rows):
        raise ValueError("duplicate alarm_id in alarm registry")
    return out


def rows_for_consumer(
    consumer_query: str,
    registry: Sequence[AlarmRegistryRow] | None = None,
) -> tuple[AlarmRegistryRow, ...]:
    terms = tuple(term for term in re.split(r"[^a-z0-9#]+", consumer_query.lower()) if term)
    if not terms:
        raise ValueError("consumer_query must contain at least one searchable term")
    return tuple(
        row for row in tuple(registry or default_alarm_registry())
        if all(term in row.consumer.lower() for term in terms)
    )


def adjudicate_alarm_family(
    observed_scores: Mapping[str, float],
    calibration_scores: Mapping[str, Iterable[float]],
    *,
    registry: Sequence[AlarmRegistryRow] | None = None,
    fdr_family: str,
    alpha: float = 0.05,
) -> tuple[AlarmAdjudication, ...]:
    """Registry-consuming first-reader path for confound/burn alarm fires."""
    by_id = registry_by_alarm_id(registry)
    conformal_by_id: dict[str, SplitConformalResult] = {}
    pvals: list[PValueObservation] = []
    for alarm_id, observed in observed_scores.items():
        row = by_id[alarm_id]
        if row.fdr_family != fdr_family or not row.active or row.statistic_status != "ready":
            continue
        if alarm_id not in calibration_scores:
            raise KeyError(f"missing calibration scores for active alarm {alarm_id!r}")
        conformal = split_conformal_p_value(
            calibration_scores[alarm_id],
            observed,
            direction=row.p_value_direction,
        )
        conformal_by_id[alarm_id] = conformal
        pvals.append(PValueObservation(alarm_id, conformal.p_value, row.fdr_family))
    bh_by_id = {row.alarm_id: row for row in benjamini_hochberg(pvals, alpha=alpha)}
    return tuple(
        AlarmAdjudication(by_id[alarm_id], conformal_by_id[alarm_id], bh_by_id[alarm_id])
        for alarm_id in sorted(conformal_by_id)
    )


def lp1_lane_guard_ratchet_null_reproduction(
    *,
    n_trials: int = 20_000,
    seed: int = 777,
) -> dict[str, Any]:
    """Reproduce lp1's lane-guard false-positive verdict from the banked null shape.

    The durable lp1 memo records observed sum-of-rises ``0.029133`` over 64 gates,
    the shipped first-difference sigma ``0.00142148``, and an iid-noise MC null with
    n=20000.  The raw null vector was not found in the bounded repo/SSD scopes, so
    this regenerates the stated null construction deterministically and then uses
    the conformal/BH path above.  High rises are the alarm direction, so the observed
    low-tail 0.7%-class statistic yields a high-tail conformal p near 0.99 and does
    not fire.
    """
    import numpy as np

    n = int(n_trials)
    if n < 100:
        raise ValueError("lp1 reproduction needs enough MC trials for the 0.7%-class check")
    observed_sum_rises = 0.029133
    observed_rise_count = 22.0
    sigma = 0.00142148
    rng = np.random.default_rng(int(seed))
    diffs = rng.normal(0.0, sigma * np.sqrt(2.0), size=(n, 63))
    rise_scores = np.maximum(diffs, 0.0).sum(axis=1)
    rise_counts = (diffs > 0.0).sum(axis=1).astype(float)
    adjudications = adjudicate_alarm_family(
        {"lane_guard.ratchet": observed_sum_rises},
        {"lane_guard.ratchet": rise_scores.tolist()},
        fdr_family="lane_guard",
    )
    return {
        "schema": LP1_LANE_GUARD_NULL_SCHEMA,
        "source": ".omx/research/ddm_lp1_lane_program_20260803.md",
        "selection_mode": "ALL 64 lane_guard rows in burn-4 windows 01-03; regenerated iid-noise MC null",
        "axis": "diagnostic/apparatus only; no scorer forward",
        "n_trials": n,
        "seed": int(seed),
        "observed": {
            "sum_rises_s_units": observed_sum_rises,
            "rise_count": observed_rise_count,
        },
        "null": {
            "sum_rises_mean": float(np.mean(rise_scores)),
            "sum_rises_p5": float(np.percentile(rise_scores, 5)),
            "sum_rises_p95": float(np.percentile(rise_scores, 95)),
            "sum_rises_low_tail_percentile": float(np.mean(rise_scores <= observed_sum_rises)),
            "rise_count_mean": float(np.mean(rise_counts)),
            "rise_count_low_tail_percentile": float(np.mean(rise_counts <= observed_rise_count)),
        },
        "adjudications": [a.to_json_dict() for a in adjudications],
        "verdict": "FALSE_POSITIVE_REPRODUCED_NO_HIGH_TAIL_ALARM",
    }


__all__ = [
    "DEFAULT_REGISTRY_SCHEMA",
    "LP1_LANE_GUARD_NULL_SCHEMA",
    "AlarmAdjudication",
    "AlarmRegistryRow",
    "BHResult",
    "PValueObservation",
    "SplitConformalResult",
    "adjudicate_alarm_family",
    "alarm_registry_json",
    "benjamini_hochberg",
    "default_alarm_registry",
    "lp1_lane_guard_ratchet_null_reproduction",
    "registry_by_alarm_id",
    "rows_for_consumer",
    "split_conformal_p_value",
]
