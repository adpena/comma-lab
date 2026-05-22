# SPDX-License-Identifier: MIT
"""Local macOS-CPU to contest-CPU drift calibration.

This module is deliberately narrow: it calibrates local macOS CPU advisory
rows against same-archive Linux contest-CPU anchors. It never produces score
authority. The intended use is spend triage and "eureka" detection when local
evidence is so strong that a widened contest projection still beats the current
auth frontier.
"""

from __future__ import annotations

import json
import statistics
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from tac.optimization.proxy_candidate_contract import apply_proxy_evidence_boundary

TRUST_REGION_DQS1_FEC6 = "dqs1_fec6_like_same_archive_segnet_rounding"
EUREKA_SIGNAL_SCHEMA = "local_cpu_contest_drift_eureka_signal.v1"
CALIBRATION_SCHEMA = "local_cpu_contest_drift_calibration.v1"


class LocalCPUContestDriftError(ValueError):
    """Raised when drift calibration inputs are malformed."""


@dataclass(frozen=True)
class PairedDriftAnchor:
    archive_sha256: str
    local_score: float
    contest_score: float
    local_path: str
    contest_path: str
    local_segnet_dist: float | None = None
    contest_segnet_dist: float | None = None
    local_posenet_dist: float | None = None
    contest_posenet_dist: float | None = None
    local_rate_unscaled: float | None = None
    contest_rate_unscaled: float | None = None
    trust_region: str = TRUST_REGION_DQS1_FEC6

    @property
    def score_delta_local_minus_contest(self) -> float:
        return self.local_score - self.contest_score

    @property
    def segnet_delta_local_minus_contest(self) -> float | None:
        if self.local_segnet_dist is None or self.contest_segnet_dist is None:
            return None
        return self.local_segnet_dist - self.contest_segnet_dist

    @property
    def posenet_delta_local_minus_contest(self) -> float | None:
        if self.local_posenet_dist is None or self.contest_posenet_dist is None:
            return None
        return self.local_posenet_dist - self.contest_posenet_dist

    @property
    def rate_delta_local_minus_contest(self) -> float | None:
        if self.local_rate_unscaled is None or self.contest_rate_unscaled is None:
            return None
        return self.local_rate_unscaled - self.contest_rate_unscaled

    def to_dict(self) -> dict[str, Any]:
        return {
            "archive_sha256": self.archive_sha256,
            "local_score": self.local_score,
            "contest_score": self.contest_score,
            "score_delta_local_minus_contest": self.score_delta_local_minus_contest,
            "local_path": self.local_path,
            "contest_path": self.contest_path,
            "local_segnet_dist": self.local_segnet_dist,
            "contest_segnet_dist": self.contest_segnet_dist,
            "segnet_delta_local_minus_contest": self.segnet_delta_local_minus_contest,
            "local_posenet_dist": self.local_posenet_dist,
            "contest_posenet_dist": self.contest_posenet_dist,
            "posenet_delta_local_minus_contest": self.posenet_delta_local_minus_contest,
            "local_rate_unscaled": self.local_rate_unscaled,
            "contest_rate_unscaled": self.contest_rate_unscaled,
            "rate_delta_local_minus_contest": self.rate_delta_local_minus_contest,
            "trust_region": self.trust_region,
        }


@dataclass(frozen=True)
class DriftCalibration:
    trust_region: str
    anchor_count: int
    bias_local_minus_contest: float
    min_delta: float
    max_delta: float
    mean_delta: float
    median_delta: float
    pstdev_delta: float
    guard_band: float
    confidence: Literal["empty", "single_anchor", "stable_core", "wide_or_mixed"]
    anchors: tuple[PairedDriftAnchor, ...]

    def projected_contest_score(self, local_score: float) -> float:
        """Return the point projection. Advisory only."""
        return float(local_score) - self.bias_local_minus_contest

    def conservative_projected_contest_score(self, local_score: float) -> float:
        """Return a worst-case-low-confidence projection. Advisory only.

        Lower score is better, so the conservative projection subtracts the
        fitted bias and then adds the guard band.
        """
        return self.projected_contest_score(local_score) + self.guard_band

    def to_dict(self) -> dict[str, Any]:
        return apply_proxy_evidence_boundary(
            {
                "schema": CALIBRATION_SCHEMA,
                "trust_region": self.trust_region,
                "anchor_count": self.anchor_count,
                "bias_local_minus_contest": self.bias_local_minus_contest,
                "min_delta": self.min_delta,
                "max_delta": self.max_delta,
                "mean_delta": self.mean_delta,
                "median_delta": self.median_delta,
                "pstdev_delta": self.pstdev_delta,
                "guard_band": self.guard_band,
                "confidence": self.confidence,
                "anchors": [anchor.to_dict() for anchor in self.anchors],
                "authority": "false_authority_proxy_calibration_only",
            },
            dispatch_blockers=(
                "local_cpu_contest_drift_calibration_is_not_score_authority",
                "requires_same_archive_contest_cpu_anchor_before_score_claim",
            ),
        )


def fit_drift_calibration(
    anchors: Iterable[PairedDriftAnchor],
    *,
    trust_region: str = TRUST_REGION_DQS1_FEC6,
    stability_threshold: float = 3.0e-6,
    min_guard_band: float = 3.0e-6,
) -> DriftCalibration:
    selected = tuple(anchor for anchor in anchors if anchor.trust_region == trust_region)
    if not selected:
        return DriftCalibration(
            trust_region=trust_region,
            anchor_count=0,
            bias_local_minus_contest=0.0,
            min_delta=0.0,
            max_delta=0.0,
            mean_delta=0.0,
            median_delta=0.0,
            pstdev_delta=0.0,
            guard_band=float("inf"),
            confidence="empty",
            anchors=(),
        )
    deltas = [anchor.score_delta_local_minus_contest for anchor in selected]
    min_delta = min(deltas)
    max_delta = max(deltas)
    mean_delta = statistics.mean(deltas)
    median_delta = statistics.median(deltas)
    pstdev_delta = statistics.pstdev(deltas) if len(deltas) > 1 else 0.0
    span = max_delta - min_delta
    confidence: Literal["single_anchor", "stable_core", "wide_or_mixed"]
    if len(deltas) == 1:
        confidence = "single_anchor"
    elif span <= stability_threshold:
        confidence = "stable_core"
    else:
        confidence = "wide_or_mixed"
    guard_band = max(min_guard_band, span, 3.0 * pstdev_delta)
    return DriftCalibration(
        trust_region=trust_region,
        anchor_count=len(selected),
        bias_local_minus_contest=median_delta,
        min_delta=min_delta,
        max_delta=max_delta,
        mean_delta=mean_delta,
        median_delta=median_delta,
        pstdev_delta=pstdev_delta,
        guard_band=guard_band,
        confidence=confidence,
        anchors=selected,
    )


def build_eureka_signal(
    *,
    candidate_id: str,
    local_score: float,
    auth_frontier_score: float,
    calibration: DriftCalibration,
    local_axis: str = "macOS-CPU advisory",
    target_axis: str = "contest-CPU",
    min_margin: float = 0.0,
    source_artifact: str = "",
) -> dict[str, Any]:
    projected = calibration.projected_contest_score(local_score)
    conservative = calibration.conservative_projected_contest_score(local_score)
    margin = float(auth_frontier_score) - conservative
    trigger = (
        calibration.confidence == "stable_core"
        and calibration.anchor_count >= 2
        and margin > float(min_margin)
    )
    urgency = "dispatch_exact_auth_anchor" if trigger else "observe_only"
    return apply_proxy_evidence_boundary(
        {
            "schema": EUREKA_SIGNAL_SCHEMA,
            "candidate_id": candidate_id,
            "local_axis": local_axis,
            "target_axis": target_axis,
            "local_score": float(local_score),
            "auth_frontier_score": float(auth_frontier_score),
            "projected_contest_score": projected,
            "conservative_projected_contest_score": conservative,
            "bias_local_minus_contest": calibration.bias_local_minus_contest,
            "guard_band": calibration.guard_band,
            "calibration_confidence": calibration.confidence,
            "calibration_anchor_count": calibration.anchor_count,
            "trust_region": calibration.trust_region,
            "eureka_trigger": trigger,
            "eureka_margin": margin,
            "recommended_action": urgency,
            "source_artifact": source_artifact,
            "authority": "false_authority_exact_eval_spend_trigger_only",
        },
        dispatch_blockers=(
            "eureka_signal_is_not_score_authority",
            "exact_contest_cpu_eval_required_before_frontier_claim",
        ),
    )


def _load_json(path: Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise LocalCPUContestDriftError(f"{path}: could not load JSON: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise LocalCPUContestDriftError(f"{path}: expected JSON object")
    return payload


def _score(payload: Mapping[str, Any]) -> float:
    for key in ("score_recomputed_from_components", "canonical_score", "final_score"):
        value = payload.get(key)
        if isinstance(value, int | float):
            return float(value)
    raise LocalCPUContestDriftError("payload missing score field")


def _archive_sha256(payload: Mapping[str, Any]) -> str:
    provenance = payload.get("provenance")
    if isinstance(provenance, Mapping):
        value = provenance.get("archive_sha256")
        if isinstance(value, str) and value:
            return value
    value = payload.get("archive_sha256")
    if isinstance(value, str) and value:
        return value
    raise LocalCPUContestDriftError("payload missing archive sha256")


def _optional_float(payload: Mapping[str, Any], key: str) -> float | None:
    value = payload.get(key)
    if isinstance(value, int | float):
        return float(value)
    return None


def paired_anchor_from_json_files(
    *,
    local_path: str | Path,
    contest_path: str | Path,
    trust_region: str = TRUST_REGION_DQS1_FEC6,
) -> PairedDriftAnchor:
    local_file = Path(local_path)
    contest_file = Path(contest_path)
    local = _load_json(local_file)
    contest = _load_json(contest_file)
    local_archive = _archive_sha256(local)
    contest_archive = _archive_sha256(contest)
    if local_archive != contest_archive:
        raise LocalCPUContestDriftError(
            "local and contest payloads must reference the same archive sha256: "
            f"{local_archive} != {contest_archive}"
        )
    return PairedDriftAnchor(
        archive_sha256=local_archive,
        local_score=_score(local),
        contest_score=_score(contest),
        local_path=str(local_file),
        contest_path=str(contest_file),
        local_segnet_dist=_optional_float(local, "avg_segnet_dist"),
        contest_segnet_dist=_optional_float(contest, "avg_segnet_dist"),
        local_posenet_dist=_optional_float(local, "avg_posenet_dist"),
        contest_posenet_dist=_optional_float(contest, "avg_posenet_dist"),
        local_rate_unscaled=_optional_float(local, "rate_unscaled"),
        contest_rate_unscaled=_optional_float(contest, "rate_unscaled"),
        trust_region=trust_region,
    )


__all__ = [
    "CALIBRATION_SCHEMA",
    "EUREKA_SIGNAL_SCHEMA",
    "TRUST_REGION_DQS1_FEC6",
    "DriftCalibration",
    "LocalCPUContestDriftError",
    "PairedDriftAnchor",
    "build_eureka_signal",
    "fit_drift_calibration",
    "paired_anchor_from_json_files",
]
