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
import math
import statistics
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from tac.optimization.proxy_candidate_contract import (
    PROXY_FALSE_AUTHORITY_FIELDS,
    apply_proxy_evidence_boundary,
)

TRUST_REGION_DQS1_FEC6 = "dqs1_fec6_like_same_archive_segnet_rounding"
EUREKA_SIGNAL_SCHEMA = "local_cpu_contest_drift_eureka_signal.v1"
CALIBRATION_SCHEMA = "local_cpu_contest_drift_calibration.v1"
TRUST_REGION_POLICY_DQS1_FEC6 = "dqs1_fec6_same_archive_segnet_only_v1"
DQS1_FEC6_MAX_SCORE_DELTA = 5.0e-5
DQS1_FEC6_MAX_SEGNET_DELTA = 5.0e-7
DQS1_FEC6_MAX_ABS_POSENET_DELTA = 5.0e-12
DQS1_FEC6_MAX_ABS_RATE_DELTA = 5.0e-12
EMPTY_CALIBRATION_GUARD_BAND = 1.0
EUREKA_EXTRA_FALSE_AUTHORITY_FIELDS = (
    "score_claim_valid",
    "gpu_launched",
)
EUREKA_FALSE_AUTHORITY_FIELDS = tuple(
    dict.fromkeys(
        (
            *PROXY_FALSE_AUTHORITY_FIELDS.keys(),
            *EUREKA_EXTRA_FALSE_AUTHORITY_FIELDS,
        )
    )
)

LOCAL_CPU_ADVISORY_AXIS_LABELS = frozenset(
    {
        "cpu-advisory",
        "macos-cpu advisory",
        "macos-cpu-advisory",
        "macos cpu advisory",
    }
)


class LocalCPUContestDriftError(ValueError):
    """Raised when drift calibration inputs are malformed."""


@dataclass(frozen=True)
class TrustRegionAssessment:
    trust_region: str
    policy: str
    in_trust_region: bool
    blockers: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "trust_region": self.trust_region,
            "policy": self.policy,
            "in_trust_region": self.in_trust_region,
            "blockers": list(self.blockers),
            "warnings": list(self.warnings),
        }


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
    rejected_anchor_count: int = 0
    rejected_anchors: tuple[dict[str, Any], ...] = ()
    trust_region_policy: str = TRUST_REGION_POLICY_DQS1_FEC6

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
                "rejected_anchor_count": self.rejected_anchor_count,
                "rejected_anchors": list(self.rejected_anchors),
                "trust_region_policy": self.trust_region_policy,
                "authority": "false_authority_proxy_calibration_only",
            },
            dispatch_blockers=(
                "local_cpu_contest_drift_calibration_is_not_score_authority",
                "requires_same_archive_contest_cpu_anchor_before_score_claim",
            ),
        )


def assess_anchor_trust_region(
    anchor: PairedDriftAnchor,
    *,
    trust_region: str = TRUST_REGION_DQS1_FEC6,
) -> TrustRegionAssessment:
    """Return a fail-closed trust-region assessment for a paired anchor."""

    if trust_region != TRUST_REGION_DQS1_FEC6:
        return TrustRegionAssessment(
            trust_region=trust_region,
            policy="unknown_trust_region_policy",
            in_trust_region=False,
            blockers=("unknown_trust_region_requires_dedicated_calibration_policy",),
        )

    blockers: list[str] = []
    warnings: list[str] = []
    if anchor.trust_region != trust_region:
        blockers.append("anchor_trust_region_label_mismatch")
    if not anchor.archive_sha256:
        blockers.append("missing_archive_sha256")
    if not math.isfinite(anchor.local_score) or not math.isfinite(anchor.contest_score):
        blockers.append("nonfinite_score")
    delta = anchor.score_delta_local_minus_contest
    if delta < 0.0 or delta > DQS1_FEC6_MAX_SCORE_DELTA:
        blockers.append("score_delta_outside_dqs1_fec6_segnet_rounding_band")

    segnet_delta = anchor.segnet_delta_local_minus_contest
    if segnet_delta is None:
        blockers.append("missing_segnet_delta_for_dqs1_fec6_trust_region")
    elif segnet_delta < 0.0 or segnet_delta > DQS1_FEC6_MAX_SEGNET_DELTA:
        blockers.append("segnet_delta_outside_dqs1_fec6_rounding_band")

    posenet_delta = anchor.posenet_delta_local_minus_contest
    if posenet_delta is None:
        warnings.append("missing_posenet_delta")
    elif abs(posenet_delta) > DQS1_FEC6_MAX_ABS_POSENET_DELTA:
        blockers.append("posenet_delta_not_zero_for_segnet_rounding_region")

    rate_delta = anchor.rate_delta_local_minus_contest
    if rate_delta is None:
        warnings.append("missing_rate_delta")
    elif abs(rate_delta) > DQS1_FEC6_MAX_ABS_RATE_DELTA:
        blockers.append("rate_delta_not_zero_for_same_archive_region")

    path_markers = f"{anchor.local_path} {anchor.contest_path}".lower()
    if "mps" in path_markers:
        blockers.append("mps_path_marker_out_of_class")

    return TrustRegionAssessment(
        trust_region=trust_region,
        policy=TRUST_REGION_POLICY_DQS1_FEC6,
        in_trust_region=not blockers,
        blockers=tuple(blockers),
        warnings=tuple(warnings),
    )


def fit_drift_calibration(
    anchors: Iterable[PairedDriftAnchor],
    *,
    trust_region: str = TRUST_REGION_DQS1_FEC6,
    stability_threshold: float = 3.0e-6,
    min_guard_band: float = 3.0e-6,
) -> DriftCalibration:
    selected: list[PairedDriftAnchor] = []
    rejected: list[dict[str, Any]] = []
    for anchor in anchors:
        assessment = assess_anchor_trust_region(anchor, trust_region=trust_region)
        if assessment.in_trust_region:
            selected.append(anchor)
        else:
            rejected.append(
                {
                    "anchor": anchor.to_dict(),
                    "assessment": assessment.to_dict(),
                }
            )
    selected_tuple = tuple(selected)
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
            guard_band=EMPTY_CALIBRATION_GUARD_BAND,
            confidence="empty",
            anchors=(),
            rejected_anchor_count=len(rejected),
            rejected_anchors=tuple(rejected),
        )
    deltas = [anchor.score_delta_local_minus_contest for anchor in selected_tuple]
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
        anchor_count=len(selected_tuple),
        bias_local_minus_contest=median_delta,
        min_delta=min_delta,
        max_delta=max_delta,
        mean_delta=mean_delta,
        median_delta=median_delta,
        pstdev_delta=pstdev_delta,
        guard_band=guard_band,
        confidence=confidence,
        anchors=selected_tuple,
        rejected_anchor_count=len(rejected),
        rejected_anchors=tuple(rejected),
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
    candidate_archive_sha256: str = "",
    candidate_trust_region: str | None = None,
    candidate_trust_region_blockers: Iterable[str] = (),
) -> dict[str, Any]:
    projected = calibration.projected_contest_score(local_score)
    conservative = calibration.conservative_projected_contest_score(local_score)
    margin = float(auth_frontier_score) - conservative
    candidate_region = candidate_trust_region or calibration.trust_region
    trust_region_blockers = list(candidate_trust_region_blockers)
    if candidate_region != calibration.trust_region:
        trust_region_blockers.append("candidate_trust_region_mismatch")
    trigger = (
        calibration.confidence == "stable_core"
        and calibration.anchor_count >= 2
        and margin > float(min_margin)
        and not trust_region_blockers
    )
    urgency = "dispatch_exact_auth_anchor" if trigger else "observe_only"
    return apply_proxy_evidence_boundary(
        {
            "schema": EUREKA_SIGNAL_SCHEMA,
            "candidate_id": candidate_id,
            "candidate_archive_sha256": candidate_archive_sha256,
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
            "candidate_trust_region": candidate_region,
            "candidate_trust_region_matches_calibration": candidate_region
            == calibration.trust_region,
            "candidate_trust_region_blockers": trust_region_blockers,
            "eureka_trigger": trigger,
            "eureka_margin": margin,
            "recommended_action": urgency,
            "source_artifact": source_artifact,
            "score_claim_valid": False,
            "gpu_launched": False,
            "authority": "false_authority_exact_eval_spend_trigger_only",
        },
        dispatch_blockers=(
            "eureka_signal_is_not_score_authority",
            "exact_contest_cpu_eval_required_before_frontier_claim",
            *trust_region_blockers,
        ),
    )


def eureka_false_authority_violations(payload: Mapping[str, Any]) -> list[str]:
    """Return eureka authority fields that are missing or not exactly false."""

    violations: list[str] = []
    for field in EUREKA_FALSE_AUTHORITY_FIELDS:
        if payload.get(field) is not False:
            violations.append(field)
    return violations


def require_eureka_false_authority(
    payload: Mapping[str, Any],
    *,
    context: str = "eureka signal",
) -> None:
    """Fail closed unless every eureka authority field is explicitly false."""

    violations = eureka_false_authority_violations(payload)
    if violations:
        raise LocalCPUContestDriftError(
            f"{context} must set authority field(s) exactly false: {', '.join(violations)}"
        )


def _anchor_from_mapping(payload: Mapping[str, Any]) -> PairedDriftAnchor:
    try:
        return PairedDriftAnchor(
            archive_sha256=str(payload["archive_sha256"]),
            local_score=float(payload["local_score"]),
            contest_score=float(payload["contest_score"]),
            local_path=str(payload.get("local_path", "")),
            contest_path=str(payload.get("contest_path", "")),
            local_segnet_dist=_optional_float(payload, "local_segnet_dist"),
            contest_segnet_dist=_optional_float(payload, "contest_segnet_dist"),
            local_posenet_dist=_optional_float(payload, "local_posenet_dist"),
            contest_posenet_dist=_optional_float(payload, "contest_posenet_dist"),
            local_rate_unscaled=_optional_float(payload, "local_rate_unscaled"),
            contest_rate_unscaled=_optional_float(payload, "contest_rate_unscaled"),
            trust_region=str(payload.get("trust_region", TRUST_REGION_DQS1_FEC6)),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise LocalCPUContestDriftError(f"malformed calibration anchor: {exc}") from exc


def calibration_from_mapping(payload: Mapping[str, Any]) -> DriftCalibration:
    if payload.get("schema") != CALIBRATION_SCHEMA:
        raise LocalCPUContestDriftError(
            f"expected {CALIBRATION_SCHEMA}, got {payload.get('schema')!r}"
        )
    anchors_value = payload.get("anchors", [])
    if not isinstance(anchors_value, list):
        raise LocalCPUContestDriftError("calibration anchors must be a list")
    raw_anchor_payloads: list[Mapping[str, Any]] = []
    for item in anchors_value:
        if not isinstance(item, Mapping):
            raise LocalCPUContestDriftError("calibration anchors must contain objects")
        raw_anchor_payloads.append(item)
    rejected_value = payload.get("rejected_anchors", [])
    if rejected_value is not None and not isinstance(rejected_value, list):
        raise LocalCPUContestDriftError("calibration rejected_anchors must be a list")
    for item in rejected_value or []:
        if isinstance(item, Mapping) and isinstance(item.get("anchor"), Mapping):
            raw_anchor_payloads.append(item["anchor"])
    anchors: list[PairedDriftAnchor] = []
    seen_anchor_keys: set[tuple[str, str, str, float, float]] = set()
    for item in raw_anchor_payloads:
        anchor = _anchor_from_mapping(item)
        key = (
            anchor.archive_sha256,
            anchor.local_path,
            anchor.contest_path,
            anchor.local_score,
            anchor.contest_score,
        )
        if key in seen_anchor_keys:
            continue
        seen_anchor_keys.add(key)
        anchors.append(anchor)
    trust_region = str(payload.get("trust_region", TRUST_REGION_DQS1_FEC6))
    return fit_drift_calibration(anchors, trust_region=trust_region)


def load_calibration_json(path: str | Path) -> DriftCalibration:
    return calibration_from_mapping(_load_json(Path(path)))


def _payload_axis_label(payload: Mapping[str, Any]) -> str:
    for key in ("evidence_grade", "score_axis", "lane_tag"):
        value = payload.get(key)
        if isinstance(value, str) and value:
            return value
    return ""


def _normalize_axis_label(value: str) -> str:
    return value.strip().replace("_", "-").lower()


def build_eureka_signal_from_local_payload(
    *,
    candidate_id: str,
    local_payload: Mapping[str, Any],
    auth_frontier_score: float,
    calibration: DriftCalibration,
    candidate_trust_region: str | None = None,
    min_margin: float = 0.0,
    source_artifact: str = "",
) -> dict[str, Any]:
    """Build a false-authority eureka signal directly from local eval JSON."""

    local_axis = _payload_axis_label(local_payload)
    blockers: list[str] = []
    if not local_axis:
        blockers.append("candidate_local_axis_missing")
        local_axis = "unknown-local-axis"
    elif _normalize_axis_label(local_axis) not in LOCAL_CPU_ADVISORY_AXIS_LABELS:
        blockers.append("candidate_local_axis_not_macos_cpu_advisory")
    return build_eureka_signal(
        candidate_id=candidate_id,
        local_score=_score(local_payload),
        auth_frontier_score=auth_frontier_score,
        calibration=calibration,
        local_axis=local_axis,
        min_margin=min_margin,
        source_artifact=source_artifact,
        candidate_archive_sha256=_archive_sha256(local_payload),
        candidate_trust_region=candidate_trust_region or calibration.trust_region,
        candidate_trust_region_blockers=blockers,
    )


def build_eureka_signal_from_local_json_file(
    *,
    candidate_id: str,
    local_path: str | Path,
    auth_frontier_score: float,
    calibration: DriftCalibration,
    candidate_trust_region: str | None = None,
    min_margin: float = 0.0,
) -> dict[str, Any]:
    path = Path(local_path)
    return build_eureka_signal_from_local_payload(
        candidate_id=candidate_id,
        local_payload=_load_json(path),
        auth_frontier_score=auth_frontier_score,
        calibration=calibration,
        candidate_trust_region=candidate_trust_region,
        min_margin=min_margin,
        source_artifact=str(path),
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
    "EUREKA_FALSE_AUTHORITY_FIELDS",
    "EUREKA_SIGNAL_SCHEMA",
    "TRUST_REGION_DQS1_FEC6",
    "DriftCalibration",
    "LocalCPUContestDriftError",
    "PairedDriftAnchor",
    "TrustRegionAssessment",
    "assess_anchor_trust_region",
    "build_eureka_signal",
    "build_eureka_signal_from_local_json_file",
    "build_eureka_signal_from_local_payload",
    "calibration_from_mapping",
    "eureka_false_authority_violations",
    "fit_drift_calibration",
    "load_calibration_json",
    "paired_anchor_from_json_files",
    "require_eureka_false_authority",
]
