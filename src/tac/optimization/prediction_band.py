# SPDX-License-Identifier: MIT
"""Custody model for planning-only predicted score bands.

Prediction bands are not score claims. They are ranking priors used by the
Cathedral/autopilot stack. This module makes the prior itself auditable: a band
that lacks axis, baseline, source, uncertainty, supersession, or empirical
anchor custody must stay visible as a planning annotation, but it must not look
like clean rank authority.
"""

from __future__ import annotations

import dataclasses
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from tac.optimization.research_basis import (
    RESEARCH_SOURCES,
    ResearchBasisError,
    canonical_research_basis_id,
)

SHA256_HEX_RE = re.compile(r"^[0-9a-fA-F]{64}$")

BandKind = Literal["delta_score", "absolute_score"]
SupersessionStatus = Literal["active", "superseded", "deprecated"]
AnchorStatus = Literal["absent", "pending", "diagnostic_only", "landed", "superseded"]


@dataclass(frozen=True)
class BaselineRef:
    label: str
    axis: str
    score: float | None = None
    archive_sha256: str = ""
    runtime_tree_sha256: str = ""
    artifact_path: str = ""


@dataclass(frozen=True)
class BandSource:
    local_ledger_paths: tuple[str, ...] = ()
    research_basis_ids: tuple[str, ...] = ()
    claim_scope: str = ""


@dataclass(frozen=True)
class UncertaintyRef:
    method: str = ""
    confidence_tag: str = ""
    n_empirical_anchors: int = 0
    notes: str = ""


@dataclass(frozen=True)
class SupersessionRef:
    status: SupersessionStatus = "active"
    superseded_by: str = ""
    reason: str = ""


@dataclass(frozen=True)
class EmpiricalAnchorRef:
    status: AnchorStatus = "absent"
    anchors: tuple[Mapping[str, Any], ...] = ()


@dataclass(frozen=True)
class PredictionBand:
    band_id: str
    subject_id: str
    band_kind: BandKind
    low: float
    high: float
    axis: str
    baseline: BaselineRef
    band_source: BandSource
    uncertainty: UncertaintyRef
    supersession: SupersessionRef
    empirical_anchor: EmpiricalAnchorRef
    planning_only: bool = True
    score_claim: bool = False


@dataclass(frozen=True)
class PredictionBandVerdict:
    valid_for_rank_reward: bool
    valid_for_dispatch_planning: bool
    valid_for_promotion: bool
    blockers: tuple[str, ...] = ()
    annotations: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid_for_rank_reward": self.valid_for_rank_reward,
            "valid_for_dispatch_planning": self.valid_for_dispatch_planning,
            "valid_for_promotion": self.valid_for_promotion,
            "blockers": list(self.blockers),
            "annotations": list(self.annotations),
        }


def is_sha256_hex(value: object) -> bool:
    return isinstance(value, str) and bool(SHA256_HEX_RE.fullmatch(value))


def _as_tuple_str(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,) if value.strip() else ()
    if isinstance(value, Sequence):
        out: list[str] = []
        for item in value:
            if isinstance(item, str) and item.strip():
                out.append(item)
        return tuple(out)
    return ()


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def prediction_band_from_mapping(payload: Mapping[str, Any]) -> PredictionBand:
    """Parse a JSON-style mapping into a typed prediction band."""
    baseline = _mapping(payload.get("baseline"))
    source = _mapping(payload.get("band_source"))
    uncertainty = _mapping(payload.get("uncertainty"))
    supersession = _mapping(payload.get("supersession"))
    empirical_anchor = _mapping(payload.get("empirical_anchor"))
    anchors = empirical_anchor.get("anchors", ())
    if not isinstance(anchors, Sequence) or isinstance(anchors, (str, bytes)):
        anchors = ()
    anchor_tuple: tuple[Mapping[str, Any], ...] = tuple(
        item for item in anchors if isinstance(item, Mapping)
    )
    score_obj = baseline.get("score")
    score = float(score_obj) if isinstance(score_obj, int | float) else None
    return PredictionBand(
        band_id=str(payload.get("band_id") or ""),
        subject_id=str(payload.get("subject_id") or ""),
        band_kind=str(payload.get("band_kind") or "delta_score"),  # type: ignore[arg-type]
        low=float(payload.get("low")),
        high=float(payload.get("high")),
        axis=str(payload.get("axis") or ""),
        baseline=BaselineRef(
            label=str(baseline.get("label") or ""),
            axis=str(baseline.get("axis") or ""),
            score=score,
            archive_sha256=str(baseline.get("archive_sha256") or ""),
            runtime_tree_sha256=str(baseline.get("runtime_tree_sha256") or ""),
            artifact_path=str(baseline.get("artifact_path") or ""),
        ),
        band_source=BandSource(
            local_ledger_paths=_as_tuple_str(source.get("local_ledger_paths")),
            research_basis_ids=_as_tuple_str(source.get("research_basis_ids")),
            claim_scope=str(source.get("claim_scope") or ""),
        ),
        uncertainty=UncertaintyRef(
            method=str(uncertainty.get("method") or ""),
            confidence_tag=str(uncertainty.get("confidence_tag") or ""),
            n_empirical_anchors=int(uncertainty.get("n_empirical_anchors") or 0),
            notes=str(uncertainty.get("notes") or ""),
        ),
        supersession=SupersessionRef(
            status=str(supersession.get("status") or "active"),  # type: ignore[arg-type]
            superseded_by=str(supersession.get("superseded_by") or ""),
            reason=str(supersession.get("reason") or ""),
        ),
        empirical_anchor=EmpiricalAnchorRef(
            status=str(empirical_anchor.get("status") or "absent"),  # type: ignore[arg-type]
            anchors=anchor_tuple,
        ),
        planning_only=bool(payload.get("planning_only", True)),
        score_claim=bool(payload.get("score_claim", False)),
    )


def missing_prediction_band_verdict(
    *,
    subject_id: str,
    low: float,
    high: float,
    axis: str,
) -> PredictionBandVerdict:
    """Return the fail-closed verdict for an un-custodied numeric band."""
    if low == 0.0 and high == 0.0:
        return PredictionBandVerdict(
            valid_for_rank_reward=True,
            valid_for_dispatch_planning=True,
            valid_for_promotion=False,
            annotations=("prediction_band_zero_delta_no_rank_reward",),
        )
    annotations = (
        f"subject_id={subject_id}",
        f"band=[{low:.6g}, {high:.6g}]",
        f"axis={axis}",
    )
    return PredictionBandVerdict(
        valid_for_rank_reward=False,
        valid_for_dispatch_planning=True,
        valid_for_promotion=False,
        blockers=("prediction_band_missing",),
        annotations=annotations,
    )


def validate_prediction_band(
    band: PredictionBand,
    *,
    expected_subject_id: str | None = None,
    expected_low: float | None = None,
    expected_high: float | None = None,
    known_research_basis_ids: Mapping[str, object] | None = None,
) -> PredictionBandVerdict:
    """Validate that a numeric band has enough custody to influence ranking."""
    known = known_research_basis_ids or RESEARCH_SOURCES
    blockers: list[str] = []
    annotations: list[str] = []

    if band.score_claim:
        blockers.append("prediction_band_score_claim_forbidden")
    if band.band_kind not in ("delta_score", "absolute_score"):
        blockers.append("prediction_band_kind_invalid")
    if not band.band_id.strip():
        blockers.append("prediction_band_id_missing")
    if not band.subject_id.strip():
        blockers.append("prediction_band_subject_missing")
    if expected_subject_id and band.subject_id != expected_subject_id:
        blockers.append("prediction_band_subject_mismatch")
    if not math.isfinite(band.low) or not math.isfinite(band.high):
        blockers.append("prediction_band_non_finite")
    elif band.low > band.high:
        blockers.append("prediction_band_low_gt_high")
    if expected_low is not None and math.isfinite(band.low) and band.low != expected_low:
        blockers.append("prediction_band_low_mismatch")
    if expected_high is not None and math.isfinite(band.high) and band.high != expected_high:
        blockers.append("prediction_band_high_mismatch")
    if not band.axis.strip():
        blockers.append("prediction_band_axis_missing")

    baseline = band.baseline
    if not baseline.label.strip() or not baseline.axis.strip() or baseline.score is None:
        blockers.append("prediction_band_baseline_missing")
    elif not math.isfinite(baseline.score):
        blockers.append("prediction_band_baseline_score_non_finite")
    if not is_sha256_hex(baseline.archive_sha256) or not is_sha256_hex(
        baseline.runtime_tree_sha256
    ):
        blockers.append("prediction_band_baseline_custody_missing")
    if not baseline.artifact_path.strip():
        blockers.append("prediction_band_baseline_artifact_missing")

    source = band.band_source
    if not source.local_ledger_paths or not source.claim_scope.strip():
        blockers.append("prediction_band_source_missing")
    if not source.research_basis_ids:
        blockers.append("prediction_band_research_basis_missing")
    else:
        unknown: list[str] = []
        for source_id in source.research_basis_ids:
            try:
                canonical_id = canonical_research_basis_id(source_id)
            except ResearchBasisError:
                unknown.append(source_id)
                continue
            if canonical_id not in known:
                unknown.append(source_id)
        if unknown:
            blockers.append("prediction_band_unknown_research_basis")
            annotations.append(f"unknown_research_basis_ids={unknown!r}")

    uncertainty = band.uncertainty
    if not uncertainty.method.strip() or not uncertainty.confidence_tag.strip():
        blockers.append("prediction_band_uncertainty_missing")
    if uncertainty.n_empirical_anchors < 0:
        blockers.append("prediction_band_uncertainty_anchor_count_invalid")

    supersession = band.supersession
    if supersession.status not in ("active", "superseded", "deprecated"):
        blockers.append("prediction_band_supersession_status_invalid")
    elif supersession.status != "active":
        blockers.append("prediction_band_superseded")

    empirical = band.empirical_anchor
    if empirical.status not in (
        "absent",
        "pending",
        "diagnostic_only",
        "landed",
        "superseded",
    ):
        blockers.append("prediction_band_empirical_anchor_status_invalid")
    elif empirical.status in ("absent", "pending"):
        blockers.append("prediction_band_empirical_anchor_missing")
    elif empirical.status == "diagnostic_only":
        blockers.append("prediction_band_proxy_anchor_not_promotable")
    elif empirical.status == "superseded":
        blockers.append("prediction_band_empirical_anchor_superseded")
    elif empirical.status == "landed" and not empirical.anchors:
        blockers.append("prediction_band_empirical_anchor_missing")

    valid_for_rank_reward = not blockers
    valid_for_dispatch_planning = "prediction_band_score_claim_forbidden" not in blockers
    valid_for_promotion = (
        valid_for_rank_reward
        and not band.planning_only
        and empirical.status == "landed"
    )
    return PredictionBandVerdict(
        valid_for_rank_reward=valid_for_rank_reward,
        valid_for_dispatch_planning=valid_for_dispatch_planning,
        valid_for_promotion=valid_for_promotion,
        blockers=tuple(dict.fromkeys(blockers)),
        annotations=tuple(annotations),
    )


def validate_optional_prediction_band(
    payload: Mapping[str, Any] | None,
    *,
    subject_id: str,
    low: float,
    high: float,
    axis: str,
) -> PredictionBandVerdict:
    """Validate an optional JSON-style band payload.

    Missing non-zero bands become explicit blockers. Missing zero-delta bands are
    allowed as no-rank-reward annotations.
    """
    if payload is None:
        return missing_prediction_band_verdict(
            subject_id=subject_id,
            low=low,
            high=high,
            axis=axis,
        )
    try:
        band = prediction_band_from_mapping(payload)
    except (TypeError, ValueError) as exc:
        return PredictionBandVerdict(
            valid_for_rank_reward=False,
            valid_for_dispatch_planning=True,
            valid_for_promotion=False,
            blockers=("prediction_band_parse_failed",),
            annotations=(f"parse_error={exc}",),
        )
    return validate_prediction_band(
        band,
        expected_subject_id=subject_id,
        expected_low=low,
        expected_high=high,
    )


def verdict_to_dict(verdict: PredictionBandVerdict) -> dict[str, Any]:
    return verdict.to_dict()


def prediction_band_to_dict(band: PredictionBand) -> dict[str, Any]:
    return dataclasses.asdict(band)


__all__ = [
    "AnchorStatus",
    "BandKind",
    "BandSource",
    "BaselineRef",
    "EmpiricalAnchorRef",
    "PredictionBand",
    "PredictionBandVerdict",
    "SupersessionRef",
    "SupersessionStatus",
    "UncertaintyRef",
    "is_sha256_hex",
    "missing_prediction_band_verdict",
    "prediction_band_from_mapping",
    "prediction_band_to_dict",
    "validate_optional_prediction_band",
    "validate_prediction_band",
    "verdict_to_dict",
]
