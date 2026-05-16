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
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from tac.exact_eval_custody import (
    is_sha256_hex,
    validate_exact_eval_evidence,
)
from tac.optimization.research_basis import (
    RESEARCH_SOURCES,
    ResearchBasisError,
    canonical_research_basis_id,
)

BandKind = Literal["delta_score", "absolute_score"]
SupersessionStatus = Literal["active", "superseded", "deprecated"]
AnchorStatus = Literal["absent", "pending", "diagnostic_only", "landed", "superseded"]
EXACT_EVAL_AXES = ("contest_cpu", "contest_cuda")
MIXED_EXACT_EVAL_AXIS = "mixed"


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


def _literal_json_bool(
    payload: Mapping[str, Any],
    key: str,
    *,
    default: bool,
) -> bool:
    if key not in payload:
        return default
    value = payload[key]
    if isinstance(value, bool):
        return value
    raise ValueError(f"{key} must be a literal JSON boolean")


def _is_transient_artifact_path(path_text: str) -> bool:
    return (
        path_text.startswith("/tmp/")
        or path_text.startswith("/var/tmp/")
        or path_text.startswith("/private/tmp/")
    )


def _repo_local_artifact_path(
    path_text: str,
    artifact_base_dir: Path | str,
) -> tuple[Path, str | None]:
    base_dir = Path(artifact_base_dir).resolve()
    stripped = path_text.removeprefix("file:").strip()
    if _is_transient_artifact_path(stripped):
        return Path(stripped), "transient"
    path = Path(stripped).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    resolved = path.resolve(strict=False)
    try:
        resolved.relative_to(base_dir)
    except ValueError:
        return resolved, "outside_base_dir"
    return resolved, None


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
        planning_only=_literal_json_bool(payload, "planning_only", default=True),
        score_claim=_literal_json_bool(payload, "score_claim", default=False),
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
    artifact_base_dir: Path | str | None = None,
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
    if band.axis.strip() and baseline.axis.strip() and baseline.axis != band.axis:
        blockers.append("prediction_band_baseline_axis_mismatch")
    if not is_sha256_hex(baseline.archive_sha256) or not is_sha256_hex(
        baseline.runtime_tree_sha256
    ):
        blockers.append("prediction_band_baseline_custody_missing")
    if not baseline.artifact_path.strip():
        blockers.append("prediction_band_baseline_artifact_missing")
    elif artifact_base_dir is None:
        blockers.append("prediction_band_baseline_artifact_base_dir_missing")
    else:
        baseline_artifact_path, baseline_artifact_error = _repo_local_artifact_path(
            baseline.artifact_path,
            artifact_base_dir,
        )
        if baseline_artifact_error == "transient":
            blockers.append("prediction_band_baseline_artifact_transient")
        elif baseline_artifact_error == "outside_base_dir":
            blockers.append("prediction_band_baseline_artifact_outside_repo")
        elif not baseline_artifact_path.is_file():
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
    elif empirical.status == "landed":
        anchor_exact_axes_seen: set[str] = set()
        if band.axis not in (*EXACT_EVAL_AXES, MIXED_EXACT_EVAL_AXIS):
            blockers.append("prediction_band_axis_not_exact_eval")
        if artifact_base_dir is None:
            blockers.append("prediction_band_empirical_anchor_artifact_base_dir_missing")
        for anchor_idx, anchor in enumerate(empirical.anchors):
            anchor_axis = str(anchor.get("axis") or "")
            expected_anchor_axis: str | None = None
            if not anchor_axis.strip():
                blockers.append("prediction_band_empirical_anchor_axis_missing")
                annotations.append(f"empirical_anchor_axis_missing_index={anchor_idx}")
            elif band.axis == MIXED_EXACT_EVAL_AXIS:
                if anchor_axis in EXACT_EVAL_AXES:
                    anchor_exact_axes_seen.add(anchor_axis)
                    expected_anchor_axis = anchor_axis
                else:
                    blockers.append("prediction_band_empirical_anchor_axis_mismatch")
                    annotations.append(
                        f"empirical_anchor_axis_mismatch_index={anchor_idx}:"
                        f"{anchor_axis!r}!={band.axis!r}"
                    )
            elif band.axis in EXACT_EVAL_AXES:
                expected_anchor_axis = band.axis
            elif band.axis.strip() and anchor_axis != band.axis:
                blockers.append("prediction_band_empirical_anchor_axis_mismatch")
                annotations.append(
                    f"empirical_anchor_axis_mismatch_index={anchor_idx}:"
                    f"{anchor_axis!r}!={band.axis!r}"
                )
            validation = validate_exact_eval_evidence(
                anchor,
                expected_axis=expected_anchor_axis,
                require_artifact_path=True,
                require_hardware=True,
                require_auth_eval_command=True,
                require_log_path=True,
                require_devices=True,
                annotation_prefix=f"empirical_anchor_{anchor_idx}",
                artifact_base_dir=artifact_base_dir,
            )
            blocker_map = {
                "axis_missing": "prediction_band_empirical_anchor_axis_missing",
                "axis_mismatch": "prediction_band_empirical_anchor_axis_mismatch",
                "score_missing": "prediction_band_empirical_anchor_score_missing",
                "archive_sha_invalid": "prediction_band_empirical_anchor_custody_missing",
                "runtime_tree_sha_invalid": "prediction_band_empirical_anchor_custody_missing",
                "artifact_path_missing": "prediction_band_empirical_anchor_artifact_missing",
                "artifact_path_file_missing": "prediction_band_empirical_anchor_artifact_missing",
                "artifact_path_transient": "prediction_band_empirical_anchor_artifact_missing",
                "artifact_path_outside_base_dir": "prediction_band_empirical_anchor_artifact_missing",
                "n_samples_missing": "prediction_band_empirical_anchor_n_samples_missing",
                "n_samples_not_contest_exact": "prediction_band_empirical_anchor_n_samples_not_contest_exact",
                "hardware_missing": "prediction_band_empirical_anchor_hardware_missing",
                "hardware_not_cuda": "prediction_band_empirical_anchor_hardware_not_cuda",
                "auth_eval_command_missing": "prediction_band_empirical_anchor_command_missing",
                "log_path_missing": "prediction_band_empirical_anchor_log_missing",
                "log_path_file_missing": "prediction_band_empirical_anchor_log_missing",
                "log_path_transient": "prediction_band_empirical_anchor_log_missing",
                "log_path_outside_base_dir": "prediction_band_empirical_anchor_log_missing",
                "inflate_device_missing": "prediction_band_empirical_anchor_inflate_device_missing",
                "inflate_device_not_cpu": "prediction_band_empirical_anchor_inflate_device_not_cpu",
                "inflate_device_not_cuda": "prediction_band_empirical_anchor_inflate_device_not_cuda",
                "eval_device_missing": "prediction_band_empirical_anchor_eval_device_missing",
                "eval_device_not_cpu": "prediction_band_empirical_anchor_eval_device_not_cpu",
                "eval_device_not_cuda": "prediction_band_empirical_anchor_eval_device_not_cuda",
                "archive_bytes_missing": "prediction_band_empirical_anchor_archive_bytes_missing",
                "seg_dist_missing": "prediction_band_empirical_anchor_seg_dist_missing",
                "pose_dist_missing": "prediction_band_empirical_anchor_pose_dist_missing",
                "score_formula_mismatch": "prediction_band_empirical_anchor_score_formula_mismatch",
            }
            annotation_map = {
                "score_missing": "empirical_anchor_score_missing_index",
                "archive_sha_invalid": "empirical_anchor_custody_missing_index",
                "runtime_tree_sha_invalid": "empirical_anchor_custody_missing_index",
                "artifact_path_missing": "empirical_anchor_artifact_missing_index",
                "artifact_path_file_missing": "empirical_anchor_artifact_missing_index",
                "artifact_path_transient": "empirical_anchor_artifact_missing_index",
                "artifact_path_outside_base_dir": "empirical_anchor_artifact_missing_index",
                "n_samples_missing": "empirical_anchor_n_samples_missing_index",
                "n_samples_not_contest_exact": "empirical_anchor_n_samples_not_contest_exact_index",
                "hardware_missing": "empirical_anchor_hardware_missing_index",
                "hardware_not_cuda": "empirical_anchor_hardware_not_cuda_index",
                "auth_eval_command_missing": "empirical_anchor_command_missing_index",
                "log_path_missing": "empirical_anchor_log_missing_index",
                "log_path_file_missing": "empirical_anchor_log_missing_index",
                "log_path_transient": "empirical_anchor_log_missing_index",
                "log_path_outside_base_dir": "empirical_anchor_log_missing_index",
                "inflate_device_missing": "empirical_anchor_inflate_device_missing_index",
                "inflate_device_not_cpu": "empirical_anchor_inflate_device_not_cpu_index",
                "inflate_device_not_cuda": "empirical_anchor_inflate_device_not_cuda_index",
                "eval_device_missing": "empirical_anchor_eval_device_missing_index",
                "eval_device_not_cpu": "empirical_anchor_eval_device_not_cpu_index",
                "eval_device_not_cuda": "empirical_anchor_eval_device_not_cuda_index",
                "archive_bytes_missing": "empirical_anchor_archive_bytes_missing_index",
                "seg_dist_missing": "empirical_anchor_seg_dist_missing_index",
                "pose_dist_missing": "empirical_anchor_pose_dist_missing_index",
            }
            for blocker in validation.blockers:
                public_blocker = blocker_map.get(blocker)
                if public_blocker is None:
                    continue
                blockers.append(public_blocker)
                annotation_key = annotation_map.get(blocker)
                if annotation_key is not None:
                    annotations.append(f"{annotation_key}={anchor_idx}")
            for validation_annotation in validation.annotations:
                if "score_formula_mismatch:" in validation_annotation:
                    mismatch = validation_annotation.rsplit(":", 1)[-1]
                    annotations.append(
                        f"empirical_anchor_score_formula_mismatch_index={anchor_idx}:"
                        f"{mismatch}"
                    )
        if band.axis == MIXED_EXACT_EVAL_AXIS:
            missing_axes = [
                axis for axis in EXACT_EVAL_AXES if axis not in anchor_exact_axes_seen
            ]
            if missing_axes:
                blockers.append("prediction_band_empirical_anchor_paired_axes_missing")
                annotations.append(
                    "empirical_anchor_paired_axes_missing="
                    + ",".join(missing_axes)
                )

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
    artifact_base_dir: Path | str | None = None,
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
        artifact_base_dir=artifact_base_dir,
    )


def verdict_to_dict(verdict: PredictionBandVerdict) -> dict[str, Any]:
    return verdict.to_dict()


def prediction_band_to_dict(band: PredictionBand) -> dict[str, Any]:
    return dataclasses.asdict(band)


__all__ = [
    "EXACT_EVAL_AXES",
    "MIXED_EXACT_EVAL_AXIS",
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
