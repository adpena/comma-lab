# SPDX-License-Identifier: MIT
"""Typed scorer-evidence bindings for parser-section MDL reports.

This module is intentionally analysis-only. It accepts already-produced scorer
evidence artifacts, such as component-response curves or penultimate-feature
saliency manifests, and binds them to parser-proven archive sections. The
binding is not a score claim and does not authorize dispatch.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SCHEMA = "tac_section_scorer_evidence_map_v1"
TOOL = "tac.analysis.scorer_section_evidence"

VALID_COMPONENTS = ("posenet", "segnet", "combined")
VALID_EVIDENCE_TYPES = ("component_response_curve", "penultimate_feature_saliency")
_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")

AXIS_LABELS = {
    "contest_cuda": "[contest-CUDA]",
    "cuda": "[contest-CUDA]",
    "contest_cpu": "[contest-CPU]",
    "cpu": "[contest-CPU]",
    "macos_cpu_advisory": "[macOS-CPU advisory]",
    "mps": "[MPS advisory]",
    "mps_advisory": "[MPS advisory]",
    "mps_proxy": "[MPS advisory]",
    "proxy": "[proxy]",
    "diagnostic_cuda": "[diagnostic-CUDA]",
    "unknown": "[axis-unknown]",
}


class SectionScorerEvidenceError(ValueError):
    """Raised when a section scorer-evidence map cannot be parsed."""


def axis_label(axis: str | None) -> str:
    """Return the explicit report label for an evidence axis."""

    key = (axis or "unknown").strip().lower().replace("-", "_")
    return AXIS_LABELS.get(key, "[axis-unknown]")


def _as_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    return default


def _as_int_or_none(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


@dataclass(frozen=True)
class ScorerEvidenceArtifact:
    """Custody envelope for a scorer-evidence artifact."""

    artifact_type: str
    path: str
    sha256: str | None = None
    bytes: int | None = None
    evidence_grade: str = ""
    official_component_response: bool = False
    canonical_scorer_path: bool = False
    passed: bool = False
    promotion_eligible: bool = False
    score_claim: bool = False
    analysis_time_only: bool | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "ScorerEvidenceArtifact":
        artifact_type = str(
            payload.get("artifact_type")
            or payload.get("type")
            or payload.get("format")
            or "unknown_artifact"
        )
        return cls(
            artifact_type=artifact_type,
            path=str(payload.get("path") or ""),
            sha256=str(payload["sha256"]) if isinstance(payload.get("sha256"), str) else None,
            bytes=_as_int_or_none(payload.get("bytes")),
            evidence_grade=str(payload.get("evidence_grade") or ""),
            official_component_response=_as_bool(payload.get("official_component_response")),
            canonical_scorer_path=_as_bool(payload.get("canonical_scorer_path")),
            passed=_as_bool(payload.get("passed")),
            promotion_eligible=_as_bool(payload.get("promotion_eligible")),
            score_claim=_as_bool(payload.get("score_claim")),
            analysis_time_only=(
                bool(payload["analysis_time_only"])
                if isinstance(payload.get("analysis_time_only"), bool)
                else None
            ),
            metadata=dict(payload.get("metadata") or {}),
        )

    def custody_blockers(self, *, repo_root: Path | None = None) -> list[str]:
        blockers: list[str] = []
        if not self.path:
            blockers.append("evidence_artifact_path_missing")
        if self.bytes is None:
            blockers.append("evidence_artifact_bytes_missing")
        elif self.bytes < 0:
            blockers.append("evidence_artifact_bytes_negative")
        if not self.sha256:
            blockers.append("evidence_artifact_sha256_missing")
        elif not _SHA256_RE.match(self.sha256):
            blockers.append("evidence_artifact_sha256_invalid")
        if self.score_claim:
            blockers.append("evidence_artifact_score_claim_not_allowed")
        if self.promotion_eligible:
            blockers.append("evidence_artifact_promotion_eligible_not_allowed")

        if repo_root is not None and self.path:
            artifact_path = Path(self.path)
            if not artifact_path.is_absolute():
                artifact_path = repo_root / artifact_path
            if artifact_path.exists() and artifact_path.is_file() and self.bytes is not None:
                actual_bytes = artifact_path.stat().st_size
                if actual_bytes != self.bytes:
                    blockers.append("evidence_artifact_bytes_mismatch_on_disk")
        return blockers

    def to_manifest(self) -> dict[str, Any]:
        return {
            "artifact_type": self.artifact_type,
            "path": self.path,
            "bytes": self.bytes,
            "sha256": self.sha256,
            "evidence_grade": self.evidence_grade,
            "official_component_response": self.official_component_response,
            "canonical_scorer_path": self.canonical_scorer_path,
            "passed": self.passed,
            "promotion_eligible": self.promotion_eligible,
            "score_claim": self.score_claim,
            "analysis_time_only": self.analysis_time_only,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class SectionScorerEvidenceBinding:
    """One section-to-scorer-component binding."""

    archive_label: str
    section_name: str
    component: str
    evidence_type: str
    axis: str
    artifact: ScorerEvidenceArtifact
    binding_strength: str = "unknown"
    feature_target_id: str = ""
    notes: str = ""

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "SectionScorerEvidenceBinding":
        artifact_payload = payload.get("artifact") or payload.get("source_artifact") or {}
        if isinstance(artifact_payload, str):
            artifact_payload = {"path": artifact_payload}
        if not isinstance(artifact_payload, Mapping):
            raise SectionScorerEvidenceError("binding artifact must be an object or path string")
        return cls(
            archive_label=str(payload.get("archive_label") or ""),
            section_name=str(payload.get("section_name") or ""),
            component=str(payload.get("component") or ""),
            evidence_type=str(payload.get("evidence_type") or ""),
            axis=str(payload.get("axis") or payload.get("score_axis") or "unknown"),
            artifact=ScorerEvidenceArtifact.from_mapping(artifact_payload),
            binding_strength=str(payload.get("binding_strength") or "unknown"),
            feature_target_id=str(payload.get("feature_target_id") or ""),
            notes=str(payload.get("notes") or ""),
        )

    @property
    def axis_label(self) -> str:
        return axis_label(self.axis)

    def blockers(self, *, repo_root: Path | None = None) -> list[str]:
        blockers = self.artifact.custody_blockers(repo_root=repo_root)
        if not self.archive_label:
            blockers.append("archive_label_missing")
        if not self.section_name:
            blockers.append("section_name_missing")
        if self.component not in VALID_COMPONENTS:
            blockers.append("component_not_in_posenet_segnet_combined")
        if self.evidence_type not in VALID_EVIDENCE_TYPES:
            blockers.append("unsupported_section_scorer_evidence_type")
        if self.axis_label == "[axis-unknown]":
            blockers.append("explicit_axis_label_missing")

        if self.evidence_type == "component_response_curve":
            if not self.artifact.official_component_response:
                blockers.append("official_component_response_not_true")
            if not self.artifact.canonical_scorer_path:
                blockers.append("canonical_scorer_path_not_true")
            if not self.artifact.passed:
                blockers.append("component_response_curve_not_passed")
        elif self.evidence_type == "penultimate_feature_saliency":
            ready = (
                self.artifact.metadata.get("byte_to_scorer_feature_binding_ready")
                or self.artifact.metadata.get("feature_binding_ready")
            )
            if self.artifact.analysis_time_only is not True:
                blockers.append("penultimate_saliency_analysis_time_only_not_true")
            if ready is not True:
                blockers.append("byte_to_scorer_feature_binding_ready_not_true")
        return blockers

    def true_scorer_ready(self, *, repo_root: Path | None = None) -> bool:
        return not self.blockers(repo_root=repo_root)

    def to_manifest(self, *, repo_root: Path | None = None) -> dict[str, Any]:
        blockers = self.blockers(repo_root=repo_root)
        return {
            "archive_label": self.archive_label,
            "section_name": self.section_name,
            "component": self.component,
            "evidence_type": self.evidence_type,
            "axis": self.axis,
            "axis_label": self.axis_label,
            "binding_strength": self.binding_strength,
            "feature_target_id": self.feature_target_id,
            "notes": self.notes,
            "artifact": self.artifact.to_manifest(),
            "true_scorer_ready": not blockers,
            "blockers": blockers,
            "score_claim": False,
        }


@dataclass(frozen=True)
class SectionScorerEvidenceMap:
    """Typed section-level scorer evidence map."""

    bindings: tuple[SectionScorerEvidenceBinding, ...]
    schema: str = SCHEMA
    schema_version: int = 1
    source: str = ""
    score_claim: bool = False
    promotion_eligible: bool = False
    ready_for_exact_eval_dispatch: bool = False

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "SectionScorerEvidenceMap":
        raw_bindings = payload.get("bindings")
        if not isinstance(raw_bindings, Sequence) or isinstance(raw_bindings, (str, bytes)):
            raise SectionScorerEvidenceError("section scorer evidence map requires bindings[]")
        if _as_bool(payload.get("score_claim")):
            raise SectionScorerEvidenceError("section scorer evidence map must not claim score")
        return cls(
            bindings=tuple(
                SectionScorerEvidenceBinding.from_mapping(item)
                for item in raw_bindings
                if isinstance(item, Mapping)
            ),
            schema=str(payload.get("schema") or SCHEMA),
            schema_version=int(payload.get("schema_version") or 1),
            source=str(payload.get("source") or ""),
            score_claim=False,
            promotion_eligible=_as_bool(payload.get("promotion_eligible")),
            ready_for_exact_eval_dispatch=_as_bool(payload.get("ready_for_exact_eval_dispatch")),
        )

    @classmethod
    def from_json_path(cls, path: str | Path) -> "SectionScorerEvidenceMap":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise SectionScorerEvidenceError("section scorer evidence json must be an object")
        return cls.from_mapping(payload)

    def bindings_for(self, archive_label: str, section_name: str) -> tuple[SectionScorerEvidenceBinding, ...]:
        return tuple(
            binding
            for binding in self.bindings
            if binding.archive_label == archive_label and binding.section_name == section_name
        )

    def to_manifest(self, *, repo_root: Path | None = None) -> dict[str, Any]:
        binding_rows = [binding.to_manifest(repo_root=repo_root) for binding in self.bindings]
        blockers: list[str] = []
        if self.promotion_eligible:
            blockers.append("section_evidence_map_promotion_eligible_not_allowed")
        if self.ready_for_exact_eval_dispatch:
            blockers.append("section_evidence_map_dispatch_ready_not_allowed")
        if not binding_rows:
            blockers.append("section_evidence_map_bindings_missing")
        for row in binding_rows:
            for blocker in row.get("blockers", []):
                blockers.append(f"{row.get('archive_label')}:{row.get('section_name')}:{blocker}")
        return {
            "schema": self.schema,
            "schema_version": self.schema_version,
            "tool": TOOL,
            "source": self.source,
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
            "binding_count": len(binding_rows),
            "axis_labels": sorted({str(row.get("axis_label")) for row in binding_rows}),
            "true_scorer_ready_binding_count": sum(
                1 for row in binding_rows if row.get("true_scorer_ready") is True
            ),
            "bindings": binding_rows,
            "blockers": sorted(set(blockers)),
        }


def coerce_section_scorer_evidence_map(
    value: SectionScorerEvidenceMap | Mapping[str, Any] | str | Path | None,
) -> SectionScorerEvidenceMap | None:
    """Return a typed map from common API/CLI inputs."""

    if value is None:
        return None
    if isinstance(value, SectionScorerEvidenceMap):
        return value
    if isinstance(value, Mapping):
        return SectionScorerEvidenceMap.from_mapping(value)
    return SectionScorerEvidenceMap.from_json_path(value)


__all__ = [
    "AXIS_LABELS",
    "SCHEMA",
    "TOOL",
    "ScorerEvidenceArtifact",
    "SectionScorerEvidenceBinding",
    "SectionScorerEvidenceError",
    "SectionScorerEvidenceMap",
    "axis_label",
    "coerce_section_scorer_evidence_map",
]
