# SPDX-License-Identifier: MIT
"""Fail-closed adapters for heterogeneous counted-application receipts.

J8F, PF3, and J12 predate :mod:`tac.analysis.applied_action_receipt` and do
not share a wire format.  These adapters relocate only identities that the
source artifacts actually carry.  A source that omits a required identity is
returned as a typed blocker result; no archive, receiver, uint8-change, or
byte-home identity is synthesized to make the row fit.

PF3's physical edge is always ``V19C_BASE -> one RG3 coordinate``.  It is not
an RD1 hull edge, even when an RD1 cell names the same score stratum.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tac.analysis.applied_action_receipt import (
    APPLIED_ACTION_RECEIPT_SCHEMA,
    AppliedActionReceipt,
)
from tac.score_geometry import contest_score

ADAPTER_RESULT_SCHEMA = "tac.applied_action_adapter_result.v1"
ADAPTER_MANIFEST_SCHEMA = "tac.applied_action_adapter_manifest.v1"

J8F_SMOKE_SCHEMA = "ddm_j8f_counted_application_smoke.v1"
J8F_CHECKPOINT_SCHEMA = "ddm_j8f_counted_application_checkpoint.v1"
J8F_APPLICATION_SCHEMA = "ddm_dm4_j5_counted_application.v1"
J8F_CONFIG_SCHEMA = "ddm_dm4_j5_counted_application_config.v1"
PF3_RECEIPT_SCHEMA = "ddm_pf3_finite_price_materialization_receipt.v1"
PF3_CHECKPOINT_SCHEMA = "ddm_pf3_coordinate_measurement_checkpoint.v1"
J12_RECEIPT_SCHEMA = "ddm_j12_366_receiver_coordinate_custody_compact_receipt.v1"

PF3_PHYSICAL_EDGE = "V19C_BASE_TO_ONE_EXACT_RG3_COORDINATE"
LOCAL_ADVISORY_AXIS = "[macOS-CPU frozen-scorer advisory]"
_MANIFEST_KEYS = frozenset(
    {
        "schema",
        "results",
        "receipt_count",
        "blocked_source_count",
        "source_artifact_count",
        "research_only",
        "promotion_eligible",
        "score_claim",
        "content_sha256",
    }
)
_RESULT_KEYS = frozenset(
    {
        "schema",
        "source_kind",
        "source_schema",
        "source_id",
        "ok",
        "receipt",
        "blockers",
        "source_artifacts",
        "source_artifact_count",
        "source_artifact_digest_sha256",
        "source_counts",
        "research_only",
        "promotion_eligible",
        "score_claim",
    }
)
_EXPECTED_SOURCE_COUNTS: Mapping[str, Mapping[str, int]] = {
    "J8F": {"application_count": 12, "checkpoint_artifact_count": 12},
    "PF3": {
        "candidate_artifact_count": 16,
        "coordinate_family_count": 3,
        "uphill_edge_count": 16,
    },
    "J12": {"composite_count": 8, "sealed_proposal_count": 4, "single_count": 16},
}


class AppliedActionAdapterError(ValueError):
    """Raised when a source claims one schema but violates its own contract."""


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise AppliedActionAdapterError(f"{name} must be a mapping")
    return value


def _require_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AppliedActionAdapterError(f"{name} must be a non-empty string")
    return value.strip()


def _require_sha256(value: Any, name: str) -> str:
    text = _require_text(value, name).lower()
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
        raise AppliedActionAdapterError(f"{name} must be lowercase SHA-256 hex")
    return text


def _require_int(value: Any, name: str, *, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise AppliedActionAdapterError(f"{name} must be an exact integer")
    if value < (1 if positive else 0):
        bound = "positive" if positive else "non-negative"
        raise AppliedActionAdapterError(f"{name} must be {bound}")
    return value


def _require_signed_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise AppliedActionAdapterError(f"{name} must be an exact integer")
    return value


def _canonical_sha256(value: Any) -> str:
    raw = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def canonical_json_sha256(value: Any) -> str:
    """Return the deterministic semantic hash used to bind parsed JSON."""

    return _canonical_sha256(value)


def _close(left: float, right: float, *, tolerance: float = 1e-12) -> bool:
    return math.isclose(left, right, rel_tol=0.0, abs_tol=tolerance)


def _require_finite(value: Any, name: str, *, nonnegative: bool = True) -> float:
    if isinstance(value, bool):
        raise AppliedActionAdapterError(f"{name} must be finite")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise AppliedActionAdapterError(f"{name} must be finite") from exc
    if not math.isfinite(result) or (nonnegative and result < 0.0):
        raise AppliedActionAdapterError(f"{name} must be finite and non-negative")
    return result


@dataclass(frozen=True)
class SourceArtifactIdentity:
    """One file identity verified against the exact bytes consumed."""

    path: str
    bytes: int
    sha256: str
    role: str
    json_content_sha256: str | None = None

    def __post_init__(self) -> None:
        _require_text(self.path, "artifact path")
        _require_int(self.bytes, "artifact bytes")
        _require_sha256(self.sha256, "artifact sha256")
        _require_text(self.role, "artifact role")
        if self.json_content_sha256 is not None:
            _require_sha256(self.json_content_sha256, "artifact json_content_sha256")

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "bytes": self.bytes,
            "sha256": self.sha256,
            "role": self.role,
            "json_content_sha256": self.json_content_sha256,
        }


def verify_source_artifact(
    path: str | Path,
    *,
    role: str,
    expected_bytes: int | None = None,
    expected_sha256: str | None = None,
    display_path: str | None = None,
) -> SourceArtifactIdentity:
    """Hash one regular, non-symlink file and optionally enforce declared custody."""

    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise AppliedActionAdapterError(f"source artifact is not a regular file: {source}")
    digest = hashlib.sha256()
    size = 0
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            size += len(chunk)
            digest.update(chunk)
    actual_sha = digest.hexdigest()
    if expected_bytes is not None and size != expected_bytes:
        raise AppliedActionAdapterError(f"source artifact byte count differs: {source}")
    if expected_sha256 is not None and actual_sha != _require_sha256(expected_sha256, f"expected sha256 for {source}"):
        raise AppliedActionAdapterError(f"source artifact SHA-256 differs: {source}")
    return SourceArtifactIdentity(
        path=display_path or str(source),
        bytes=size,
        sha256=actual_sha,
        role=role,
    )


def load_json_artifact(
    path: str | Path,
    *,
    role: str,
    expected_bytes: int | None = None,
    expected_sha256: str | None = None,
    display_path: str | None = None,
) -> tuple[Mapping[str, Any], SourceArtifactIdentity]:
    """Load JSON from the exact bytes whose identity is returned alongside it."""

    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise AppliedActionAdapterError(f"source artifact is not a regular file: {source}")
    try:
        raw = source.read_bytes()
        payload = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AppliedActionAdapterError(f"invalid JSON artifact: {source}") from exc
    actual_sha = hashlib.sha256(raw).hexdigest()
    if expected_bytes is not None and len(raw) != expected_bytes:
        raise AppliedActionAdapterError(f"source artifact byte count differs: {source}")
    if expected_sha256 is not None and actual_sha != _require_sha256(expected_sha256, f"expected sha256 for {source}"):
        raise AppliedActionAdapterError(f"source artifact SHA-256 differs: {source}")
    mapping = _require_mapping(payload, str(source))
    return mapping, SourceArtifactIdentity(
        path=display_path or str(source),
        bytes=len(raw),
        sha256=actual_sha,
        role=role,
        json_content_sha256=_canonical_sha256(mapping),
    )


def _bind_json_payload(payload: Mapping[str, Any], artifact: SourceArtifactIdentity, name: str) -> None:
    if artifact.json_content_sha256 is None:
        raise AppliedActionAdapterError(f"{name} artifact lacks JSON content identity")
    if _canonical_sha256(payload) != artifact.json_content_sha256:
        raise AppliedActionAdapterError(f"{name} payload differs from bound artifact")


def _normalize_artifacts(
    artifacts: Sequence[SourceArtifactIdentity],
) -> tuple[SourceArtifactIdentity, ...]:
    ordered = tuple(sorted(artifacts, key=lambda item: (item.path, item.role)))
    if not ordered:
        raise AppliedActionAdapterError("adaptation result requires source artifacts")
    if len({item.path for item in ordered}) != len(ordered):
        raise AppliedActionAdapterError("source artifact paths must be unique")
    return ordered


@dataclass(frozen=True)
class AdapterBlocker:
    """One exact missing or contradictory source obligation."""

    code: str
    source_key: str
    owed_field: str
    detail: str

    def __post_init__(self) -> None:
        for name, value in (
            ("code", self.code),
            ("source_key", self.source_key),
            ("owed_field", self.owed_field),
            ("detail", self.detail),
        ):
            _require_text(value, name)

    def as_dict(self) -> dict[str, str]:
        return {
            "code": self.code,
            "source_key": self.source_key,
            "owed_field": self.owed_field,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class AdaptationResult:
    """Exactly one valid receipt or one-or-more explicit blockers."""

    source_kind: str
    source_schema: str
    source_id: str
    source_artifacts: tuple[SourceArtifactIdentity, ...]
    source_counts: tuple[tuple[str, int], ...]
    receipt: AppliedActionReceipt | None = None
    blockers: tuple[AdapterBlocker, ...] = ()

    def __post_init__(self) -> None:
        for name, value in (
            ("source_kind", self.source_kind),
            ("source_schema", self.source_schema),
            ("source_id", self.source_id),
        ):
            _require_text(value, name)
        if (self.receipt is None) == (not self.blockers):
            raise AppliedActionAdapterError("adaptation result must carry exactly one of receipt or blockers")
        if self.receipt is not None and self.receipt.schema != APPLIED_ACTION_RECEIPT_SCHEMA:
            raise AppliedActionAdapterError("adapter emitted a foreign receipt schema")
        normalized = _normalize_artifacts(self.source_artifacts)
        if normalized != self.source_artifacts:
            raise AppliedActionAdapterError("source artifacts must use canonical path order")
        if tuple(sorted(self.source_counts)) != self.source_counts:
            raise AppliedActionAdapterError("source counts must use canonical key order")
        if len({key for key, _ in self.source_counts}) != len(self.source_counts):
            raise AppliedActionAdapterError("source count keys must be unique")
        for key, value in self.source_counts:
            _require_text(key, "source count key")
            _require_int(value, f"source_counts.{key}")

    @property
    def ok(self) -> bool:
        return self.receipt is not None

    def as_dict(self) -> dict[str, Any]:
        artifacts = [artifact.as_dict() for artifact in self.source_artifacts]
        return {
            "schema": ADAPTER_RESULT_SCHEMA,
            "source_kind": self.source_kind,
            "source_schema": self.source_schema,
            "source_id": self.source_id,
            "ok": self.ok,
            "receipt": self.receipt.as_dict() if self.receipt is not None else None,
            "blockers": [blocker.as_dict() for blocker in self.blockers],
            "source_artifacts": artifacts,
            "source_artifact_count": len(artifacts),
            "source_artifact_digest_sha256": _canonical_sha256(artifacts),
            "source_counts": dict(self.source_counts),
            "research_only": True,
            "promotion_eligible": False,
            "score_claim": False,
        }


def _j8f_operator_sha(config: Mapping[str, Any]) -> str:
    if config.get("schema") != J8F_CONFIG_SCHEMA:
        raise AppliedActionAdapterError("J8F source config schema differs")
    bindings = _require_mapping(config.get("source_bindings"), "J8F source_bindings")
    operator = _require_mapping(bindings.get("operator_source"), "J8F operator_source")
    return _require_sha256(operator.get("sha256"), "J8F operator_source.sha256")


def _verify_declared_artifacts(
    declarations: Mapping[str, Any], artifacts: Sequence[SourceArtifactIdentity], name: str
) -> None:
    available = {artifact.path: artifact for artifact in artifacts}
    for key, raw in declarations.items():
        row = _require_mapping(raw, f"{name}.{key}")
        path = _require_text(row.get("path"), f"{name}.{key}.path")
        artifact = available.get(path)
        if artifact is None:
            raise AppliedActionAdapterError(f"{name}.{key} bytes were not custody-verified")
        if artifact.bytes != _require_int(row.get("bytes"), f"{name}.{key}.bytes") or (
            artifact.sha256 != _require_sha256(row.get("sha256"), f"{name}.{key}.sha256")
        ):
            raise AppliedActionAdapterError(f"{name}.{key} custody differs")


def _has_custody(
    artifacts: Sequence[SourceArtifactIdentity], *, sha256: str, bytes_count: int, path: str | None = None
) -> bool:
    return any(
        artifact.sha256 == sha256 and artifact.bytes == bytes_count and (path is None or artifact.path == path)
        for artifact in artifacts
    )


def _j8f_source_state(smoke: Mapping[str, Any]) -> tuple[str, int]:
    step4 = _require_mapping(smoke.get("step4"), "J8F step4")
    return (
        _require_sha256(step4.get("archive_sha256"), "J8F step4.archive_sha256"),
        _require_int(step4.get("archive_bytes"), "J8F step4.archive_bytes", positive=True),
    )


def adapt_j8f_checkpoints(
    smoke: Mapping[str, Any],
    checkpoints: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    *,
    smoke_artifact: SourceArtifactIdentity,
    checkpoint_artifacts: Sequence[SourceArtifactIdentity],
    config_artifact: SourceArtifactIdentity,
    custody_artifacts: Sequence[SourceArtifactIdentity],
) -> tuple[AdaptationResult, ...]:
    """Validate J8F's ordered composition and expose its exact relocation debt.

    J8F preserves twelve exact per-step sparse change identities and measures
    the final composition.  It does *not* preserve the final base-to-candidate
    sparse uint8 identity, a typed single stream home, or independent scorer
    transitions for the twelve steps.  Those are distinct obligations in the
    universal receipt.  The adapter therefore emits blockers instead of
    relabeling incidence sums as a final diff, inventing a byte home, or
    manufacturing a macro receipt.
    """

    smoke = _require_mapping(smoke, "J8F smoke")
    config = _require_mapping(config, "J8F config")
    _bind_json_payload(smoke, smoke_artifact, "J8F smoke")
    _bind_json_payload(config, config_artifact, "J8F config")
    if len(checkpoints) != len(checkpoint_artifacts):
        raise AppliedActionAdapterError("J8F checkpoint artifact count differs")
    for index, (checkpoint, artifact) in enumerate(zip(checkpoints, checkpoint_artifacts, strict=True)):
        _bind_json_payload(checkpoint, artifact, f"J8F checkpoint {index}")
    if smoke.get("schema") != J8F_SMOKE_SCHEMA:
        raise AppliedActionAdapterError("J8F smoke schema differs")
    if smoke.get("research_only") is not True or smoke.get("score_claim") is not False:
        raise AppliedActionAdapterError("J8F false-authority contract differs")
    run_id = _require_text(smoke.get("run_id"), "J8F run_id")
    evidence_axis = _require_text(smoke.get("evidence_axis"), "J8F evidence_axis")
    if evidence_axis != LOCAL_ADVISORY_AXIS:
        raise AppliedActionAdapterError("J8F evidence axis differs")
    _j8f_operator_sha(config)
    config_bindings = _require_mapping(config.get("source_bindings"), "J8F source_bindings")
    _verify_declared_artifacts(config_bindings, custody_artifacts, "J8F source binding")
    prior_archive_sha, base_archive_bytes = _j8f_source_state(smoke)
    prior_archive_bytes = base_archive_bytes

    ordered = sorted(
        (_require_mapping(checkpoint, "J8F checkpoint") for checkpoint in checkpoints),
        key=lambda row: _require_int(row.get("step_index"), "J8F checkpoint.step_index"),
    )
    if not ordered:
        raise AppliedActionAdapterError("J8F checkpoints are empty")
    if len(ordered) != 12:
        raise AppliedActionAdapterError("J8F counted-application horizon differs from 12")
    expected_steps = list(range(len(ordered)))
    observed_steps = [int(row["step_index"]) for row in ordered]
    if observed_steps != expected_steps:
        raise AppliedActionAdapterError(f"J8F checkpoint sequence differs: {observed_steps} != {expected_steps}")
    prior_prefix_hashes: tuple[str, ...] = ()
    final_applications: list[Mapping[str, Any]] = []
    for ordinal, checkpoint in enumerate(ordered):
        if checkpoint.get("schema") != J8F_CHECKPOINT_SCHEMA:
            raise AppliedActionAdapterError("J8F checkpoint schema differs")
        if checkpoint.get("score_claim") is not False:
            raise AppliedActionAdapterError("J8F checkpoint score_claim differs")
        applications = checkpoint.get("application_receipts")
        if not isinstance(applications, list) or len(applications) != ordinal + 1:
            raise AppliedActionAdapterError("J8F checkpoint must preserve the complete cumulative application prefix")
        if [row.get("step_index") for row in applications if isinstance(row, Mapping)] != list(range(ordinal + 1)):
            raise AppliedActionAdapterError("J8F cumulative application prefix differs")
        application_rows = tuple(
            _require_mapping(row, f"J8F application prefix {index}") for index, row in enumerate(applications)
        )
        prefix_hashes = tuple(_canonical_sha256(row) for row in application_rows)
        if prefix_hashes[:-1] != prior_prefix_hashes:
            raise AppliedActionAdapterError("J8F cumulative application content prefix differs")
        prior_prefix_hashes = prefix_hashes
        final_applications = list(application_rows)
        application = _require_mapping(applications[-1], "J8F application")
        if application.get("schema") != J8F_APPLICATION_SCHEMA:
            raise AppliedActionAdapterError("J8F application schema differs")
        if application.get("evidence_axis") != evidence_axis:
            raise AppliedActionAdapterError("J8F application evidence axis differs")
        projected = _require_mapping(application.get("projected_application"), "J8F projected_application")
        proposal = _require_mapping(application.get("proposal"), "J8F proposal")
        support = _require_mapping(proposal.get("support_footprint"), "J8F support_footprint")
        trust = _require_mapping(application.get("trust_region"), "J8F trust_region")
        projected_state = _require_mapping(checkpoint.get("projected_state"), "J8F projected_state")
        if projected_state.get("parseback_exact") is not True:
            raise AppliedActionAdapterError("J8F cumulative projected state parseback differs")

        _require_sha256(projected.get("archive_sha256"), "J8F projected archive SHA")
        candidate_archive_sha = _require_sha256(
            projected_state.get("archive_sha256"), "J8F projected state archive SHA"
        )
        candidate_archive_bytes = _require_int(
            projected_state.get("archive_bytes"), "J8F projected state archive bytes", positive=True
        )
        _require_int(projected.get("archive_bytes"), "J8F application archive bytes", positive=True)
        _require_signed_int(projected.get("archive_byte_delta"), "J8F archive_byte_delta")

        _require_int(projected.get("coordinate_index"), "J8F coordinate_index")
        _require_text(projected.get("coordinate_name"), "J8F coordinate_name")
        _require_int(projected.get("pair_id"), "J8F pair_id")
        direction = projected.get("direction")
        if direction not in {-1, 1}:
            raise AppliedActionAdapterError("J8F direction must be -1 or 1")
        _require_int(projected.get("changed_channel_values"), "J8F changed_channel_values", positive=True)
        _require_sha256(
            projected.get("delta_sha256_int64_indices_int16_values"),
            "J8F changed uint8 sparse identity",
        )
        _require_sha256(support.get("stem_block_indices_sha256_uint32le"), "J8F support SHA")
        _require_text(proposal.get("proposal_id"), "J8F proposal_id")
        _require_text(
            _require_mapping(proposal.get("aimed_cell"), "J8F aimed_cell").get("bucket_id"),
            "J8F bucket_id",
        )
        quantum = _require_int(trust.get("coordinate_quantum"), "J8F coordinate_quantum", positive=True)
        if quantum != 1:
            raise AppliedActionAdapterError("J8F source is no longer one-quantum-per-coordinate")

        prior_archive_sha = candidate_archive_sha
        prior_archive_bytes = candidate_archive_bytes
        _require_sha256(projected_state.get("theta_sha256_float32le"), "J8F projected theta SHA")

    final_arm = _require_mapping(smoke.get("range_gauge_projected_arm"), "J8F range_gauge_projected_arm")
    final_archive = _require_mapping(final_arm.get("archive"), "J8F final archive")
    final_verdict = _require_mapping(final_arm.get("verdict"), "J8F final verdict")
    step4 = _require_mapping(smoke.get("step4"), "J8F step4")
    reference = _require_mapping(step4.get("reference"), "J8F step4.reference")
    horizon = _require_mapping(smoke.get("application"), "J8F application horizon")
    stage_receipts = horizon.get("stage_receipts")
    if horizon.get("horizon") != 12 or not isinstance(stage_receipts, list):
        raise AppliedActionAdapterError("J8F application horizon shape differs")
    if tuple(_canonical_sha256(row) for row in stage_receipts) != tuple(
        _canonical_sha256(row) for row in final_applications
    ):
        raise AppliedActionAdapterError("J8F final horizon does not bind the checkpoint prefix")
    final_archive_sha = _require_sha256(final_archive.get("sha256"), "J8F final archive SHA")
    if final_archive_sha != prior_archive_sha:
        raise AppliedActionAdapterError("J8F final archive is not the last projected checkpoint")
    final_archive_bytes = _require_int(final_archive.get("bytes"), "J8F final archive bytes", positive=True)
    if final_archive_bytes != prior_archive_bytes:
        raise AppliedActionAdapterError("J8F final archive bytes differ from checkpoint chain")
    if final_archive.get("parseback_exact") is not True:
        raise AppliedActionAdapterError("J8F final archive parseback is not exact")
    if not _has_custody(
        custody_artifacts,
        sha256=final_archive_sha,
        bytes_count=final_archive_bytes,
        path=_require_text(final_archive.get("path"), "J8F final archive path"),
    ):
        raise AppliedActionAdapterError("J8F final archive bytes were not custody-verified")

    for label, verdict, expected_sha, expected_bytes in (
        ("reference", reference, _j8f_source_state(smoke)[0], base_archive_bytes),
        ("candidate", final_verdict, final_archive_sha, final_archive_bytes),
    ):
        if verdict.get("num_pairs") != 600:
            raise AppliedActionAdapterError(f"J8F {label} verdict is not n600")
        if verdict.get("evidence_axis") != evidence_axis:
            raise AppliedActionAdapterError(f"J8F {label} verdict evidence axis differs")
        if verdict.get("archive_sha256") != expected_sha or verdict.get("archive_bytes") != expected_bytes:
            raise AppliedActionAdapterError(f"J8F {label} scorer-to-archive foreign key differs")
        if verdict.get("score_claim") is not False or verdict.get("promotion_eligible") is not False:
            raise AppliedActionAdapterError(f"J8F {label} false-authority contract differs")

    old_d_seg = _require_finite(reference.get("d_seg"), "J8F reference d_seg")
    old_d_pose = _require_finite(reference.get("d_pose"), "J8F reference d_pose")
    new_d_seg = _require_finite(final_verdict.get("d_seg"), "J8F candidate d_seg")
    new_d_pose = _require_finite(final_verdict.get("d_pose"), "J8F candidate d_pose")
    delta_score = contest_score(new_d_seg, new_d_pose, final_archive_bytes) - contest_score(
        old_d_seg, old_d_pose, base_archive_bytes
    )
    source_delta = _require_mapping(final_arm.get("delta_vs_step4"), "J8F delta_vs_step4")
    seg_term = 100.0 * (new_d_seg - old_d_seg)
    pose_term = math.sqrt(10.0 * new_d_pose) - math.sqrt(10.0 * old_d_pose)
    rate_term = 25.0 * (final_archive_bytes - base_archive_bytes) / 37_545_489
    for key, expected in (
        ("seg_term", seg_term),
        ("pose_term", pose_term),
        ("rate_term", rate_term),
        ("joint_delta", delta_score),
    ):
        if not _close(_require_finite(source_delta.get(key), f"J8F {key}", nonnegative=False), expected):
            raise AppliedActionAdapterError(f"J8F nonlinear {key} does not reconcile")
    if source_delta.get("acceptance_authority") != "strict_exact_n600_joint_delta_lt_zero":
        raise AppliedActionAdapterError("J8F acceptance authority differs")
    if delta_score >= 0.0:
        raise AppliedActionAdapterError("J8F projected composite is not measured downhill")
    blockers = (
        AdapterBlocker(
            code="J8F_FINAL_CHANGED_UINT8_IDENTITY_ABSENT",
            source_key="application[*].projected_application.delta_sha256_int64_indices_int16_values",
            owed_field="changed_uint8_count,changed_uint8_sha256",
            detail=(
                "The twelve ordered per-step sparse identities are preserved, but their incidence "
                "sum/hash is not the unique final base-to-candidate uint8 diff required by v1."
            ),
        ),
        AdapterBlocker(
            code="J8F_SINGLE_LAWFUL_BYTE_HOME_ABSENT",
            source_key="range_gauge_projected_arm.archive.bytes",
            owed_field="stream_home",
            detail=(
                "The source preserves whole-archive bytes but no typed stream/layer/coder-owner "
                "foreign key; J5 coordinate sequence is not an EV2 seven-home allocation."
            ),
        ),
        AdapterBlocker(
            code="J8F_PER_STEP_SCORE_TRANSITIONS_ABSENT",
            source_key="application[*].projected_application",
            owed_field="action_effect.old/new d_seg,d_pose,bytes per applied codeword",
            detail=(
                "Only the final twelve-action composition has an n600 scorer transition, so the "
                "individually identity-complete applications cannot be emitted as measured rows."
            ),
        ),
        AdapterBlocker(
            code="J8F_REFERENCE_ARCHIVE_BYTES_ABSENT",
            source_key="step4.reference.archive_sha256",
            owed_field="file-backed reference archive bytes",
            detail=(
                "The n600 reference verdict and its archive foreign key are preserved, but the "
                "referenced base archive bytes are not present in the J8F custody bundle."
            ),
        ),
    )
    artifacts = _normalize_artifacts((smoke_artifact, config_artifact, *checkpoint_artifacts, *custody_artifacts))
    return (
        AdaptationResult(
            source_kind="J8F",
            source_schema=J8F_SMOKE_SCHEMA,
            source_id=run_id,
            source_artifacts=artifacts,
            source_counts=(("application_count", 12), ("checkpoint_artifact_count", 12)),
            blockers=blockers,
        ),
    )


def adapt_pf3_checkpoints(
    receipt: Mapping[str, Any],
    checkpoints: Sequence[Mapping[str, Any]],
    *,
    receipt_artifact: SourceArtifactIdentity,
    checkpoint_artifacts: Sequence[SourceArtifactIdentity],
    custody_artifacts: Sequence[SourceArtifactIdentity],
) -> AdaptationResult:
    """Validate all PF3 measured edges and expose their shared producer debt.

    Current PF3 checkpoints intentionally stop short: they preserve the count
    of changed uint8 channel values, but not the sparse change-set hash or the
    implementation hashes required by ``AppliedActionReceipt``.  The adapter
    reports those exact producer debts and never copies the physical edge into
    an RD1 hull cell.
    """

    receipt = _require_mapping(receipt, "PF3 receipt")
    _bind_json_payload(receipt, receipt_artifact, "PF3 receipt")
    if receipt.get("schema") != PF3_RECEIPT_SCHEMA:
        raise AppliedActionAdapterError("PF3 receipt schema differs")
    inventory = _require_mapping(receipt.get("inventory"), "PF3 inventory")
    custody = _require_mapping(inventory.get("candidate_checkpoint_custody"), "PF3 checkpoint custody")
    declared = custody.get("artifacts")
    if custody.get("count") != 16 or inventory.get("materialized_coordinate_count") != 16:
        raise AppliedActionAdapterError("PF3 inventory does not declare exactly 16 artifacts")
    if not isinstance(declared, list) or len(declared) != 16:
        raise AppliedActionAdapterError("PF3 checkpoint artifact inventory differs")
    if len(checkpoints) != 16 or len(checkpoint_artifacts) != 16:
        raise AppliedActionAdapterError("PF3 adapter must cover all 16 artifacts")
    declared_hashes: list[str] = []
    for index, (row, artifact) in enumerate(zip(declared, checkpoint_artifacts, strict=True)):
        row = _require_mapping(row, f"PF3 declared artifact {index}")
        if (
            artifact.path != str(row.get("path"))
            or artifact.bytes != row.get("bytes")
            or artifact.sha256 != row.get("sha256")
        ):
            raise AppliedActionAdapterError("PF3 checkpoint artifact custody differs")
        declared_hashes.append(_require_sha256(row.get("sha256"), "PF3 artifact sha256"))
        _bind_json_payload(checkpoints[index], artifact, f"PF3 checkpoint {index}")
    digest = hashlib.sha256("".join(declared_hashes).encode()).hexdigest()
    if custody.get("digest_chain_sha256") != digest:
        raise AppliedActionAdapterError("PF3 16-artifact digest chain differs")

    base_archive = _require_mapping(
        _require_mapping(receipt.get("source_custody"), "PF3 source_custody").get("base_archive"),
        "PF3 base archive",
    )
    base_sha = _require_sha256(base_archive.get("sha256"), "PF3 base archive SHA")
    if not _has_custody(
        custody_artifacts,
        sha256=base_sha,
        bytes_count=_require_int(base_archive.get("bytes"), "PF3 base archive bytes", positive=True),
        path=_require_text(base_archive.get("path"), "PF3 base archive path"),
    ):
        raise AppliedActionAdapterError("PF3 base archive bytes were not custody-verified")

    candidate_ids: set[str] = set()
    families: set[str] = set()
    joint_deltas: list[float] = []
    for index, checkpoint in enumerate(checkpoints):
        checkpoint = _require_mapping(checkpoint, f"PF3 checkpoint {index}")
        if checkpoint.get("schema") != PF3_CHECKPOINT_SCHEMA:
            raise AppliedActionAdapterError("PF3 checkpoint schema differs")
        if checkpoint.get("research_only") is not True or checkpoint.get("score_claim") is not False:
            raise AppliedActionAdapterError("PF3 false-authority contract differs")
        if checkpoint.get("evidence_axis") != LOCAL_ADVISORY_AXIS:
            raise AppliedActionAdapterError("PF3 evidence axis differs")
        candidate_id = _require_text(checkpoint.get("candidate_id"), "PF3 candidate_id")
        coordinate_id = _require_text(checkpoint.get("coordinate_id"), "PF3 coordinate_id")
        if candidate_id in candidate_ids:
            raise AppliedActionAdapterError("PF3 candidate identities must be unique")
        candidate_ids.add(candidate_id)
        parts = coordinate_id.split(".")
        if len(parts) < 2 or parts[0] != "rg3":
            raise AppliedActionAdapterError("PF3 coordinate family differs")
        families.add(parts[1])
        edges = _require_mapping(checkpoint.get("five_pf3_edges"), "PF3 five_pf3_edges")
        rate_home = _require_mapping(edges.get("dimension_rate_home"), "PF3 dimension_rate_home")
        if rate_home.get("physical_edge") != PF3_PHYSICAL_EDGE:
            raise AppliedActionAdapterError("PF3 physical edge is not V19C_BASE -> one RG3 coordinate")
        geometry = _require_mapping(
            _require_mapping(edges.get("candidate_delta"), "PF3 candidate delta").get("batch_geometry"),
            "PF3 batch geometry",
        )
        if geometry.get("authority") != "MATCHES_V19C_N600_BATCH16_ENDPOINT_GEOMETRY":
            raise AppliedActionAdapterError("PF3 n600 endpoint geometry differs")
        scorer = _require_mapping(checkpoint.get("scorer_custody"), "PF3 scorer custody")
        if scorer.get("evidence_axis") != LOCAL_ADVISORY_AXIS or scorer.get("score_claim") is not False:
            raise AppliedActionAdapterError("PF3 scorer authority differs")
        delta = _require_mapping(edges.get("candidate_delta"), "PF3 candidate delta")
        old_pose = _require_finite(delta.get("base_global_d_pose"), "PF3 base d_pose")
        new_pose = _require_finite(delta.get("candidate_global_d_pose"), "PF3 candidate d_pose")
        error_delta = _require_signed_int(delta.get("delta_global_errors"), "PF3 error delta")
        seg_term = 100.0 * error_delta / (600 * 384 * 512)
        pose_term = math.sqrt(10.0 * new_pose) - math.sqrt(10.0 * old_pose)
        joint = _require_finite(delta.get("delta_D_joint"), "PF3 joint delta", nonnegative=False)
        if not _close(_require_finite(delta.get("delta_D_seg"), "PF3 seg delta", nonnegative=False), seg_term):
            raise AppliedActionAdapterError("PF3 nonlinear seg delta differs")
        if not _close(_require_finite(delta.get("delta_D_pose"), "PF3 pose delta", nonnegative=False), pose_term):
            raise AppliedActionAdapterError("PF3 nonlinear pose delta differs")
        if not _close(joint, seg_term + pose_term):
            raise AppliedActionAdapterError("PF3 nonlinear joint delta differs")
        joint_deltas.append(joint)
        candidate_archive = _require_mapping(
            _require_mapping(edges.get("receiver_object_builder"), "PF3 receiver builder").get("candidate_archive"),
            "PF3 candidate archive",
        )
        if not _has_custody(
            custody_artifacts,
            sha256=_require_sha256(candidate_archive.get("sha256"), "PF3 candidate archive SHA"),
            bytes_count=_require_int(candidate_archive.get("bytes"), "PF3 candidate archive bytes", positive=True),
            path=_require_text(candidate_archive.get("path"), "PF3 candidate archive path"),
        ):
            raise AppliedActionAdapterError("PF3 candidate archive bytes were not custody-verified")
        realized = _require_mapping(edges.get("realized_uint8_quantum"), "PF3 realized uint8")
        _require_int(realized.get("changed_channel_values"), "PF3 changed values", positive=True)
    extrema = _require_mapping(inventory.get("measurement_extrema"), "PF3 extrema")
    if (
        not all(value > 0.0 for value in joint_deltas)
        or extrema.get("all_measured_coordinate_edges_worsen_joint_D") is not True
    ):
        raise AppliedActionAdapterError("PF3 measured family is not uniformly uphill")
    if not _close(min(joint_deltas), float(extrema.get("delta_D_joint_min"))) or not _close(
        max(joint_deltas), float(extrema.get("delta_D_joint_max"))
    ):
        raise AppliedActionAdapterError("PF3 family extrema differ")
    if families != {"class_birth", "finer_event", "fisher_stratum"}:
        raise AppliedActionAdapterError("PF3 coordinate-family inventory differs")
    blockers = (
        AdapterBlocker(
            code="PF3_APPLICATION_OPERATOR_VERSION_ABSENT",
            source_key="five_pf3_edges.receiver_object_builder.pipeline",
            owed_field="application_operator_version",
            detail=(
                "Pipeline symbols and a source-coordinate checkpoint hash do not bind the exact "
                "application implementation bytes that built the candidate."
            ),
        ),
        AdapterBlocker(
            code="PF3_RECEIVER_SHA256_ABSENT",
            source_key="five_pf3_edges.receiver_object_builder",
            owed_field="receiver_sha256",
            detail=(
                "The archive and frozen scorer modules are SHA-bound, but the complete "
                "builder/parse-back/R receiver chain has no single implementation identity."
            ),
        ),
        AdapterBlocker(
            code="PF3_CHANGED_UINT8_SHA256_ABSENT",
            source_key="five_pf3_edges.realized_uint8_quantum.changed_channel_values",
            owed_field="changed_uint8_sha256",
            detail=(
                "Producer must preserve the exact sparse uint8 index/value identity for this "
                "V19C_BASE -> one RG3 coordinate edge; candidate archive SHA is not a substitute."
            ),
        ),
    )
    artifacts = _normalize_artifacts((receipt_artifact, *checkpoint_artifacts, *custody_artifacts))
    return AdaptationResult(
        source_kind="PF3",
        source_schema=PF3_CHECKPOINT_SCHEMA,
        source_id=_require_text(checkpoints[0].get("run_id"), "PF3 run_id"),
        source_artifacts=artifacts,
        source_counts=(
            ("candidate_artifact_count", 16),
            ("coordinate_family_count", len(families)),
            ("uphill_edge_count", len(joint_deltas)),
        ),
        blockers=blockers,
    )


def adapt_j12_receipt(
    receipt: Mapping[str, Any],
    pricing_receipt: Mapping[str, Any],
    *,
    receipt_artifact: SourceArtifactIdentity,
    pricing_artifact: SourceArtifactIdentity,
    custody_artifacts: Sequence[SourceArtifactIdentity],
) -> AdaptationResult:
    """Expose the exact J12 custody fields still owed by its producer."""

    receipt = _require_mapping(receipt, "J12 receipt")
    pricing_receipt = _require_mapping(pricing_receipt, "J12 pricing receipt")
    _bind_json_payload(receipt, receipt_artifact, "J12 receipt")
    _bind_json_payload(pricing_receipt, pricing_artifact, "J12 pricing receipt")
    if receipt.get("schema") != J12_RECEIPT_SCHEMA:
        raise AppliedActionAdapterError("J12 receipt schema differs")
    if receipt.get("research_only") is not True or receipt.get("score_claim") is not False:
        raise AppliedActionAdapterError("J12 false-authority contract differs")
    source = _require_mapping(receipt.get("source"), "J12 source")
    geometry = _require_mapping(receipt.get("measurement_geometry"), "J12 geometry")
    if geometry.get("pair_count") != 600 or receipt.get("evidence_axis") != LOCAL_ADVISORY_AXIS:
        raise AppliedActionAdapterError("J12 measurement is not n600 on its declared axis")
    step16 = _require_mapping(
        _require_mapping(receipt.get("pc1_adapter"), "J12 pc1_adapter").get("step16"),
        "J12 pc1 step16",
    )
    source_sha = _require_sha256(source.get("archive_sha256"), "J12 source archive SHA")
    step16_sha = _require_sha256(step16.get("archive_sha256"), "J12 step16 archive SHA")
    source_bytes = _require_int(source.get("archive_bytes"), "J12 source bytes", positive=True)
    _require_int(step16.get("archive_bytes"), "J12 step16 bytes", positive=True)
    pc1 = _require_mapping(receipt.get("pc1_adapter"), "J12 pc1 adapter")
    if (
        pc1.get("active_zero_archive_sha256") != source_sha
        or pc1.get("active_zero_archive_bytes") != source_bytes
        or pc1.get("active_zero_byte_identical") is not True
    ):
        raise AppliedActionAdapterError("J12 active-zero source identity differs")
    if pricing_receipt.get("schema") != "ddm_j12_decomposition_pricing.v1":
        raise AppliedActionAdapterError("J12 pricing schema differs")
    if any(
        pricing_receipt.get(key) is not expected
        for key, expected in (
            ("research_only", True),
            ("score_claim", False),
            ("promotion_eligible", False),
        )
    ):
        raise AppliedActionAdapterError("J12 pricing false-authority contract differs")
    counts = _require_mapping(pricing_receipt.get("counts"), "J12 pricing counts")
    if counts.get("singles_total") != 16 or counts.get("composites_total") != 8:
        raise AppliedActionAdapterError("J12 pricing family counts differ")
    pricing_pc1 = _require_mapping(pricing_receipt.get("pc1_adapter"), "J12 pricing pc1")
    if (
        pricing_pc1.get("active_zero_archive_sha256") != source_sha
        or pricing_pc1.get("active_zero_archive_bytes") != source_bytes
        or pricing_pc1.get("archive_byte_identity") is not True
    ):
        raise AppliedActionAdapterError("J12 pricing active-zero identity differs")
    rows = pricing_pc1.get("local_pose_descent_remeasurement")
    if not isinstance(rows, list) or [row.get("accepted_step") for row in rows] != [0, 8, 16]:
        raise AppliedActionAdapterError("J12 local remeasurement sequence differs")
    for row, name in zip(rows, ("source", "step8", "step16"), strict=True):
        compact = source if name == "source" else _require_mapping(pc1.get(name), f"J12 {name}")
        if any(row.get(key) != compact.get(key) for key in ("archive_bytes", "archive_sha256", "d_seg", "d_pose")):
            raise AppliedActionAdapterError(f"J12 compact {name} foreign keys differ from pricing")
        old_d_seg = _require_finite(source.get("d_seg"), "J12 source d_seg")
        old_d_pose = _require_finite(source.get("d_pose"), "J12 source d_pose")
        new_d_seg = _require_finite(row.get("d_seg"), f"J12 {name} d_seg")
        new_d_pose = _require_finite(row.get("d_pose"), f"J12 {name} d_pose")
        new_bytes = _require_int(row.get("archive_bytes"), f"J12 {name} bytes", positive=True)
        expected_delta = contest_score(new_d_seg, new_d_pose, new_bytes) - contest_score(
            old_d_seg, old_d_pose, source_bytes
        )
        priced = _require_mapping(row.get("pure_priced_from_rehomed_step0"), f"J12 {name} priced delta")
        if not _close(
            _require_finite(priced.get("joint_delta"), f"J12 {name} joint delta", nonnegative=False),
            expected_delta,
        ):
            raise AppliedActionAdapterError(f"J12 {name} nonlinear joint delta differs")
        if name != "source" and not _close(
            _require_finite(compact.get("joint_delta"), f"J12 compact {name} delta", nonnegative=False),
            expected_delta,
        ):
            raise AppliedActionAdapterError(f"J12 compact {name} delta differs")
    step8_sha = _require_sha256(
        _require_mapping(pc1.get("step8"), "J12 step8").get("archive_sha256"),
        "J12 step8 archive SHA",
    )
    endpoints = (
        (source_sha, source_bytes),
        (
            step8_sha,
            _require_int(
                _require_mapping(pc1.get("step8"), "J12 step8").get("archive_bytes"),
                "J12 step8 archive bytes",
                positive=True,
            ),
        ),
        (
            step16_sha,
            _require_int(step16.get("archive_bytes"), "J12 step16 archive bytes", positive=True),
        ),
    )
    if not all(_has_custody(custody_artifacts, sha256=sha, bytes_count=size) for sha, size in endpoints):
        raise AppliedActionAdapterError("J12 endpoint archive bytes were not custody-verified")
    reseal = _require_mapping(receipt.get("reseal_state"), "J12 reseal state")
    if reseal.get("status") != "PREPARED_REVIEW_REQUIRED":
        raise AppliedActionAdapterError("J12 reseal blocker differs")
    warning = _require_mapping(receipt.get("pc1_receiver_numerical_warning"), "J12 warning")
    if warning.get("observed") is not True or warning.get("suppressed") is not False:
        raise AppliedActionAdapterError("J12 numerical warning custody differs")
    if set(warning.get("classes", ())) != {"divide_by_zero", "overflow", "invalid_value"}:
        raise AppliedActionAdapterError("J12 numerical warning classes differ")
    blockers = (
        AdapterBlocker(
            code="J12_APPLICATION_OPERATOR_VERSION_ABSENT",
            source_key="pc1_adapter.receiver_equation",
            owed_field="application_operator_version",
            detail="Bind the exact PC1 apply/parse/receive implementation source SHA-256.",
        ),
        AdapterBlocker(
            code="J12_RECEIVER_SHA256_ABSENT",
            source_key="measurement_geometry.receiver_chain",
            owed_field="receiver_sha256",
            detail="Bind the exact receiver/R implementation identity, not only its prose chain.",
        ),
        AdapterBlocker(
            code="J12_CHANGED_UINT8_IDENTITY_ABSENT",
            source_key="pc1_adapter.step16",
            owed_field="changed_uint8_count,changed_uint8_sha256",
            detail="Preserve the final source->step16 sparse uint8 change count and identity.",
        ),
        AdapterBlocker(
            code="J12_SINGLE_LAWFUL_BYTE_HOME_ABSENT",
            source_key="pc1_adapter.step16.archive_bytes",
            owed_field="stream_home",
            detail=(
                "Name one typed stream/layer home with coder owner and before/after bytes; "
                "whole-archive byte deltas alone do not identify the PC1 home."
            ),
        ),
        AdapterBlocker(
            code="J12_MAIN_RESEAL_REVIEW_REQUIRED",
            source_key="reseal_state",
            owed_field="merged-main reseal and review",
            detail="The producer explicitly leaves the merged-main SHA reseal and MAIN review outstanding.",
        ),
        AdapterBlocker(
            code="J12_PC1_NUMERICAL_WARNING_UNRESOLVED",
            source_key="pc1_receiver_numerical_warning",
            owed_field="fail-closed sanitize-or-reject disposition",
            detail="Observed divide-by-zero, overflow, and invalid-value warnings remain unsuppressed.",
        ),
    )
    artifacts = _normalize_artifacts((receipt_artifact, pricing_artifact, *custody_artifacts))
    return AdaptationResult(
        source_kind="J12",
        source_schema=J12_RECEIPT_SCHEMA,
        source_id=_require_text(receipt.get("run_id"), "J12 run_id"),
        source_artifacts=artifacts,
        source_counts=(
            ("composite_count", 8),
            ("sealed_proposal_count", 4),
            ("single_count", 16),
        ),
        blockers=blockers,
    )


def build_adapter_manifest(results: Sequence[AdaptationResult]) -> dict[str, Any]:
    """Build and parse-check a deterministic research-only adapter manifest."""

    rows = [result.as_dict() for result in results]
    for row in rows:
        receipt = row.get("receipt")
        if receipt is not None:
            AppliedActionReceipt.from_dict(_require_mapping(receipt, "manifest receipt"))
    payload: dict[str, Any] = {
        "schema": ADAPTER_MANIFEST_SCHEMA,
        "results": rows,
        "receipt_count": sum(result.ok for result in results),
        "blocked_source_count": sum(not result.ok for result in results),
        "source_artifact_count": sum(len(result.source_artifacts) for result in results),
        "research_only": True,
        "promotion_eligible": False,
        "score_claim": False,
    }
    payload["content_sha256"] = _canonical_sha256(payload)
    validate_adapter_manifest(payload)
    return payload


def validate_adapter_manifest(manifest: Mapping[str, Any]) -> None:
    """Strictly validate counts, custody digests, authority, and self-hash."""

    manifest = _require_mapping(manifest, "adapter manifest")
    if set(manifest) != _MANIFEST_KEYS or manifest.get("schema") != ADAPTER_MANIFEST_SCHEMA:
        raise AppliedActionAdapterError("adapter manifest keys/schema differ")
    if any(
        manifest.get(key) is not expected
        for key, expected in (
            ("research_only", True),
            ("promotion_eligible", False),
            ("score_claim", False),
        )
    ):
        raise AppliedActionAdapterError("adapter manifest false-authority contract differs")
    declared_hash = _require_sha256(manifest.get("content_sha256"), "manifest content_sha256")
    unhashed = dict(manifest)
    unhashed.pop("content_sha256")
    if _canonical_sha256(unhashed) != declared_hash:
        raise AppliedActionAdapterError("adapter manifest content hash differs")
    rows = manifest.get("results")
    if not isinstance(rows, list):
        raise AppliedActionAdapterError("adapter manifest results must be an array")
    receipts = blocked = artifact_total = 0
    source_kinds: set[str] = set()
    for index, row in enumerate(rows):
        row = _require_mapping(row, f"adapter result {index}")
        if set(row) != _RESULT_KEYS or row.get("schema") != ADAPTER_RESULT_SCHEMA:
            raise AppliedActionAdapterError("adapter result keys/schema differ")
        for key in ("source_kind", "source_schema", "source_id"):
            _require_text(row.get(key), f"adapter result {key}")
        source_kind = str(row["source_kind"])
        if source_kind in source_kinds or source_kind not in _EXPECTED_SOURCE_COUNTS:
            raise AppliedActionAdapterError("adapter source kind inventory differs")
        source_kinds.add(source_kind)
        if any(
            row.get(key) is not expected
            for key, expected in (
                ("research_only", True),
                ("promotion_eligible", False),
                ("score_claim", False),
            )
        ):
            raise AppliedActionAdapterError("adapter result false-authority contract differs")
        ok = row.get("ok")
        blockers = row.get("blockers")
        receipt = row.get("receipt")
        if not isinstance(blockers, list) or type(ok) is not bool:
            raise AppliedActionAdapterError("adapter result outcome shape differs")
        if ok is True:
            if blockers or not isinstance(receipt, Mapping):
                raise AppliedActionAdapterError("successful adapter result differs")
            AppliedActionReceipt.from_dict(receipt)
            receipts += 1
        else:
            if not blockers or receipt is not None:
                raise AppliedActionAdapterError("blocked adapter result differs")
            for blocker in blockers:
                blocker = _require_mapping(blocker, "adapter blocker")
                if set(blocker) != {"code", "source_key", "owed_field", "detail"}:
                    raise AppliedActionAdapterError("adapter blocker keys differ")
                for key in blocker:
                    _require_text(blocker[key], f"adapter blocker {key}")
            blocked += 1
        artifacts = row.get("source_artifacts")
        if not isinstance(artifacts, list) or not artifacts:
            raise AppliedActionAdapterError("adapter source artifacts differ")
        if row.get("source_artifact_count") != len(artifacts):
            raise AppliedActionAdapterError("adapter source artifact count differs")
        if row.get("source_artifact_digest_sha256") != _canonical_sha256(artifacts):
            raise AppliedActionAdapterError("adapter source artifact digest differs")
        paths: set[str] = set()
        identities: list[SourceArtifactIdentity] = []
        for artifact in artifacts:
            artifact = _require_mapping(artifact, "source artifact")
            if set(artifact) != {"path", "bytes", "sha256", "role", "json_content_sha256"}:
                raise AppliedActionAdapterError("source artifact keys differ")
            identity = SourceArtifactIdentity(**artifact)
            if identity.path in paths:
                raise AppliedActionAdapterError("source artifact path duplicated")
            paths.add(identity.path)
            identities.append(identity)
        if identities != sorted(identities, key=lambda item: (item.path, item.role)):
            raise AppliedActionAdapterError("source artifacts are not canonically ordered")
        artifact_total += len(artifacts)
        counts = row.get("source_counts")
        if not isinstance(counts, Mapping) or not counts:
            raise AppliedActionAdapterError("adapter source counts differ")
        for key, value in counts.items():
            _require_text(key, "source count key")
            _require_int(value, f"source_counts.{key}")
        if dict(counts) != dict(_EXPECTED_SOURCE_COUNTS[source_kind]):
            raise AppliedActionAdapterError("adapter source semantic counts differ")
    if manifest.get("receipt_count") != receipts or manifest.get("blocked_source_count") != blocked:
        raise AppliedActionAdapterError("adapter manifest result counts differ")
    if manifest.get("source_artifact_count") != artifact_total:
        raise AppliedActionAdapterError("adapter manifest artifact count differs")
    if source_kinds != set(_EXPECTED_SOURCE_COUNTS):
        raise AppliedActionAdapterError("adapter source kind inventory differs")


def load_json(path: str | Path) -> Mapping[str, Any]:
    """Load one JSON object without accepting arrays or scalar coercions."""

    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    return _require_mapping(payload, str(source))


__all__ = [
    "ADAPTER_MANIFEST_SCHEMA",
    "ADAPTER_RESULT_SCHEMA",
    "AdaptationResult",
    "AdapterBlocker",
    "AppliedActionAdapterError",
    "SourceArtifactIdentity",
    "adapt_j8f_checkpoints",
    "adapt_j12_receipt",
    "adapt_pf3_checkpoints",
    "build_adapter_manifest",
    "canonical_json_sha256",
    "load_json",
    "load_json_artifact",
    "validate_adapter_manifest",
    "verify_source_artifact",
]
