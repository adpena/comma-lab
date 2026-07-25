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
            raise AppliedActionAdapterError(
                "adaptation result must carry exactly one of receipt or blockers"
            )
        if self.receipt is not None and self.receipt.schema != APPLIED_ACTION_RECEIPT_SCHEMA:
            raise AppliedActionAdapterError("adapter emitted a foreign receipt schema")

    @property
    def ok(self) -> bool:
        return self.receipt is not None

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": ADAPTER_RESULT_SCHEMA,
            "source_kind": self.source_kind,
            "source_schema": self.source_schema,
            "source_id": self.source_id,
            "ok": self.ok,
            "receipt": self.receipt.as_dict() if self.receipt is not None else None,
            "blockers": [blocker.as_dict() for blocker in self.blockers],
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
    if smoke.get("schema") != J8F_SMOKE_SCHEMA:
        raise AppliedActionAdapterError("J8F smoke schema differs")
    if smoke.get("research_only") is not True or smoke.get("score_claim") is not False:
        raise AppliedActionAdapterError("J8F false-authority contract differs")
    run_id = _require_text(smoke.get("run_id"), "J8F run_id")
    _require_text(smoke.get("evidence_axis"), "J8F evidence_axis")
    _j8f_operator_sha(config)
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
        raise AppliedActionAdapterError(
            f"J8F checkpoint sequence differs: {observed_steps} != {expected_steps}"
        )
    for ordinal, checkpoint in enumerate(ordered):
        if checkpoint.get("schema") != J8F_CHECKPOINT_SCHEMA:
            raise AppliedActionAdapterError("J8F checkpoint schema differs")
        if checkpoint.get("score_claim") is not False:
            raise AppliedActionAdapterError("J8F checkpoint score_claim differs")
        applications = checkpoint.get("application_receipts")
        if not isinstance(applications, list) or len(applications) != ordinal + 1:
            raise AppliedActionAdapterError(
                "J8F checkpoint must preserve the complete cumulative application prefix"
            )
        if [row.get("step_index") for row in applications if isinstance(row, Mapping)] != list(
            range(ordinal + 1)
        ):
            raise AppliedActionAdapterError("J8F cumulative application prefix differs")
        application = _require_mapping(applications[-1], "J8F application")
        if application.get("schema") != J8F_APPLICATION_SCHEMA:
            raise AppliedActionAdapterError("J8F application schema differs")
        projected = _require_mapping(
            application.get("projected_application"), "J8F projected_application"
        )
        proposal = _require_mapping(application.get("proposal"), "J8F proposal")
        support = _require_mapping(proposal.get("support_footprint"), "J8F support_footprint")
        trust = _require_mapping(application.get("trust_region"), "J8F trust_region")
        projected_state = _require_mapping(checkpoint.get("projected_state"), "J8F projected_state")

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
        _require_int(
            projected.get("changed_channel_values"), "J8F changed_channel_values", positive=True
        )
        _require_sha256(
            projected.get("delta_sha256_int64_indices_int16_values"),
            "J8F changed uint8 sparse identity",
        )
        _require_sha256(
            support.get("stem_block_indices_sha256_uint32le"), "J8F support SHA"
        )
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
        _require_sha256(
            projected_state.get("theta_sha256_float32le"), "J8F projected theta SHA"
        )

    final_arm = _require_mapping(
        smoke.get("range_gauge_projected_arm"), "J8F range_gauge_projected_arm"
    )
    final_archive = _require_mapping(final_arm.get("archive"), "J8F final archive")
    final_verdict = _require_mapping(final_arm.get("verdict"), "J8F final verdict")
    step4 = _require_mapping(smoke.get("step4"), "J8F step4")
    reference = _require_mapping(step4.get("reference"), "J8F step4.reference")
    final_archive_sha = _require_sha256(final_archive.get("sha256"), "J8F final archive SHA")
    if final_archive_sha != prior_archive_sha:
        raise AppliedActionAdapterError("J8F final archive is not the last projected checkpoint")
    final_archive_bytes = _require_int(
        final_archive.get("bytes"), "J8F final archive bytes", positive=True
    )
    if final_archive_bytes != prior_archive_bytes:
        raise AppliedActionAdapterError("J8F final archive bytes differ from checkpoint chain")
    if final_archive.get("parseback_exact") is not True:
        raise AppliedActionAdapterError("J8F final archive parseback is not exact")

    old_d_seg = float(reference["d_seg"])
    old_d_pose = float(reference["d_pose"])
    new_d_seg = float(final_verdict["d_seg"])
    new_d_pose = float(final_verdict["d_pose"])
    if not all(math.isfinite(value) for value in (old_d_seg, old_d_pose, new_d_seg, new_d_pose)):
        raise AppliedActionAdapterError("J8F final scorer transition is not finite")
    delta_score = contest_score(new_d_seg, new_d_pose, final_archive_bytes) - contest_score(
        old_d_seg, old_d_pose, base_archive_bytes
    )
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
    )
    return (
        AdaptationResult(
            source_kind="J8F",
            source_schema=J8F_SMOKE_SCHEMA,
            source_id=run_id,
            blockers=blockers,
        ),
    )


def adapt_pf3_checkpoint(
    receipt: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    *,
    source_id: str,
) -> AdaptationResult:
    """Return a PF3 receipt only when its physical edge has complete identity.

    Current PF3 checkpoints intentionally stop short: they preserve the count
    of changed uint8 channel values, but not the sparse change-set hash or the
    implementation hashes required by ``AppliedActionReceipt``.  The adapter
    reports those exact producer debts and never copies the physical edge into
    an RD1 hull cell.
    """

    receipt = _require_mapping(receipt, "PF3 receipt")
    checkpoint = _require_mapping(checkpoint, "PF3 checkpoint")
    if receipt.get("schema") != PF3_RECEIPT_SCHEMA:
        raise AppliedActionAdapterError("PF3 receipt schema differs")
    if checkpoint.get("schema") != PF3_CHECKPOINT_SCHEMA:
        raise AppliedActionAdapterError("PF3 checkpoint schema differs")
    if checkpoint.get("research_only") is not True or checkpoint.get("score_claim") is not False:
        raise AppliedActionAdapterError("PF3 false-authority contract differs")
    edges = _require_mapping(checkpoint.get("five_pf3_edges"), "PF3 five_pf3_edges")
    rate_home = _require_mapping(edges.get("dimension_rate_home"), "PF3 dimension_rate_home")
    if rate_home.get("physical_edge") != PF3_PHYSICAL_EDGE:
        raise AppliedActionAdapterError("PF3 physical edge is not V19C_BASE -> one RG3 coordinate")
    _require_sha256(
        _require_mapping(
            _require_mapping(edges.get("receiver_object_builder"), "PF3 receiver builder").get(
                "candidate_archive"
            ),
            "PF3 candidate archive",
        ).get("sha256"),
        "PF3 candidate archive SHA",
    )
    _require_sha256(
        _require_mapping(receipt.get("source_custody"), "PF3 source_custody")
        .get("base_archive", {})
        .get("sha256"),
        "PF3 base archive SHA",
    )
    realized = _require_mapping(edges.get("realized_uint8_quantum"), "PF3 realized_uint8_quantum")
    _require_int(realized.get("changed_channel_values"), "PF3 changed_channel_values", positive=True)
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
    return AdaptationResult(
        source_kind="PF3",
        source_schema=PF3_CHECKPOINT_SCHEMA,
        source_id=source_id,
        blockers=blockers,
    )


def adapt_j12_receipt(receipt: Mapping[str, Any], *, source_id: str) -> AdaptationResult:
    """Expose the exact J12 custody fields still owed by its producer."""

    receipt = _require_mapping(receipt, "J12 receipt")
    if receipt.get("schema") != J12_RECEIPT_SCHEMA:
        raise AppliedActionAdapterError("J12 receipt schema differs")
    if receipt.get("research_only") is not True or receipt.get("score_claim") is not False:
        raise AppliedActionAdapterError("J12 false-authority contract differs")
    source = _require_mapping(receipt.get("source"), "J12 source")
    step16 = _require_mapping(
        _require_mapping(receipt.get("pc1_adapter"), "J12 pc1_adapter").get("step16"),
        "J12 pc1 step16",
    )
    _require_sha256(source.get("archive_sha256"), "J12 source archive SHA")
    _require_sha256(step16.get("archive_sha256"), "J12 step16 archive SHA")
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
    )
    return AdaptationResult(
        source_kind="J12",
        source_schema=J12_RECEIPT_SCHEMA,
        source_id=source_id,
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
        "research_only": True,
        "promotion_eligible": False,
        "score_claim": False,
    }
    payload["content_sha256"] = _canonical_sha256(payload)
    return payload


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
    "adapt_j8f_checkpoints",
    "adapt_j12_receipt",
    "adapt_pf3_checkpoint",
    "build_adapter_manifest",
    "load_json",
]
